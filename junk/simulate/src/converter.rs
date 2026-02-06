use crate::ass_parser::AssDocument;
use llmsub::{LlmsubDocument, Character, Dialogue};
use std::collections::HashMap;

pub struct AssToLlmsubConverter {
    /// Threshold for considering a pause between dialogues as a scene break (in seconds)
    break_threshold_seconds: f64,
    /// Whether to group consecutive dialogues by the same character
    group_same_character: bool,
    /// Whether to add character descriptions based on style information
    use_style_descriptions: bool,
}

impl Default for AssToLlmsubConverter {
    fn default() -> Self {
        Self {
            break_threshold_seconds: 10.0, // 10 second pause = scene break
            group_same_character: true,
            use_style_descriptions: true,
        }
    }
}

impl AssToLlmsubConverter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_break_threshold(mut self, seconds: f64) -> Self {
        self.break_threshold_seconds = seconds;
        self
    }

    pub fn with_grouping(mut self, group: bool) -> Self {
        self.group_same_character = group;
        self
    }

    pub fn with_style_descriptions(mut self, use_descriptions: bool) -> Self {
        self.use_style_descriptions = use_descriptions;
        self
    }

    /// Convert an ASS document to LLMSUB format
    pub fn convert(&self, ass_doc: &AssDocument) -> Result<LlmsubDocument, String> {
        let mut llmsub_doc = LlmsubDocument::new();
        
        // Extract unique characters first
        let characters = self.extract_characters(ass_doc);
        
        // Add character definitions to the document
        for character in characters.values() {
            llmsub_doc.add_character(character.clone());
        }
        
        // Convert dialogues and add scene breaks
        self.convert_dialogues(ass_doc, &mut llmsub_doc, &characters)?;
        
        Ok(llmsub_doc)
    }

    /// Extract unique characters from the ASS document
    fn extract_characters(&self, ass_doc: &AssDocument) -> HashMap<String, Character> {
        let mut characters = HashMap::new();
        let mut character_styles = HashMap::new();
        
        // First pass: collect all unique character names and their styles
        for dialogue in &ass_doc.dialogues {
            let speaker = ass_doc.extract_speaker_name(dialogue);
            if !characters.contains_key(&speaker) {
                // Store the style for potential description generation
                character_styles.insert(speaker.clone(), dialogue.style.clone());
                
                let description = if self.use_style_descriptions {
                    self.generate_character_description(&speaker, &dialogue.style, ass_doc)
                } else {
                    format!("Character from subtitle file")
                };
                
                characters.insert(speaker.clone(), Character {
                    name: speaker,
                    description,
                });
            }
        }
        
        characters
    }

    /// Generate a character description based on style and context
    fn generate_character_description(&self, name: &str, style_name: &str, ass_doc: &AssDocument) -> String {
        let mut description_parts = Vec::new();
        
        // Add basic description
        description_parts.push(format!("Character appearing in the dialogue"));
        
        // Add style-based information if available
        if let Some(style) = ass_doc.styles.get(style_name) {
            if style.name != "Default" {
                description_parts.push(format!("Uses '{}' text style", style.name));
            }
            
            // Add font/formatting info if it seems significant
            if style.bold {
                description_parts.push("Displayed in bold text".to_string());
            }
            if style.italic {
                description_parts.push("Displayed in italic text".to_string());
            }
        }
        
        // Count dialogues for this character to infer importance
        let dialogue_count = ass_doc.dialogues.iter()
            .filter(|d| ass_doc.extract_speaker_name(d) == name)
            .count();
            
        if dialogue_count > 10 {
            description_parts.push("Major character with many lines".to_string());
        } else if dialogue_count > 3 {
            description_parts.push("Supporting character".to_string());
        } else {
            description_parts.push("Minor character".to_string());
        }
        
        description_parts.join(". ")
    }

    /// Convert dialogues and handle scene breaks
    fn convert_dialogues(
        &self, 
        ass_doc: &AssDocument, 
        llmsub_doc: &mut LlmsubDocument,
        _characters: &HashMap<String, Character>
    ) -> Result<(), String> {
        if ass_doc.dialogues.is_empty() {
            return Ok(());
        }

        let mut prev_end_time: Option<f64> = None;
        let mut current_character = String::new();
        let mut accumulated_text = Vec::new();

        for dialogue in &ass_doc.dialogues {
            let speaker = ass_doc.extract_speaker_name(dialogue);
            let clean_text = AssDocument::clean_text(&dialogue.text);
            
            // Skip empty dialogues
            if clean_text.trim().is_empty() {
                continue;
            }

            // Parse timing to detect scene breaks
            let start_seconds = self.parse_time_to_seconds(&dialogue.start_time);
            
            // Check for scene break based on timing gap
            if let (Some(prev_end), Some(current_start)) = (prev_end_time, start_seconds) {
                if current_start - prev_end > self.break_threshold_seconds {
                    // Flush any accumulated dialogue first
                    if !accumulated_text.is_empty() {
                        self.add_dialogue_to_doc(llmsub_doc, &current_character, &accumulated_text)?;
                        accumulated_text.clear();
                    }
                    
                    llmsub_doc.add_break();
                    current_character.clear();
                }
            }

            // Handle character grouping
            if self.group_same_character && speaker == current_character && !accumulated_text.is_empty() {
                // Same character continuing, accumulate the text
                accumulated_text.push(clean_text);
            } else {
                // Different character or first dialogue
                
                // Flush previous character's accumulated text
                if !accumulated_text.is_empty() {
                    self.add_dialogue_to_doc(llmsub_doc, &current_character, &accumulated_text)?;
                    accumulated_text.clear();
                }
                
                // Start new character dialogue
                current_character = speaker;
                accumulated_text.push(clean_text);
            }

            // Update timing for next iteration
            if let Some(end_seconds) = self.parse_time_to_seconds(&dialogue.end_time) {
                prev_end_time = Some(end_seconds);
            }
        }

        // Don't forget to add the last accumulated dialogue
        if !accumulated_text.is_empty() {
            self.add_dialogue_to_doc(llmsub_doc, &current_character, &accumulated_text)?;
        }

        Ok(())
    }

    /// Add accumulated dialogue text to the document
    fn add_dialogue_to_doc(
        &self,
        llmsub_doc: &mut LlmsubDocument,
        character: &str,
        text_lines: &[String]
    ) -> Result<(), String> {
        if character.is_empty() || text_lines.is_empty() {
            return Ok(());
        }

        let content = if text_lines.len() == 1 {
            text_lines[0].clone()
        } else {
            text_lines.join("\n")
        };

        llmsub_doc.add_dialogue(Dialogue {
            character: character.to_string(),
            content,
        });

        Ok(())
    }

    /// Parse ASS time format (H:MM:SS.CC) to seconds
    fn parse_time_to_seconds(&self, time_str: &str) -> Option<f64> {
        // ASS time format: H:MM:SS.CC (hours:minutes:seconds.centiseconds)
        let parts: Vec<&str> = time_str.split(':').collect();
        if parts.len() != 3 {
            return None;
        }

        let hours: f64 = parts[0].parse().ok()?;
        let minutes: f64 = parts[1].parse().ok()?;
        
        // Handle seconds.centiseconds
        let seconds_parts: Vec<&str> = parts[2].split('.').collect();
        let seconds: f64 = seconds_parts[0].parse().ok()?;
        let centiseconds: f64 = if seconds_parts.len() > 1 {
            seconds_parts[1].parse().ok().unwrap_or(0.0) / 100.0
        } else {
            0.0
        };

        Some(hours * 3600.0 + minutes * 60.0 + seconds + centiseconds)
    }
}

#[cfg(test)]
/// Convenience function for simple conversion - only used in tests
pub fn convert_ass_to_llmsub(ass_content: &str) -> Result<String, String> {
    let ass_doc = AssDocument::parse(ass_content)?;
    let converter = AssToLlmsubConverter::new();
    let llmsub_doc = converter.convert(&ass_doc)?;
    Ok(llmsub_doc.serialize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_parsing() {
        let converter = AssToLlmsubConverter::new();
        
        assert_eq!(converter.parse_time_to_seconds("0:00:00.00"), Some(0.0));
        assert_eq!(converter.parse_time_to_seconds("0:01:30.50"), Some(90.5));
        assert_eq!(converter.parse_time_to_seconds("1:23:45.67"), Some(5025.67));
    }

    #[test]
    fn test_simple_conversion() {
        let ass_content = r#"[Script Info]
Title: Test

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Default,Alice,0,0,0,,Hello world!
Dialogue: 0,0:00:05.00,0:00:10.00,Default,Bob,0,0,0,,Hi Alice!"#;

        let result = convert_ass_to_llmsub(ass_content);
        assert!(result.is_ok());
        
        let llmsub = result.unwrap();
        assert!(llmsub.contains("character Alice:"));
        assert!(llmsub.contains("character Bob:"));
        assert!(llmsub.contains("Alice said:"));
        assert!(llmsub.contains("Bob said:"));
        assert!(llmsub.contains("Hello world!"));
        assert!(llmsub.contains("Hi Alice!"));
    }

    #[test]
    fn test_scene_break_detection() {
        let ass_content = r#"[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Default,Alice,0,0,0,,First scene
Dialogue: 0,0:00:20.00,0:00:25.00,Default,Bob,0,0,0,,After long pause"#;

        let ass_doc = AssDocument::parse(ass_content).unwrap();
        let converter = AssToLlmsubConverter::new().with_break_threshold(10.0);
        let llmsub_doc = converter.convert(&ass_doc).unwrap();
        
        // Should have a break between the dialogues due to the time gap
        assert!(llmsub_doc.has_breaks());
    }
}