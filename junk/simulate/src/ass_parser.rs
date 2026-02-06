use regex::Regex;
use std::collections::HashMap;

/// Represents a dialogue line from an ASS file
#[derive(Debug, Clone)]
pub struct AssDialogue {
    pub layer: i32,
    pub start_time: String,
    pub end_time: String,
    pub style: String,
    pub name: String,
    pub margin_l: i32,
    pub margin_r: i32,
    pub margin_v: i32,
    pub effect: String,
    pub text: String,
}

/// Represents a style definition from an ASS file
#[derive(Debug, Clone)]
pub struct AssStyle {
    pub name: String,
    pub fontname: String,
    pub fontsize: i32,
    pub primary_colour: String,
    pub secondary_colour: String,
    pub outline_colour: String,
    pub back_colour: String,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikeout: bool,
    pub scale_x: f64,
    pub scale_y: f64,
    pub spacing: f64,
    pub angle: f64,
    pub border_style: i32,
    pub outline: f64,
    pub shadow: f64,
    pub alignment: i32,
    pub margin_l: i32,
    pub margin_r: i32,
    pub margin_v: i32,
    pub encoding: i32,
}

/// Main ASS document parser
#[derive(Debug)]
pub struct AssDocument {
    pub dialogues: Vec<AssDialogue>,
    pub styles: HashMap<String, AssStyle>,
    pub script_info: HashMap<String, String>,
}

impl AssDocument {
    pub fn new() -> Self {
        Self {
            dialogues: Vec::new(),
            styles: HashMap::new(),
            script_info: HashMap::new(),
        }
    }

    /// Parse an ASS file content
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut document = Self::new();
        let mut current_section = String::new();
        let mut dialogue_format: Option<Vec<String>> = None;
        let mut style_format: Option<Vec<String>> = None;

        for line in content.lines() {
            let line = line.trim();
            
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(';') {
                continue;
            }

            // Check for section headers
            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len()-1].to_string();
                continue;
            }

            match current_section.as_str() {
                "Script Info" => {
                    if let Some((key, value)) = line.split_once(':') {
                        document.script_info.insert(
                            key.trim().to_string(), 
                            value.trim().to_string()
                        );
                    }
                }
                "V4+ Styles" => {
                    if line.starts_with("Format:") {
                        style_format = Some(
                            line[7..].split(',')
                                .map(|s| s.trim().to_string())
                                .collect()
                        );
                    } else if line.starts_with("Style:") {
                        if let Some(ref format) = style_format {
                            if let Ok(style) = Self::parse_style_line(&line[6..], format) {
                                document.styles.insert(style.name.clone(), style);
                            }
                        }
                    }
                }
                "Events" => {
                    if line.starts_with("Format:") {
                        dialogue_format = Some(
                            line[7..].split(',')
                                .map(|s| s.trim().to_string())
                                .collect()
                        );
                    } else if line.starts_with("Dialogue:") {
                        if let Some(ref format) = dialogue_format {
                            if let Ok(dialogue) = Self::parse_dialogue_line(&line[9..], format) {
                                document.dialogues.push(dialogue);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(document)
    }

    fn parse_style_line(line: &str, format: &[String]) -> Result<AssStyle, String> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != format.len() {
            return Err("Style line doesn't match format".to_string());
        }

        let mut style = AssStyle {
            name: String::new(),
            fontname: "Arial".to_string(),
            fontsize: 20,
            primary_colour: "&H00FFFFFF".to_string(),
            secondary_colour: "&H000000FF".to_string(),
            outline_colour: "&H00000000".to_string(),
            back_colour: "&H80000000".to_string(),
            bold: false,
            italic: false,
            underline: false,
            strikeout: false,
            scale_x: 100.0,
            scale_y: 100.0,
            spacing: 0.0,
            angle: 0.0,
            border_style: 1,
            outline: 2.0,
            shadow: 0.0,
            alignment: 2,
            margin_l: 10,
            margin_r: 10,
            margin_v: 10,
            encoding: 1,
        };

        for (i, field) in format.iter().enumerate() {
            if i >= parts.len() {
                break;
            }
            let value = parts[i].trim();
            
            match field.as_str() {
                "Name" => style.name = value.to_string(),
                "Fontname" => style.fontname = value.to_string(),
                "Fontsize" => style.fontsize = value.parse().unwrap_or(20),
                "PrimaryColour" => style.primary_colour = value.to_string(),
                "SecondaryColour" => style.secondary_colour = value.to_string(),
                "OutlineColour" => style.outline_colour = value.to_string(),
                "BackColour" => style.back_colour = value.to_string(),
                "Bold" => style.bold = value == "-1" || value == "1",
                "Italic" => style.italic = value == "-1" || value == "1",
                "Underline" => style.underline = value == "-1" || value == "1",
                "StrikeOut" => style.strikeout = value == "-1" || value == "1",
                "ScaleX" => style.scale_x = value.parse().unwrap_or(100.0),
                "ScaleY" => style.scale_y = value.parse().unwrap_or(100.0),
                "Spacing" => style.spacing = value.parse().unwrap_or(0.0),
                "Angle" => style.angle = value.parse().unwrap_or(0.0),
                "BorderStyle" => style.border_style = value.parse().unwrap_or(1),
                "Outline" => style.outline = value.parse().unwrap_or(2.0),
                "Shadow" => style.shadow = value.parse().unwrap_or(0.0),
                "Alignment" => style.alignment = value.parse().unwrap_or(2),
                "MarginL" => style.margin_l = value.parse().unwrap_or(10),
                "MarginR" => style.margin_r = value.parse().unwrap_or(10),
                "MarginV" => style.margin_v = value.parse().unwrap_or(10),
                "Encoding" => style.encoding = value.parse().unwrap_or(1),
                _ => {}
            }
        }

        Ok(style)
    }

    fn parse_dialogue_line(line: &str, format: &[String]) -> Result<AssDialogue, String> {
        // Split by comma, but be careful about commas in the text field
        let mut parts = Vec::new();
        let mut current_part = String::new();
        let mut paren_depth = 0;
        let mut brace_depth = 0;
        
        for ch in line.chars() {
            match ch {
                ',' if paren_depth == 0 && brace_depth == 0 && parts.len() < format.len() - 1 => {
                    parts.push(current_part.trim().to_string());
                    current_part.clear();
                }
                '{' => {
                    brace_depth += 1;
                    current_part.push(ch);
                }
                '}' => {
                    brace_depth -= 1;
                    current_part.push(ch);
                }
                '(' => {
                    paren_depth += 1;
                    current_part.push(ch);
                }
                ')' => {
                    paren_depth -= 1;
                    current_part.push(ch);
                }
                _ => current_part.push(ch),
            }
        }
        parts.push(current_part.trim().to_string());

        if parts.len() != format.len() {
            return Err(format!("Dialogue line has {} parts but format expects {}", parts.len(), format.len()));
        }

        let mut dialogue = AssDialogue {
            layer: 0,
            start_time: String::new(),
            end_time: String::new(),
            style: "Default".to_string(),
            name: String::new(),
            margin_l: 0,
            margin_r: 0,
            margin_v: 0,
            effect: String::new(),
            text: String::new(),
        };

        for (i, field) in format.iter().enumerate() {
            if i >= parts.len() {
                break;
            }
            let value = &parts[i];
            
            match field.as_str() {
                "Layer" => dialogue.layer = value.parse().unwrap_or(0),
                "Start" => dialogue.start_time = value.clone(),
                "End" => dialogue.end_time = value.clone(),
                "Style" => dialogue.style = value.clone(),
                "Name" => dialogue.name = value.clone(),
                "MarginL" => dialogue.margin_l = value.parse().unwrap_or(0),
                "MarginR" => dialogue.margin_r = value.parse().unwrap_or(0),
                "MarginV" => dialogue.margin_v = value.parse().unwrap_or(0),
                "Effect" => dialogue.effect = value.clone(),
                "Text" => dialogue.text = value.clone(),
                _ => {}
            }
        }

        Ok(dialogue)
    }

    /// Clean ASS formatting tags from text
    pub fn clean_text(text: &str) -> String {
        // Remove ASS formatting tags like {\i1}, {\b1}, etc.
        let re = Regex::new(r"\{[^}]*\}").unwrap();
        let cleaned = re.replace_all(text, "").to_string();
        
        // Replace \\N with actual newlines
        let cleaned = cleaned.replace("\\N", "\n");
        
        // Replace \\n with actual newlines
        let cleaned = cleaned.replace("\\n", "\n");
        
        // Clean up extra whitespace
        cleaned.trim().to_string()
    }

    /// Extract speaker name from dialogue or style
    pub fn extract_speaker_name(&self, dialogue: &AssDialogue) -> String {
        // First try the Name field
        if !dialogue.name.is_empty() {
            return dialogue.name.clone();
        }
        
        // Then try the Style field
        if !dialogue.style.is_empty() && dialogue.style != "Default" {
            return dialogue.style.clone();
        }
        
        // As a fallback, try to extract from the text itself
        let clean_text = Self::clean_text(&dialogue.text);
        if let Some(colon_pos) = clean_text.find(':') {
            let potential_name = clean_text[..colon_pos].trim();
            if potential_name.len() < 20 && !potential_name.contains('\n') {
                return potential_name.to_string();
            }
        }
        
        // Default fallback
        "Unknown".to_string()
    }
}