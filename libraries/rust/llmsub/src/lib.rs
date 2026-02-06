use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Character {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dialogue {
    pub character: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    Character(Character),
    Dialogue(Dialogue),
    Break,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmsubDocument {
    pub elements: Vec<Element>,
}

#[derive(Debug)]
pub enum ParseError {
    InvalidFormat(String),
    UnknownCharacter(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ParseError::UnknownCharacter(name) => write!(f, "Unknown character: {}", name),
        }
    }
}

impl std::error::Error for ParseError {}

impl LlmsubDocument {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    pub fn parse(content: &str) -> Result<Self, ParseError> {
        let mut elements = Vec::new();
        let mut characters: HashMap<String, Character> = HashMap::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines
            if line.is_empty() {
                i += 1;
                continue;
            }

            // Check for character definition
            if line.starts_with("character ") && line.ends_with(':') {
                let name = line[10..line.len()-1].trim().to_string();
                let mut description_lines = Vec::new();
                
                // Collect description lines
                i += 1;
                while i < lines.len() {
                    let desc_line = lines[i];
                    // Stop if we hit an empty line or another element
                    if desc_line.trim().is_empty() || 
                       desc_line.starts_with("character ") ||
                       desc_line.contains(" said:") ||
                       desc_line.trim() == "BREAK;" {
                        break;
                    }
                    description_lines.push(desc_line.trim_start());
                    i += 1;
                }
                
                let description = description_lines.join("\n");
                let character = Character {
                    name: name.clone(),
                    description,
                };
                
                characters.insert(name.clone(), character.clone());
                elements.push(Element::Character(character));
                continue;
            }

            // Check for dialogue
            if line.contains(" said:") {
                let parts: Vec<&str> = line.splitn(2, " said:").collect();
                if parts.len() != 2 {
                    return Err(ParseError::InvalidFormat(format!("Invalid dialogue format: {}", line)));
                }
                
                let character_name = parts[0].trim().to_string();
                
                // Verify character exists
                if !characters.contains_key(&character_name) {
                    return Err(ParseError::UnknownCharacter(character_name.clone()));
                }
                
                let mut dialogue_lines = Vec::new();
                
                // If there's content after "said:", include it
                let first_content = parts[1].trim();
                if !first_content.is_empty() {
                    dialogue_lines.push(first_content);
                }
                
                // Collect subsequent dialogue lines
                i += 1;
                while i < lines.len() {
                    let dialogue_line = lines[i];
                    // Stop if we hit an empty line or another element
                    if dialogue_line.trim().is_empty() ||
                       dialogue_line.starts_with("character ") ||
                       dialogue_line.contains(" said:") ||
                       dialogue_line.trim() == "BREAK;" {
                        break;
                    }
                    dialogue_lines.push(dialogue_line.trim_start());
                    i += 1;
                }
                
                let content = dialogue_lines.join("\n");
                let dialogue = Dialogue {
                    character: character_name,
                    content,
                };
                
                elements.push(Element::Dialogue(dialogue));
                continue;
            }

            // Check for break
            if line == "BREAK;" {
                elements.push(Element::Break);
                i += 1;
                continue;
            }

            // If we get here, it's an unrecognized line format
            return Err(ParseError::InvalidFormat(format!("Unrecognized line: {}", line)));
        }

        Ok(Self { elements })
    }

    pub fn serialize(&self) -> String {
        let mut result = Vec::new();
        let mut prev_was_break = false;

        for (idx, element) in self.elements.iter().enumerate() {
            match element {
                Element::Character(character) => {
                    // Add spacing before character if needed
                    if idx > 0 && !prev_was_break {
                        result.push(String::new());
                    }
                    
                    result.push(format!("character {}:", character.name));
                    
                    // Split description into lines and indent them
                    for line in character.description.lines() {
                        result.push(format!("    {}", line));
                    }
                    prev_was_break = false;
                }
                Element::Dialogue(dialogue) => {
                    result.push(format!("{} said:", dialogue.character));
                    
                    // Split content into lines and indent them
                    for line in dialogue.content.lines() {
                        result.push(format!("    {}", line));
                    }
                    prev_was_break = false;
                }
                Element::Break => {
                    result.push(String::new());
                    result.push("BREAK;".to_string());
                    result.push(String::new());
                    prev_was_break = true;
                }
            }
        }

        result.join("\n")
    }

    pub fn add_character(&mut self, character: Character) {
        self.elements.push(Element::Character(character));
    }

    pub fn add_dialogue(&mut self, dialogue: Dialogue) {
        self.elements.push(Element::Dialogue(dialogue));
    }

    pub fn add_break(&mut self) {
        self.elements.push(Element::Break);
    }

    pub fn get_characters(&self) -> Vec<&Character> {
        self.elements.iter()
            .filter_map(|element| match element {
                Element::Character(character) => Some(character),
                _ => None,
            })
            .collect()
    }

    pub fn get_dialogues(&self) -> Vec<&Dialogue> {
        self.elements.iter()
            .filter_map(|element| match element {
                Element::Dialogue(dialogue) => Some(dialogue),
                _ => None,
            })
            .collect()
    }

    pub fn get_break_positions(&self) -> Vec<usize> {
        self.elements.iter()
            .enumerate()
            .filter_map(|(index, element)| match element {
                Element::Break => Some(index),
                _ => None,
            })
            .collect()
    }

    pub fn has_breaks(&self) -> bool {
        self.elements.iter().any(|element| matches!(element, Element::Break))
    }
}

impl Default for LlmsubDocument {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_example() {
        let content = r#"character Wife:
    Nice, careful character with a loving soul.
    Descriptions of characters can include multiple lines, like this.
character Husband:
    Husband of the wife

Wife said:
    — Good morning, darling. Do you want breakfast?
Husband said:
    — I sure do, my love!

BREAK;

Wife said:
    — Here you go!
Husband said:
    — Thank you!"#;

        let document = LlmsubDocument::parse(content).expect("Failed to parse");
        
        // Check we have the right number of elements
        assert_eq!(document.elements.len(), 7);
        
        // Check characters
        let characters = document.get_characters();
        assert_eq!(characters.len(), 2);
        assert_eq!(characters[0].name, "Wife");
        assert_eq!(characters[1].name, "Husband");
        
        // Check dialogues
        let dialogues = document.get_dialogues();
        assert_eq!(dialogues.len(), 4);
        assert_eq!(dialogues[0].character, "Wife");
        assert_eq!(dialogues[0].content, "— Good morning, darling. Do you want breakfast?");
    }

    #[test]
    fn test_serialize_roundtrip() {
        let content = r#"character Wife:
    Nice, careful character with a loving soul.
    Descriptions of characters can include multiple lines, like this.
character Husband:
    Husband of the wife

Wife said:
    — Good morning, darling. Do you want breakfast?
Husband said:
    — I sure do, my love!

BREAK;

Wife said:
    — Here you go!
Husband said:
    — Thank you!"#;

        let document = LlmsubDocument::parse(content).expect("Failed to parse");
        let serialized = document.serialize();
        let reparsed = LlmsubDocument::parse(&serialized).expect("Failed to reparse");
        
        assert_eq!(document, reparsed);
    }

    #[test]
    fn test_unknown_character_error() {
        let content = "UnknownCharacter said:\n    — Hello!";
        let result = LlmsubDocument::parse(content);
        
        assert!(matches!(result, Err(ParseError::UnknownCharacter(_))));
    }

    #[test]
    fn test_parse_actual_example_file() {
        let content = r#"character Wife:
    Nice, careful character with a loving soul.
    Descriptions of characters can include multiple lines, like this.
character Husband:
    Husband of the wife

Wife said:
    — Good morning, darling. Do you want breakfast?
Husband said:
    — I sure do, my love!

BREAK;

Wife said:
    — Here you go!
Husband said:
    — Thank you!"#;

        let document = LlmsubDocument::parse(content).expect("Failed to parse example file");
        
        // Verify the structure
        assert_eq!(document.elements.len(), 7);
        
        // Check first character
        match &document.elements[0] {
            Element::Character(c) => {
                assert_eq!(c.name, "Wife");
                assert_eq!(c.description, "Nice, careful character with a loving soul.\nDescriptions of characters can include multiple lines, like this.");
            }
            _ => panic!("Expected character element"),
        }
        
        // Check second character 
        match &document.elements[1] {
            Element::Character(c) => {
                assert_eq!(c.name, "Husband");
                assert_eq!(c.description, "Husband of the wife");
            }
            _ => panic!("Expected character element"),
        }
        
        // Check first dialogue
        match &document.elements[2] {
            Element::Dialogue(d) => {
                assert_eq!(d.character, "Wife");
                assert_eq!(d.content, "— Good morning, darling. Do you want breakfast?");
            }
            _ => panic!("Expected dialogue element"),
        }
        
        // Check break
        match &document.elements[4] {
            Element::Break => {},
            _ => panic!("Expected break element"),
        }
        
        // Test serialization matches expected format
        let serialized = document.serialize();
        let reparsed = LlmsubDocument::parse(&serialized).expect("Failed to reparse serialized content");
        assert_eq!(document, reparsed);
    }

    #[test]
    fn test_manual_construction() {
        let mut document = LlmsubDocument::new();
        
        document.add_character(Character {
            name: "Alice".to_string(),
            description: "A curious character".to_string(),
        });
        
        document.add_dialogue(Dialogue {
            character: "Alice".to_string(),
            content: "— Hello, world!".to_string(),
        });
        
        document.add_break();
        
        let serialized = document.serialize();
        assert!(serialized.contains("character Alice:"));
        assert!(serialized.contains("BREAK;"));
    }

    #[test]
    fn test_break_functionality() {
        let mut document = LlmsubDocument::new();
        
        // Initially no breaks
        assert!(!document.has_breaks());
        assert_eq!(document.get_break_positions(), vec![]);
        
        document.add_character(Character {
            name: "A".to_string(),
            description: "Test".to_string(),
        });
        document.add_dialogue(Dialogue {
            character: "A".to_string(),
            content: "First".to_string(),
        });
        document.add_break();
        document.add_dialogue(Dialogue {
            character: "A".to_string(),
            content: "Second".to_string(),
        });
        document.add_break();
        
        // Now we have breaks
        assert!(document.has_breaks());
        assert_eq!(document.get_break_positions(), vec![2, 4]);
        
        // Verify elements at those positions are breaks
        assert!(matches!(document.elements[2], Element::Break));
        assert!(matches!(document.elements[4], Element::Break));
    }
}
