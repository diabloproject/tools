use llmsub::*;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LLMSUB Parser/Serializer Demo");
    println!("==============================");
    
    // Read the example file
    let content = fs::read_to_string("examples/basic.llmsub")?;
    println!("Original content:");
    println!("{}\n", content);
    
    // Parse the content
    let document = LlmsubDocument::parse(&content)?;
    println!("Parsed {} elements", document.elements.len());
    
    // Show characters
    let characters = document.get_characters();
    println!("\nCharacters found:");
    for character in characters {
        println!("  - {}: {}", character.name, character.description.lines().next().unwrap_or(""));
    }
    
    // Show dialogues
    let dialogues = document.get_dialogues();
    println!("\nDialogues found:");
    for dialogue in dialogues {
        println!("  - {}: {}", dialogue.character, dialogue.content.lines().next().unwrap_or(""));
    }
    
    // Serialize back to string
    let serialized = document.serialize();
    println!("\nSerialized content:");
    println!("{}", serialized);
    
    // Test round-trip
    let reparsed = LlmsubDocument::parse(&serialized)?;
    println!("\nRound-trip test: {}", if document == reparsed { "PASSED" } else { "FAILED" });
    
    // Create a document programmatically
    let mut new_doc = LlmsubDocument::new();
    new_doc.add_character(Character {
        name: "Robot".to_string(),
        description: "A helpful AI assistant".to_string(),
    });
    new_doc.add_character(Character {
        name: "Human".to_string(), 
        description: "A curious person asking questions".to_string(),
    });
    new_doc.add_dialogue(Dialogue {
        character: "Human".to_string(),
        content: "— Hello, Robot!".to_string(),
    });
    new_doc.add_dialogue(Dialogue {
        character: "Robot".to_string(),
        content: "— Hello! How can I help you today?".to_string(),
    });
    new_doc.add_break();
    new_doc.add_dialogue(Dialogue {
        character: "Human".to_string(),
        content: "— Can you explain the LLMSUB format?".to_string(),
    });
    new_doc.add_dialogue(Dialogue {
        character: "Robot".to_string(),
        content: "— LLMSUB is a simple text format for characters and dialogue!".to_string(),
    });
    
    println!("\nProgrammatically created document:");
    println!("{}", new_doc.serialize());
    
    Ok(())
}