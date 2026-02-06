# LLMSUB API Documentation

## Core Types

### `LlmsubDocument`
The main document container that holds all elements (characters, dialogues, breaks).

### `Character`
```rust
pub struct Character {
    pub name: String,
    pub description: String,
}
```

### `Dialogue`
```rust
pub struct Dialogue {
    pub character: String,
    pub content: String,
}
```

### `Element`
```rust
pub enum Element {
    Character(Character),
    Dialogue(Dialogue),
    Break,
}
```

## Deserialization (Parsing)

### Parse from string
```rust
use llmsub::*;

let content = r#"character Alice:
    A curious person
Alice said:
    — Hello!"#;

let document = LlmsubDocument::parse(content)?;
```

### Parse from file
```rust
use std::fs;

let content = fs::read_to_string("example.llmsub")?;
let document = LlmsubDocument::parse(&content)?;
```

### Error Handling
```rust
match LlmsubDocument::parse(content) {
    Ok(document) => { /* use document */ }
    Err(ParseError::UnknownCharacter(name)) => {
        eprintln!("Character '{}' used before definition", name);
    }
    Err(ParseError::InvalidFormat(msg)) => {
        eprintln!("Format error: {}", msg);
    }
}
```

## Serialization

### Serialize to string
```rust
let llmsub_text = document.serialize();
println!("{}", llmsub_text);
```

### Write to file
```rust
use std::fs;

let serialized = document.serialize();
fs::write("output.llmsub", serialized)?;
```

## Building Documents

### Create empty document
```rust
let mut document = LlmsubDocument::new();
// or
let mut document = LlmsubDocument::default();
```

### Add characters
```rust
document.add_character(Character {
    name: "Alice".to_string(),
    description: "A curious person".to_string(),
});
```

### Add dialogue
```rust
document.add_dialogue(Dialogue {
    character: "Alice".to_string(),
    content: "— Hello, world!".to_string(),
});
```

### Add scene breaks
```rust
document.add_break();
```

### Complete example
```rust
let mut document = LlmsubDocument::new();

// Define characters
document.add_character(Character {
    name: "Alice".to_string(),
    description: "A curious person who asks many questions".to_string(),
});

document.add_character(Character {
    name: "Bob".to_string(),
    description: "A helpful friend".to_string(),
});

// Add dialogue
document.add_dialogue(Dialogue {
    character: "Alice".to_string(),
    content: "— Hello Bob!".to_string(),
});

document.add_dialogue(Dialogue {
    character: "Bob".to_string(),
    content: "— Hi Alice! How are you?".to_string(),
});

// Scene break
document.add_break();

// More dialogue
document.add_dialogue(Dialogue {
    character: "Alice".to_string(),
    content: "— I'm doing great, thanks!".to_string(),
});

// Serialize to LLMSUB format
let output = document.serialize();
```

## Querying Documents

### Get all characters
```rust
let characters = document.get_characters();
for character in characters {
    println!("{}: {}", character.name, character.description);
}
```

### Get all dialogues
```rust
let dialogues = document.get_dialogues();
for dialogue in dialogues {
    println!("{}: {}", dialogue.character, dialogue.content);
}
```

### Find breaks
```rust
// Check if document has any breaks
if document.has_breaks() {
    println!("Document contains scene breaks");
}

// Get positions of all breaks
let break_positions = document.get_break_positions();
for pos in break_positions {
    println!("Break at element {}", pos);
}
```

### Access raw elements
```rust
for (index, element) in document.elements.iter().enumerate() {
    match element {
        Element::Character(c) => println!("Character {}: {}", c.name, c.description),
        Element::Dialogue(d) => println!("{}: {}", d.character, d.content),
        Element::Break => println!("Scene break at position {}", index),
    }
}
```

## Round-trip Processing

```rust
// Parse → Modify → Serialize
let mut document = LlmsubDocument::parse(input_text)?;

// Add new content
document.add_dialogue(Dialogue {
    character: "Narrator".to_string(),
    content: "The end.".to_string(),
});

// Serialize back
let output = document.serialize();
```