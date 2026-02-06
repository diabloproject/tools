mod ass_parser;
mod converter;

use clap::{Arg, Command};
use std::fs;
use std::path::Path;
use converter::AssToLlmsubConverter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("ass-to-llmsub")
        .about("Convert ASS (Advanced SubStation Alpha) subtitle files to LLMSUB format")
        .version("0.1.0")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("FILE")
                .help("Input ASS file path")
                .required(true)
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output LLMSUB file path (optional, defaults to input file with .llmsub extension)")
        )
        .arg(
            Arg::new("break-threshold")
                .short('b')
                .long("break-threshold")
                .value_name("SECONDS")
                .help("Time gap threshold for scene breaks in seconds (default: 10.0)")
                .default_value("10.0")
        )
        .arg(
            Arg::new("no-grouping")
                .long("no-grouping")
                .help("Don't group consecutive dialogues by the same character")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("no-style-descriptions")
                .long("no-style-descriptions")
                .help("Don't use style information for character descriptions")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap();
    let output_path = matches.get_one::<String>("output")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            let input = Path::new(input_path);
            input.with_extension("llmsub")
                .to_string_lossy()
                .to_string()
        });

    let break_threshold: f64 = matches.get_one::<String>("break-threshold")
        .unwrap()
        .parse()
        .map_err(|_| "Invalid break threshold value")?;

    let group_characters = !matches.get_flag("no-grouping");
    let use_style_descriptions = !matches.get_flag("no-style-descriptions");

    // Read input file
    println!("Reading ASS file: {}", input_path);
    let ass_content = fs::read_to_string(input_path)
        .map_err(|e| format!("Failed to read input file '{}': {}", input_path, e))?;

    // Parse ASS document
    println!("Parsing ASS document...");
    let ass_doc = ass_parser::AssDocument::parse(&ass_content)
        .map_err(|e| format!("Failed to parse ASS file: {}", e))?;

    println!("Found {} dialogue lines", ass_doc.dialogues.len());
    println!("Found {} styles", ass_doc.styles.len());

    // Configure converter
    let converter = AssToLlmsubConverter::new()
        .with_break_threshold(break_threshold)
        .with_grouping(group_characters)
        .with_style_descriptions(use_style_descriptions);

    // Convert to LLMSUB
    println!("Converting to LLMSUB format...");
    let llmsub_doc = converter.convert(&ass_doc)
        .map_err(|e| format!("Conversion failed: {}", e))?;

    let characters = llmsub_doc.get_characters();
    let dialogues = llmsub_doc.get_dialogues();
    let breaks = llmsub_doc.get_break_positions();

    println!("Conversion complete:");
    println!("  - {} characters defined", characters.len());
    println!("  - {} dialogue blocks", dialogues.len());
    println!("  - {} scene breaks", breaks.len());

    // Serialize and save
    let llmsub_content = llmsub_doc.serialize();
    fs::write(&output_path, llmsub_content)
        .map_err(|e| format!("Failed to write output file '{}': {}", output_path, e))?;

    println!("LLMSUB file saved to: {}", output_path);

    // Show preview of character list
    if !characters.is_empty() {
        println!("\nCharacters found:");
        for character in characters {
            println!("  - {}: {}", character.name, 
                if character.description.len() > 50 {
                    format!("{}...", &character.description[..47])
                } else {
                    character.description.clone()
                }
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_full_conversion_workflow() {
        let ass_content = r#"[Script Info]
Title: Test Dialogue
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Narrator,Arial,18,&H00FFFF00,&H000000FF,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:03.00,Default,Alice,0,0,0,,Hello, how are you?
Dialogue: 0,0:00:03.50,0:00:06.00,Default,Bob,0,0,0,,I'm doing great!
Dialogue: 0,0:00:06.50,0:00:09.00,Default,Alice,0,0,0,,That's wonderful to hear.
Dialogue: 0,0:00:20.00,0:00:23.00,Narrator,,0,0,0,,{\i1}Later that day...{\i0}
Dialogue: 0,0:00:23.50,0:00:26.00,Default,Alice,0,0,0,,See you tomorrow!
"#;

        // Create temporary input file
        let mut input_file = NamedTempFile::new().unwrap();
        input_file.write_all(ass_content.as_bytes()).unwrap();

        // Test the conversion function
        use converter::convert_ass_to_llmsub;
        let result = convert_ass_to_llmsub(ass_content);
        assert!(result.is_ok());

        let llmsub_content = result.unwrap();
        
        // Verify the output contains expected elements
        assert!(llmsub_content.contains("character Alice:"));
        assert!(llmsub_content.contains("character Bob:"));
        assert!(llmsub_content.contains("Alice said:"));
        assert!(llmsub_content.contains("Bob said:"));
        assert!(llmsub_content.contains("Hello, how are you?"));
        assert!(llmsub_content.contains("I'm doing great!"));
        assert!(llmsub_content.contains("BREAK;"));
        assert!(llmsub_content.contains("Later that day..."));
    }
}
