# ASS to LLMSUB Converter

A best-effort converter from ASS (Advanced SubStation Alpha) subtitle format to LLMSUB format.

## Features

- **Character Extraction**: Automatically identifies and creates character definitions from dialogue speakers
- **Style-based Descriptions**: Generates character descriptions based on ASS style information
- **Scene Break Detection**: Detects scene breaks based on timing gaps between dialogues  
- **Dialogue Grouping**: Groups consecutive dialogues from the same character
- **Text Cleaning**: Removes ASS formatting tags and handles newline conversions
- **Flexible Configuration**: Customizable break thresholds and processing options

## Usage

### Basic Conversion
```bash
cargo run -- -i input.ass -o output.llmsub
```

### Advanced Options
```bash
cargo run -- -i input.ass -o output.llmsub \
    --break-threshold 15.0 \
    --no-grouping \
    --no-style-descriptions
```

### Command Line Options

- `-i, --input <FILE>`: Input ASS file path (required)
- `-o, --output <FILE>`: Output LLMSUB file path (optional, defaults to input file with .llmsub extension)
- `-b, --break-threshold <SECONDS>`: Time gap threshold for scene breaks in seconds (default: 10.0)
- `--no-grouping`: Don't group consecutive dialogues by the same character
- `--no-style-descriptions`: Don't use style information for character descriptions

## How It Works

### Character Identification
The converter identifies characters from:
1. The `Name` field in dialogue lines
2. The `Style` field (if different from "Default")
3. Text parsing as fallback (looking for "Speaker:" patterns)

### Character Descriptions
Descriptions are generated based on:
- Style information (font, formatting)
- Dialogue frequency (major/supporting/minor character classification)
- Text formatting hints (bold, italic)

### Scene Break Detection
Scene breaks are automatically inserted when:
- Time gap between dialogues exceeds the threshold (default: 10 seconds)
- This helps preserve narrative structure from the original subtitles

### Text Processing
- Removes ASS formatting tags like `{\i1}`, `{\b1}`, etc.
- Converts `\N` and `\n` to proper newlines
- Preserves dialogue content while cleaning presentation markup

## Input Format (ASS)

ASS files should contain standard sections:
```
[Script Info]
Title: My Script

[V4+ Styles] 
Format: Name, Fontname, Fontsize, ...
Style: Default,Arial,20,...
Style: Narrator,Arial,18,...

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Default,Alice,0,0,0,,Hello world!
```

## Output Format (LLMSUB)

The converter produces LLMSUB files like:
```
character Alice:
    Character appearing in the dialogue. Major character with many lines

character Bob:  
    Character appearing in the dialogue. Uses 'Narrator' text style. Supporting character

Alice said:
    Good morning! How did you sleep?
    I had the strangest dream last night.

Bob said:
    Pretty well, thanks for asking.

BREAK;

Alice said:
    It felt so real!
```

## Examples

See `examples/sample.ass` for a sample ASS file that demonstrates various features:
- Multiple characters with different styles
- Scene transitions with timing gaps
- Formatted text with ASS tags
- Multi-line dialogues

To convert the sample:
```bash
cargo run -- -i examples/sample.ass -o examples/sample.llmsub
```

## Limitations

This is a **best-effort** converter with some limitations:

- **Character Recognition**: May not perfectly identify all speakers, especially in complex subtitle files
- **Style Information**: Limited to what's available in the ASS file styles
- **Timing Assumptions**: Scene break detection relies on timing gaps which may not always indicate narrative breaks
- **Text Content**: Some ASS-specific formatting may be lost in translation
- **Complex Dialogues**: Overlapping or simultaneous dialogues may not be perfectly represented

## Dependencies

- `llmsub` - The LLMSUB format library
- `regex` - Text processing and tag removal  
- `clap` - Command line argument parsing

## Testing

Run the test suite:
```bash
cargo test
```

The tests cover:
- ASS parsing functionality
- Time format conversion
- Scene break detection
- Full conversion workflow
- Text cleaning operations

## Development

The codebase is organized into modules:
- `ass_parser.rs` - ASS format parsing and data structures
- `converter.rs` - Main conversion logic and LLMSUB generation  
- `main.rs` - CLI interface and application logic

To extend functionality:
1. Modify parsing in `ass_parser.rs` for new ASS features
2. Update conversion logic in `converter.rs` for improved character/scene detection
3. Add new CLI options in `main.rs` for additional configuration