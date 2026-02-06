# YSON - Yet another Serialization Object Notation

A Rust implementation of YSON parser and lexer. YSON is a data serialization format that extends JSON-like syntax with additional features including attributes, binary encoding support, and more data types.

## Features

- **Streaming lexer and parser** - Memory efficient processing of large inputs
- **Comprehensive data types** - Support for integers, floats, strings, booleans, entities, arrays, and maps  
- **Attributes** - Attach metadata to any value using `<key=value>` syntax
- **Binary encoding** - Support for protocol buffer-style varint and binary string encoding
- **Robust error handling** - Detailed error messages with position information

## YSON Format

YSON supports the following data types:

### Basic Types
- **Signed integers**: `123`, `-456`
- **Unsigned integers**: `123u`, `456u`  
- **Floating point**: `123.45`, `-0.99`
- **Booleans**: `%true`, `%false`
- **Strings**: `"quoted strings"` or `unquoted_identifiers`
- **Entity**: `#` (represents null/unit value)

### Complex Types
- **Arrays**: `[1; 2; 3]` (semicolon separated)
- **Maps**: `{key=value; other=data}` (semicolon separated key-value pairs)

### Attributes
Any value can have attributes attached: `<attr1=value1; attr2=value2>actual_value`

### Binary Encoding
YSON supports binary encoding for efficiency:
- Strings: `0x01` + varint length + bytes
- Signed integers: `0x02` + zigzag varint  
- Doubles: `0x03` + 8 bytes little-endian
- Booleans: `0x04` (true), `0x05` (false)
- Unsigned integers: `0x06` + varint

## Usage

```rust
use yson::{lexer::YsonLexer, parser::YsonParser};

// Parse from string
let input = r#"{name="example"; values=[1; 2; 3]; enabled=%true}"#;
let lexer = YsonLexer::new(input.bytes().map(Ok));
let mut parser = YsonParser::new(lexer);
let result = parser.parse_complete().unwrap();

// Parse from bytes  
let input_bytes = b"<version=1>[1;2;3]";
let lexer = YsonLexer::new(input_bytes.iter().map(|&b| Ok(b)));
let mut parser = YsonParser::new(lexer);
let result = parser.parse_complete().unwrap();
```

## Error Handling

The library provides detailed error information:
- **Lexer errors**: Invalid bytes, unexpected EOF, parse failures
- **Parser errors**: Unexpected tokens, structural errors  
- **Protocol errors**: Invalid varint encoding, excessive lengths

All errors implement `std::error::Error` and provide human-readable descriptions.

## Testing

Run the test suite with:
```bash
cargo test
```

The library includes comprehensive tests covering all data types, error conditions, and edge cases.