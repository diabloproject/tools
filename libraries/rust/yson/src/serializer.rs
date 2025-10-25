use crate::protoshim;
use crate::types::{YsonNode, YsonString, YsonValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    // Indents
    Pretty,
    // No indents, but prefer textual representations
    Compact,
    // Prefer binary representations
    Binary,
}

pub fn serialize(input: &YsonNode, mode: Mode) -> Vec<u8> {
    let mut output = Vec::new();
    serialize_node(input, &mode, &mut output, 0);
    output
}

fn serialize_node(node: &YsonNode, mode: &Mode, output: &mut Vec<u8>, indent_level: usize) {
    if !node.attributes.is_empty() {
        output.push(b'<');
        for (i, (key, value)) in node.attributes.iter().enumerate() {
            if i > 0 {
                output.push(b';');
                if matches!(mode, Mode::Pretty) {
                    output.push(b' ');
                }
            }
            serialize_string(key, mode, output);
            output.push(b'=');
            serialize_node(value, mode, output, indent_level);
        }
        output.push(b'>');
    }

    serialize_value(&node.value, mode, output, indent_level);
}

fn serialize_value(value: &YsonValue, mode: &Mode, output: &mut Vec<u8>, indent_level: usize) {
    match value {
        YsonValue::Entity => output.push(b'#'),
        YsonValue::Boolean(b) => {
            if matches!(mode, Mode::Binary) {
                output.push(if *b { 0x04 } else { 0x05 });
            } else {
                output.extend_from_slice(if *b { b"%true" } else { b"%false" });
            }
        }
        YsonValue::SignedInteger(i) => {
            if matches!(mode, Mode::Binary) {
                output.push(0x02);
                output.extend_from_slice(&protoshim::encode_sint64(*i));
            } else {
                output.extend_from_slice(i.to_string().as_bytes());
            }
        }
        YsonValue::UnsignedInteger(u) => {
            if matches!(mode, Mode::Binary) {
                output.push(0x06);
                output.extend_from_slice(&protoshim::encode_uint64(*u));
            } else {
                output.extend_from_slice(u.to_string().as_bytes());
                output.push(b'u');
            }
        }
        YsonValue::Double(f) => {
            if matches!(mode, Mode::Binary) {
                output.push(0x03);
                output.extend_from_slice(&protoshim::encode_double(*f));
            } else {
                output.extend_from_slice(f.to_string().as_bytes());
            }
        }
        YsonValue::String(s) => serialize_string(s, mode, output),
        YsonValue::Array(arr) => serialize_array(arr, mode, output, indent_level),
        YsonValue::Map(map) => serialize_map(map, mode, output, indent_level),
    }
}

fn serialize_string(s: &YsonString, mode: &Mode, output: &mut Vec<u8>) {
    if matches!(mode, Mode::Binary) {
        output.push(0x01);
        output.extend_from_slice(&protoshim::encode_uint64(s.len() as u64));
        output.extend_from_slice(s);
    } else {
        let can_be_identifier = !s.is_empty()
            && (s[0].is_ascii_alphabetic() || s[0] == b'_')
            && s.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_');

        if can_be_identifier {
            output.extend_from_slice(s);
        } else {
            output.push(b'"');
            for &byte in s {
                match byte {
                    b'"' => output.extend_from_slice(b"\\\""),
                    b'\\' => output.extend_from_slice(b"\\\\"),
                    _ => output.push(byte),
                }
            }
            output.push(b'"');
        }
    }
}

fn serialize_array(arr: &[YsonNode], mode: &Mode, output: &mut Vec<u8>, indent_level: usize) {
    output.push(b'[');

    for (i, node) in arr.iter().enumerate() {
        if i > 0 {
            output.push(b';');
        }

        if matches!(mode, Mode::Pretty) {
            output.push(b'\n');
            output.extend_from_slice(&vec![b' '; (indent_level + 1) * 4]);
        }

        serialize_node(node, mode, output, indent_level + 1);
    }

    if matches!(mode, Mode::Pretty) && !arr.is_empty() {
        output.push(b'\n');
        output.extend_from_slice(&vec![b' '; indent_level * 4]);
    }

    output.push(b']');
}

fn serialize_map(
    map: &[(YsonString, YsonNode)],
    mode: &Mode,
    output: &mut Vec<u8>,
    indent_level: usize,
) {
    output.push(b'{');

    for (i, (key, value)) in map.iter().enumerate() {
        if i > 0 {
            output.push(b';');
        }

        if matches!(mode, Mode::Pretty) {
            output.push(b'\n');
            output.extend_from_slice(&vec![b' '; (indent_level + 1) * 4]);
        }

        serialize_string(key, mode, output);
        output.push(b'=');
        serialize_node(value, mode, output, indent_level + 1);
    }

    if matches!(mode, Mode::Pretty) && !map.is_empty() {
        output.push(b'\n');
        output.extend_from_slice(&vec![b' '; indent_level * 4]);
    }

    output.push(b'}');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::YsonLexer;
    use crate::parser::YsonParser;

    fn round_trip_test(input: &str, mode: Mode) {
        let lexer = YsonLexer::new(input.bytes().map(Ok::<u8, std::convert::Infallible>));
        let mut parser = YsonParser::new(lexer);
        let node = parser.parse_complete().unwrap();

        let serialized = serialize(&node, mode);

        let lexer2 = YsonLexer::new(
            serialized
                .iter()
                .copied()
                .map(Ok::<u8, std::convert::Infallible>),
        );
        let mut parser2 = YsonParser::new(lexer2);
        let node2 = parser2.parse_complete().unwrap();

        assert_eq!(node, node2, "Round trip failed for input: {}", input);
    }

    #[test]
    fn test_serialize_entity() {
        let node = YsonNode {
            value: YsonValue::Entity,
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"#");
    }

    #[test]
    fn test_serialize_boolean() {
        let node = YsonNode {
            value: YsonValue::Boolean(true),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"%true");

        let node = YsonNode {
            value: YsonValue::Boolean(false),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"%false");
    }

    #[test]
    fn test_serialize_signed_integer() {
        let node = YsonNode {
            value: YsonValue::SignedInteger(123),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"123");
    }

    #[test]
    fn test_serialize_unsigned_integer() {
        let node = YsonNode {
            value: YsonValue::UnsignedInteger(456),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"456u");
    }

    #[test]
    fn test_serialize_double() {
        let node = YsonNode {
            value: YsonValue::Double(123.45),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"123.45");
    }

    #[test]
    fn test_serialize_string_identifier() {
        let node = YsonNode {
            value: YsonValue::String(b"hello".to_vec()),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"hello");
    }

    #[test]
    fn test_serialize_string_quoted() {
        let node = YsonNode {
            value: YsonValue::String(b"hello world".to_vec()),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"\"hello world\"");
    }

    #[test]
    fn test_serialize_string_with_escapes() {
        let node = YsonNode {
            value: YsonValue::String(b"hello \"world\"".to_vec()),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"\"hello \\\"world\\\"\"");
    }

    #[test]
    fn test_serialize_empty_array() {
        let node = YsonNode {
            value: YsonValue::Array(vec![]),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"[]");
    }

    #[test]
    fn test_serialize_array() {
        let node = YsonNode {
            value: YsonValue::Array(vec![
                YsonNode {
                    value: YsonValue::SignedInteger(1),
                    attributes: vec![],
                },
                YsonNode {
                    value: YsonValue::SignedInteger(2),
                    attributes: vec![],
                },
                YsonNode {
                    value: YsonValue::SignedInteger(3),
                    attributes: vec![],
                },
            ]),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"[1;2;3]");
    }

    #[test]
    fn test_serialize_empty_map() {
        let node = YsonNode {
            value: YsonValue::Map(vec![]),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"{}");
    }

    #[test]
    fn test_serialize_map() {
        let node = YsonNode {
            value: YsonValue::Map(vec![
                (
                    b"a".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(1),
                        attributes: vec![],
                    },
                ),
                (
                    b"b".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(2),
                        attributes: vec![],
                    },
                ),
            ]),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"{a=1;b=2}");
    }

    #[test]
    fn test_serialize_with_attributes() {
        let node = YsonNode {
            value: YsonValue::SignedInteger(42),
            attributes: vec![(
                b"foo".to_vec(),
                YsonNode {
                    value: YsonValue::String(b"bar".to_vec()),
                    attributes: vec![],
                },
            )],
        };
        assert_eq!(serialize(&node, Mode::Compact), b"<foo=bar>42");
    }

    #[test]
    fn test_serialize_binary_mode() {
        let node = YsonNode {
            value: YsonValue::Boolean(true),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x04]);

        let node = YsonNode {
            value: YsonValue::Boolean(false),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x05]);
    }

    #[test]
    fn test_round_trip_simple() {
        round_trip_test("123", Mode::Compact);
        round_trip_test("%true", Mode::Compact);
        round_trip_test("#", Mode::Compact);
        round_trip_test("\"hello\"", Mode::Compact);
    }

    #[test]
    fn test_round_trip_array() {
        round_trip_test("[1;2;3]", Mode::Compact);
        round_trip_test("[]", Mode::Compact);
        round_trip_test("[1;[2;3];4]", Mode::Compact);
    }

    #[test]
    fn test_round_trip_map() {
        round_trip_test("{a=1;b=2}", Mode::Compact);
        round_trip_test("{}", Mode::Compact);
        round_trip_test("{a={b=1;c=2};d=3}", Mode::Compact);
    }

    #[test]
    fn test_round_trip_with_attributes() {
        round_trip_test("<a=1;b=%true>42", Mode::Compact);
        round_trip_test("<foo=bar>[1;2;3]", Mode::Compact);
    }

    #[test]
    fn test_round_trip_complex() {
        round_trip_test("<a=1>[{x=y;z=<foo=bar>#}]", Mode::Compact);
    }

    #[test]
    fn test_pretty_mode_array() {
        let node = YsonNode {
            value: YsonValue::Array(vec![
                YsonNode {
                    value: YsonValue::SignedInteger(1),
                    attributes: vec![],
                },
                YsonNode {
                    value: YsonValue::SignedInteger(2),
                    attributes: vec![],
                },
            ]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Pretty);
        let expected = b"[\n    1;\n    2\n]";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pretty_mode_map() {
        let node = YsonNode {
            value: YsonValue::Map(vec![
                (
                    b"a".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(1),
                        attributes: vec![],
                    },
                ),
                (
                    b"b".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(2),
                        attributes: vec![],
                    },
                ),
            ]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Pretty);
        let expected = b"{\n    a=1;\n    b=2\n}";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pretty_mode_attributes() {
        let node = YsonNode {
            value: YsonValue::SignedInteger(42),
            attributes: vec![
                (
                    b"foo".to_vec(),
                    YsonNode {
                        value: YsonValue::String(b"bar".to_vec()),
                        attributes: vec![],
                    },
                ),
                (
                    b"baz".to_vec(),
                    YsonNode {
                        value: YsonValue::Boolean(true),
                        attributes: vec![],
                    },
                ),
            ],
        };
        let result = serialize(&node, Mode::Pretty);
        let expected = b"<foo=bar; baz=%true>42";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_mode_integers() {
        let node = YsonNode {
            value: YsonValue::SignedInteger(123),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Binary);
        assert_eq!(result[0], 0x02); // SignedInteger marker

        let node = YsonNode {
            value: YsonValue::UnsignedInteger(456),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Binary);
        assert_eq!(result[0], 0x06); // UnsignedInteger marker
    }

    #[test]
    fn test_binary_mode_double() {
        let node = YsonNode {
            value: YsonValue::Double(123.45),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Binary);
        assert_eq!(result[0], 0x03); // Double marker
        assert_eq!(result.len(), 9); // marker + 8 bytes for f64
    }

    #[test]
    fn test_binary_mode_string() {
        let node = YsonNode {
            value: YsonValue::String(b"hello".to_vec()),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Binary);
        assert_eq!(result[0], 0x01); // String marker
        assert_eq!(result[1], 0x05); // Length 5
        assert_eq!(&result[2..], b"hello");
    }

    #[test]
    fn test_round_trip_binary_mode() {
        round_trip_test("123", Mode::Binary);
        round_trip_test("%true", Mode::Binary);
        round_trip_test("456u", Mode::Binary);
        round_trip_test("\"hello\"", Mode::Binary);
        round_trip_test("[1;2;3]", Mode::Binary);
        round_trip_test("{a=1;b=2}", Mode::Binary);
    }

    #[test]
    fn test_round_trip_pretty_mode() {
        round_trip_test("123", Mode::Pretty);
        round_trip_test("%true", Mode::Pretty);
        round_trip_test("[1;2;3]", Mode::Pretty);
        round_trip_test("{a=1;b=2}", Mode::Pretty);
        round_trip_test("<foo=bar>42", Mode::Pretty);
    }

    #[test]
    fn test_nested_pretty_mode() {
        let node = YsonNode {
            value: YsonValue::Array(vec![YsonNode {
                value: YsonValue::Map(vec![(
                    b"x".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(1),
                        attributes: vec![],
                    },
                )]),
                attributes: vec![],
            }]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Pretty);
        let expected = b"[\n    {\n        x=1\n    }\n]";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_signed_integer_encoding() {
        let node = YsonNode {
            value: YsonValue::SignedInteger(0),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x02, 0x00]);

        let node = YsonNode {
            value: YsonValue::SignedInteger(1),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x02, 0x02]);

        let node = YsonNode {
            value: YsonValue::SignedInteger(-1),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x02, 0x01]);

        let node = YsonNode {
            value: YsonValue::SignedInteger(127),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x02, 0xfe, 0x01]);
    }

    #[test]
    fn test_binary_unsigned_integer_encoding() {
        let node = YsonNode {
            value: YsonValue::UnsignedInteger(0),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x06, 0x00]);

        let node = YsonNode {
            value: YsonValue::UnsignedInteger(127),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x06, 0x7f]);

        let node = YsonNode {
            value: YsonValue::UnsignedInteger(128),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x06, 0x80, 0x01]);
    }

    #[test]
    fn test_binary_string_encoding() {
        let node = YsonNode {
            value: YsonValue::String(b"".to_vec()),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x01, 0x00]);

        let node = YsonNode {
            value: YsonValue::String(b"a".to_vec()),
            attributes: vec![],
        };
        assert_eq!(serialize(&node, Mode::Binary), vec![0x01, 0x01, b'a']);

        let node = YsonNode {
            value: YsonValue::String(b"test".to_vec()),
            attributes: vec![],
        };
        assert_eq!(
            serialize(&node, Mode::Binary),
            vec![0x01, 0x04, b't', b'e', b's', b't']
        );
    }

    #[test]
    fn test_binary_array_structure() {
        let node = YsonNode {
            value: YsonValue::Array(vec![
                YsonNode {
                    value: YsonValue::SignedInteger(1),
                    attributes: vec![],
                },
                YsonNode {
                    value: YsonValue::SignedInteger(2),
                    attributes: vec![],
                },
            ]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Binary);
        assert_eq!(result, vec![b'[', 0x02, 0x02, b';', 0x02, 0x04, b']']);
    }

    #[test]
    fn test_binary_map_structure() {
        let node = YsonNode {
            value: YsonValue::Map(vec![(
                b"x".to_vec(),
                YsonNode {
                    value: YsonValue::SignedInteger(10),
                    attributes: vec![],
                },
            )]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Binary);
        // { 0x01 0x01 'x' = 0x02 <varint(10)> }
        assert_eq!(result, vec![b'{', 0x01, 0x01, b'x', b'=', 0x02, 0x14, b'}']);
    }

    #[test]
    fn test_compact_complex_nested() {
        let node = YsonNode {
            value: YsonValue::Map(vec![(
                b"users".to_vec(),
                YsonNode {
                    value: YsonValue::Array(vec![YsonNode {
                        value: YsonValue::Map(vec![
                            (
                                b"name".to_vec(),
                                YsonNode {
                                    value: YsonValue::String(b"Alice".to_vec()),
                                    attributes: vec![],
                                },
                            ),
                            (
                                b"age".to_vec(),
                                YsonNode {
                                    value: YsonValue::SignedInteger(30),
                                    attributes: vec![],
                                },
                            ),
                        ]),
                        attributes: vec![],
                    }]),
                    attributes: vec![],
                },
            )]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Compact);
        let expected = b"{users=[{name=Alice;age=30}]}";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compact_with_nested_attributes() {
        let node = YsonNode {
            value: YsonValue::Array(vec![
                YsonNode {
                    value: YsonValue::SignedInteger(1),
                    attributes: vec![(
                        b"id".to_vec(),
                        YsonNode {
                            value: YsonValue::String(b"a".to_vec()),
                            attributes: vec![],
                        },
                    )],
                },
                YsonNode {
                    value: YsonValue::SignedInteger(2),
                    attributes: vec![(
                        b"id".to_vec(),
                        YsonNode {
                            value: YsonValue::String(b"b".to_vec()),
                            attributes: vec![],
                        },
                    )],
                },
            ]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Compact);
        let expected = b"[<id=a>1;<id=b>2]";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pretty_deeply_nested() {
        let node = YsonNode {
            value: YsonValue::Map(vec![(
                b"data".to_vec(),
                YsonNode {
                    value: YsonValue::Array(vec![YsonNode {
                        value: YsonValue::Map(vec![(
                            b"x".to_vec(),
                            YsonNode {
                                value: YsonValue::SignedInteger(1),
                                attributes: vec![],
                            },
                        )]),
                        attributes: vec![],
                    }]),
                    attributes: vec![],
                },
            )]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Pretty);
        let expected = b"{\n    data=[\n        {\n            x=1\n        }\n    ]\n}";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_string_escaping_backslash() {
        let node = YsonNode {
            value: YsonValue::String(b"a\\b".to_vec()),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Compact);
        assert_eq!(result, b"\"a\\\\b\"");
    }

    #[test]
    fn test_mixed_types_in_array() {
        let node = YsonNode {
            value: YsonValue::Array(vec![
                YsonNode {
                    value: YsonValue::SignedInteger(42),
                    attributes: vec![],
                },
                YsonNode {
                    value: YsonValue::String(b"hello".to_vec()),
                    attributes: vec![],
                },
                YsonNode {
                    value: YsonValue::Boolean(true),
                    attributes: vec![],
                },
                YsonNode {
                    value: YsonValue::Entity,
                    attributes: vec![],
                },
            ]),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Compact);
        assert_eq!(result, b"[42;hello;%true;#]");
    }

    #[test]
    fn test_empty_string_serialization() {
        let node = YsonNode {
            value: YsonValue::String(b"".to_vec()),
            attributes: vec![],
        };
        let result = serialize(&node, Mode::Compact);
        assert_eq!(result, b"\"\"");
    }

    #[test]
    fn test_multiple_attributes_formatting() {
        let node = YsonNode {
            value: YsonValue::Entity,
            attributes: vec![
                (
                    b"a".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(1),
                        attributes: vec![],
                    },
                ),
                (
                    b"b".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(2),
                        attributes: vec![],
                    },
                ),
                (
                    b"c".to_vec(),
                    YsonNode {
                        value: YsonValue::SignedInteger(3),
                        attributes: vec![],
                    },
                ),
            ],
        };

        let result_compact = serialize(&node, Mode::Compact);
        assert_eq!(result_compact, b"<a=1;b=2;c=3>#");

        let result_pretty = serialize(&node, Mode::Pretty);
        assert_eq!(result_pretty, b"<a=1; b=2; c=3>#");
    }
}
