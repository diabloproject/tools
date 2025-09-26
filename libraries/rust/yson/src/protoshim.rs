#![allow(dead_code)]
use std::fmt;

/// Error types for protobuf parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoshimError {
    Incomplete {
        expected: usize,
        got: usize,
        at: usize,
    },
    ExcessiveLength {
        input: Vec<u8>,
        error_position: usize,
        data_type: &'static str,
        at: usize,
    },
}

impl ProtoshimError {
    pub fn at(&self) -> usize {
        match self {
            Self::Incomplete { at, .. } => *at,
            Self::ExcessiveLength { at, .. } => *at,
        }
    }
}

impl fmt::Display for ProtoshimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtoshimError::Incomplete {
                expected,
                got,
                at: _,
            } => {
                write!(f, "Input sequence is incomplete\n\n")?;
                write!(
                    f,
                    "Expected at least {} bytes, but got only {} bytes",
                    expected, got
                )
            }
            ProtoshimError::ExcessiveLength {
                input,
                error_position,
                data_type,
                at: _,
            } => {
                write!(f, "Input sequence is excessive in length\n\n")?;
                write!(f, "Input: ")?;

                // Print hex bytes
                for (i, byte) in input.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "0x{:02X}", byte)?;
                }

                if input.len() > 12 {
                    write!(f, " ...")?;
                }

                write!(f, "\n")?;

                // Print pointer to error position
                let mut spaces = 0;
                for i in 0..*error_position {
                    if i > 0 {
                        spaces += 1;
                    } // space between bytes
                    spaces += 4; // "0x??"
                }

                write!(f, "{:width$}|\n", "", width = spaces + 1)?;
                write!(f, "At byte here {}\n", "—".repeat(spaces).trim_end())?;

                if *error_position < input.len() {
                    let error_byte = input[*error_position];
                    write!(
                        f,
                        "0x{:02X} & 0x80 == {}, therefore we must continue\n",
                        error_byte,
                        (error_byte & 0x80) != 0
                    )?;
                }

                write!(
                    f,
                    "But, {} will overflow if tried to include one more byte\n",
                    data_type
                )?;
                write!(
                    f,
                    "Therefore, we conclude that the input sequence is invalid"
                )
            }
        }
    }
}

impl std::error::Error for ProtoshimError {}

pub fn encode_sint64(value: i64) -> Vec<u8> {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint_u64(zigzag)
}

pub fn decode_sint64(bytes: &[u8]) -> Result<(i64, usize), ProtoshimError> {
    let (zigzag, bytes_read) = decode_varint_u64_internal(bytes, "sint64")?;
    let decoded = ((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64));
    Ok((decoded, bytes_read))
}

pub fn encode_sint32(value: i32) -> Vec<u8> {
    let zigzag = ((value << 1) ^ (value >> 31)) as u32;
    encode_varint_u32(zigzag)
}

pub fn decode_sint32(bytes: &[u8]) -> Result<(i32, usize), ProtoshimError> {
    let (zigzag, bytes_read) = decode_varint_u32_internal(bytes, "sint32")?;
    let decoded = ((zigzag >> 1) as i32) ^ (-((zigzag & 1) as i32));
    Ok((decoded, bytes_read))
}

pub fn encode_uint64(value: u64) -> Vec<u8> {
    encode_varint_u64(value)
}

pub fn decode_uint64(bytes: &[u8]) -> Result<(u64, usize), ProtoshimError> {
    decode_varint_u64_internal(bytes, "uint64")
}

pub fn encode_uint32(value: u32) -> Vec<u8> {
    encode_varint_u32(value)
}

pub fn decode_uint32(bytes: &[u8]) -> Result<(u32, usize), ProtoshimError> {
    decode_varint_u32_internal(bytes, "uint32")
}

pub fn encode_double(value: f64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

pub fn decode_double(bytes: &[u8]) -> Result<(f64, usize), ProtoshimError> {
    if bytes.len() < 8 {
        return Err(ProtoshimError::Incomplete {
            expected: 8,
            got: bytes.len(),
            at: bytes.len(),
        });
    }

    let mut array = [0u8; 8];
    array.copy_from_slice(&bytes[0..8]);
    Ok((f64::from_le_bytes(array), 8))
}

fn encode_varint_u64(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80;
        }

        result.push(byte);

        if value == 0 {
            break;
        }
    }

    result
}

fn decode_varint_u64_internal(
    bytes: &[u8],
    data_type: &'static str,
) -> Result<(u64, usize), ProtoshimError> {
    if bytes.is_empty() {
        return Err(ProtoshimError::Incomplete {
            expected: 1,
            got: 0,
            at: 0,
        });
    }

    let mut result = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in bytes {
        bytes_read += 1;

        result |= ((byte & 0x7F) as u64) << shift;

        if (byte & 0x80) == 0 {
            return Ok((result, bytes_read));
        }

        if bytes_read >= 10 {
            return Err(ProtoshimError::ExcessiveLength {
                input: bytes.iter().take(11).copied().collect(),
                error_position: 9,
                data_type,
                at: bytes_read,
            });
        }

        if shift >= 64 {
            unreachable!("Can this case even occur?")
        }

        shift += 7;
    }

    Err(ProtoshimError::Incomplete {
        expected: bytes.len() + 1,
        got: bytes.len(),
        at: bytes.len(),
    })
}

fn encode_varint_u32(mut value: u32) -> Vec<u8> {
    let mut result = Vec::new();

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80;
        }

        result.push(byte);

        if value == 0 {
            break;
        }
    }

    result
}

fn decode_varint_u32_internal(
    bytes: &[u8],
    data_type: &'static str,
) -> Result<(u32, usize), ProtoshimError> {
    if bytes.is_empty() {
        return Err(ProtoshimError::Incomplete {
            expected: 1,
            got: 0,
            at: 0,
        });
    }

    let mut result = 0u32;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in bytes {
        bytes_read += 1;

        result |= ((byte & 0x7F) as u32) << shift;

        if (byte & 0x80) == 0 {
            return Ok((result, bytes_read));
        }

        if bytes_read >= 5 {
            return Err(ProtoshimError::ExcessiveLength {
                input: bytes.iter().take(6).copied().collect(),
                error_position: 4,
                data_type,
                at: bytes_read,
            });
        }

        if shift >= 32 {
            unreachable!("Can this case even occur?")
        }

        shift += 7;
    }

    Err(ProtoshimError::Incomplete {
        expected: bytes.len() + 1,
        got: bytes.len(),
        at: bytes.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sint64() {
        assert_eq!(encode_sint64(0), vec![0x00]);
        assert_eq!(encode_sint64(1), vec![0x02]);
        assert_eq!(encode_sint64(-1), vec![0x01]);

        let (val, len) = decode_sint64(&[0x00]).unwrap();
        assert_eq!((val, len), (0, 1));

        let (val, len) = decode_sint64(&[0x02]).unwrap();
        assert_eq!((val, len), (1, 1));

        let (val, len) = decode_sint64(&[0x01]).unwrap();
        assert_eq!((val, len), (-1, 1));
    }

    #[test]
    fn test_sint32() {
        assert_eq!(encode_sint32(0), vec![0x00]);
        assert_eq!(encode_sint32(1), vec![0x02]);
        assert_eq!(encode_sint32(-1), vec![0x01]);

        let (val, len) = decode_sint32(&[0x00]).unwrap();
        assert_eq!((val, len), (0, 1));
    }

    #[test]
    fn test_uint64() {
        assert_eq!(encode_uint64(0), vec![0x00]);
        assert_eq!(encode_uint64(127), vec![0x7F]);
        assert_eq!(encode_uint64(128), vec![0x80, 0x01]);

        let (val, len) = decode_uint64(&[0x00]).unwrap();
        assert_eq!((val, len), (0, 1));

        let (val, len) = decode_uint64(&[0x7F]).unwrap();
        assert_eq!((val, len), (127, 1));

        let (val, len) = decode_uint64(&[0x80, 0x01]).unwrap();
        assert_eq!((val, len), (128, 2));
    }

    #[test]
    fn test_uint32() {
        assert_eq!(encode_uint32(0), vec![0x00]);
        assert_eq!(encode_uint32(127), vec![0x7F]);
        assert_eq!(encode_uint32(128), vec![0x80, 0x01]);

        let (val, len) = decode_uint32(&[0x00]).unwrap();
        assert_eq!((val, len), (0, 1));
    }

    #[test]
    fn test_double() {
        let pi = std::f64::consts::PI;
        let encoded = encode_double(pi);
        assert_eq!(encoded.len(), 8);

        let (decoded, len) = decode_double(&encoded).unwrap();
        assert_eq!((decoded, len), (pi, 8));

        // Test zero
        let encoded = encode_double(0.0);
        let (decoded, len) = decode_double(&encoded).unwrap();
        assert_eq!((decoded, len), (0.0, 8));

        // Test negative
        let encoded = encode_double(-123.456);
        let (decoded, len) = decode_double(&encoded).unwrap();
        assert_eq!((decoded, len), (-123.456, 8));
    }

    #[test]
    fn test_incomplete_errors() {
        // Test empty input
        match decode_sint64(&[]) {
            Err(ProtoshimError::Incomplete {
                expected: 1,
                got: 0,
                at: 0,
            }) => {}
            _ => panic!("Expected Incomplete error"),
        }

        // Test incomplete double
        match decode_double(&[0x00, 0x01, 0x02]) {
            Err(ProtoshimError::Incomplete {
                expected: 8,
                got: 3,
                at: 3,
            }) => {}
            _ => panic!("Expected Incomplete error"),
        }

        // Test incomplete varint
        match decode_uint64(&[0x80]) {
            Err(ProtoshimError::Incomplete {
                expected: 2,
                got: 1,
                at: 1,
            }) => {}
            _ => panic!("Expected Incomplete error"),
        }
    }

    #[test]
    fn test_excessive_length_errors() {
        // Test u64 overflow (all 10 bytes with continuation bit)
        let excessive_bytes = vec![0xFF; 10];
        match decode_uint64(&excessive_bytes) {
            Err(ProtoshimError::ExcessiveLength {
                data_type: "uint64",
                error_position: 9,
                at: 9,
                ..
            }) => {}
            other => panic!("Expected ExcessiveLength error, got: {:?}", other),
        }

        // Test u32 overflow (all 5 bytes with continuation bit)
        let excessive_bytes = vec![0xFF; 5];
        match decode_uint32(&excessive_bytes) {
            Err(ProtoshimError::ExcessiveLength {
                data_type: "uint32",
                error_position: 4,
                at: 4,
                ..
            }) => {}
            other => panic!("Expected ExcessiveLength error, got: {:?}", other),
        }
    }

    #[test]
    fn test_error_formatting() {
        let excessive_bytes = vec![0xFF; 10];
        let error = decode_uint64(&excessive_bytes).unwrap_err();
        let error_string = format!("{}", error);

        assert!(error_string.contains("Input sequence is excessive in length"));
        assert!(error_string.contains("0xFF 0xFF 0xFF"));
        assert!(error_string.contains("uint64 will overflow"));
    }
}
