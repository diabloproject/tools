#![allow(dead_code)]
use std::error::Error;
use std::fmt;

/// Error types for protobuf parsing
#[derive(Debug, Clone)]
pub enum ProtoshimError<E: Error> {
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
    IteratorError(E),
}

impl<E: Error> PartialEq for ProtoshimError<E> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Incomplete {
                    expected: l_expected,
                    got: l_got,
                    at: l_at,
                },
                Self::Incomplete {
                    expected: r_expected,
                    got: r_got,
                    at: r_at,
                },
            ) => l_expected == r_expected && l_got == r_got && l_at == r_at,
            (
                Self::ExcessiveLength {
                    input: l_input,
                    error_position: l_error_position,
                    data_type: l_data_type,
                    at: l_at,
                },
                Self::ExcessiveLength {
                    input: r_input,
                    error_position: r_error_position,
                    data_type: r_data_type,
                    at: r_at,
                },
            ) => {
                l_input == r_input
                    && l_error_position == r_error_position
                    && l_data_type == r_data_type
                    && l_at == r_at
            }
            (Self::IteratorError(_), Self::IteratorError(_)) => true, // Cant compare E
            _ => false,
        }
    }
}

impl<E: Error> ProtoshimError<E> {
    pub fn at(&self) -> usize {
        match self {
            Self::Incomplete { at, .. } => *at,
            Self::ExcessiveLength { at, .. } => *at,
            Self::IteratorError(_) => todo!(),
        }
    }
}

impl<E: Error> fmt::Display for ProtoshimError<E> {
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
            ProtoshimError::IteratorError(msg) => {
                write!(f, "Inner iterator has thrown an error: {}", msg)
            }
        }
    }
}

impl<E: Error> Error for ProtoshimError<E> {}

pub fn encode_sint64(value: i64) -> Vec<u8> {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint_u64(zigzag)
}

pub fn decode_sint64<E: Error, R: Iterator<Item = Result<u8, E>>>(
    reader: &mut R,
) -> Result<i64, ProtoshimError<E>> {
    let zigzag = decode_varint_u64_internal(reader, "sint64")?;
    let decoded = ((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64));
    Ok(decoded)
}

pub fn encode_sint32(value: i32) -> Vec<u8> {
    let zigzag = ((value << 1) ^ (value >> 31)) as u32;
    encode_varint_u32(zigzag)
}

pub fn decode_sint32<E: Error, R: Iterator<Item = Result<u8, E>>>(
    reader: &mut R,
) -> Result<i32, ProtoshimError<E>> {
    let zigzag = decode_varint_u32_internal(reader, "sint32")?;
    let decoded = ((zigzag >> 1) as i32) ^ (-((zigzag & 1) as i32));
    Ok(decoded)
}

pub fn encode_uint64(value: u64) -> Vec<u8> {
    encode_varint_u64(value)
}

pub fn decode_uint64<E: Error, R: Iterator<Item = Result<u8, E>>>(
    reader: &mut R,
) -> Result<u64, ProtoshimError<E>> {
    decode_varint_u64_internal(reader, "uint64")
}

pub fn encode_uint32(value: u32) -> Vec<u8> {
    encode_varint_u32(value)
}

pub fn decode_uint32<E: Error, R: Iterator<Item = Result<u8, E>>>(
    reader: &mut R,
) -> Result<u32, ProtoshimError<E>> {
    decode_varint_u32_internal(reader, "uint32")
}

pub fn encode_double(value: f64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

pub fn decode_double<E: Error, R: Iterator<Item = Result<u8, E>>>(
    reader: &mut R,
) -> Result<f64, ProtoshimError<E>> {
    let mut buffer = [0u8; 8];
    let mut bytes_read = 0;

    for i in 0..8 {
        match reader.next() {
            Some(Ok(byte)) => {
                buffer[i] = byte;
                bytes_read += 1;
            }
            Some(Err(err)) => {
                return Err(ProtoshimError::IteratorError(err));
            }
            None => {
                return Err(ProtoshimError::Incomplete {
                    expected: 8,
                    got: bytes_read,
                    at: bytes_read,
                });
            }
        }
    }

    Ok(f64::from_le_bytes(buffer))
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

fn decode_varint_u64_internal<E: Error, R: Iterator<Item = Result<u8, E>>>(
    reader: &mut R,
    data_type: &'static str,
) -> Result<u64, ProtoshimError<E>> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;
    let mut read_bytes = Vec::new();

    loop {
        match reader.next() {
            Some(Ok(byte)) => {
                bytes_read += 1;
                read_bytes.push(byte);

                result |= ((byte & 0x7F) as u64) << shift;

                if (byte & 0x80) == 0 {
                    return Ok(result);
                }

                if bytes_read >= 10 {
                    return Err(ProtoshimError::ExcessiveLength {
                        input: read_bytes,
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
            Some(Err(err)) => {
                return Err(ProtoshimError::IteratorError(err));
            }
            None => {
                if bytes_read == 0 {
                    return Err(ProtoshimError::Incomplete {
                        expected: 1,
                        got: 0,
                        at: 0,
                    });
                } else {
                    return Err(ProtoshimError::Incomplete {
                        expected: bytes_read + 1,
                        got: bytes_read,
                        at: bytes_read,
                    });
                }
            }
        }
    }
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

fn decode_varint_u32_internal<E: Error, R: Iterator<Item = Result<u8, E>>>(
    reader: &mut R,
    data_type: &'static str,
) -> Result<u32, ProtoshimError<E>> {
    let mut result = 0u32;
    let mut shift = 0;
    let mut bytes_read = 0;
    let mut read_bytes = Vec::new();

    loop {
        match reader.next() {
            Some(Ok(byte)) => {
                bytes_read += 1;
                read_bytes.push(byte);

                result |= ((byte & 0x7F) as u32) << shift;

                if (byte & 0x80) == 0 {
                    return Ok(result);
                }

                if bytes_read >= 5 {
                    return Err(ProtoshimError::ExcessiveLength {
                        input: read_bytes,
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
            Some(Err(err)) => {
                return Err(ProtoshimError::IteratorError(err));
            }
            None => {
                if bytes_read == 0 {
                    return Err(ProtoshimError::Incomplete {
                        expected: 1,
                        got: 0,
                        at: 0,
                    });
                } else {
                    return Err(ProtoshimError::Incomplete {
                        expected: bytes_read + 1,
                        got: bytes_read,
                        at: bytes_read,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestError;

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "TestError")
        }
    }

    impl Error for TestError {}

    #[test]
    fn test_sint64() {
        assert_eq!(encode_sint64(0), vec![0x00]);
        assert_eq!(encode_sint64(1), vec![0x02]);
        assert_eq!(encode_sint64(-1), vec![0x01]);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x00].into_iter().map(Ok));
        let val = decode_sint64(&mut cursor).unwrap();
        assert_eq!(val, 0);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x02].into_iter().map(Ok));
        let val = decode_sint64(&mut cursor).unwrap();
        assert_eq!(val, 1);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x01].into_iter().map(Ok));
        let val = decode_sint64(&mut cursor).unwrap();
        assert_eq!(val, -1);
    }

    #[test]
    fn test_sint32() {
        assert_eq!(encode_sint32(0), vec![0x00]);
        assert_eq!(encode_sint32(1), vec![0x02]);
        assert_eq!(encode_sint32(-1), vec![0x01]);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x00].into_iter().map(Ok));
        let val = decode_sint32(&mut cursor).unwrap();
        assert_eq!(val, 0);
    }

    #[test]
    fn test_uint64() {
        assert_eq!(encode_uint64(0), vec![0x00]);
        assert_eq!(encode_uint64(127), vec![0x7F]);
        assert_eq!(encode_uint64(128), vec![0x80, 0x01]);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x00].into_iter().map(Ok));
        let val = decode_uint64(&mut cursor).unwrap();
        assert_eq!(val, 0);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x7F].into_iter().map(Ok));
        let val = decode_uint64(&mut cursor).unwrap();
        assert_eq!(val, 127);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x80, 0x01].into_iter().map(Ok));
        let val = decode_uint64(&mut cursor).unwrap();
        assert_eq!(val, 128);
    }

    #[test]
    fn test_uint32() {
        assert_eq!(encode_uint32(0), vec![0x00]);
        assert_eq!(encode_uint32(127), vec![0x7F]);
        assert_eq!(encode_uint32(128), vec![0x80, 0x01]);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x00].into_iter().map(Ok));
        let val = decode_uint32(&mut cursor).unwrap();
        assert_eq!(val, 0);
    }

    #[test]
    fn test_double() {
        let pi = std::f64::consts::PI;
        let encoded = encode_double(pi);
        assert_eq!(encoded.len(), 8);

        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(encoded.into_iter().map(Ok));
        let decoded = decode_double(&mut cursor).unwrap();
        assert_eq!(decoded, pi);

        // Test zero
        let encoded = encode_double(0.0);
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(encoded.into_iter().map(Ok));
        let decoded = decode_double(&mut cursor).unwrap();
        assert_eq!(decoded, 0.0);

        // Test negative
        let encoded = encode_double(-123.456);
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(encoded.into_iter().map(Ok));
        let decoded = decode_double(&mut cursor).unwrap();
        assert_eq!(decoded, -123.456);
    }

    #[test]
    fn test_incomplete_errors() {
        // Test empty input
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![].into_iter().map(Ok));
        match decode_sint64(&mut cursor) {
            Err(ProtoshimError::Incomplete {
                expected: 1,
                got: 0,
                at: 0,
            }) => {} // Expected
            other => panic!("Expected Incomplete error, got: {:?}", other),
        }

        // Test incomplete double
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x00, 0x01, 0x02].into_iter().map(Ok));
        match decode_double(&mut cursor) {
            Err(ProtoshimError::Incomplete { .. }) => {} // Expected
            other => panic!("Expected Incomplete error, got: {:?}", other),
        }

        // Test incomplete varint
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(vec![0x80].into_iter().map(Ok));
        match decode_uint64(&mut cursor) {
            Err(ProtoshimError::Incomplete {
                expected: 2,
                got: 1,
                at: 1,
            }) => {} // Expected
            other => panic!("Expected Incomplete error, got: {:?}", other),
        }
    }

    #[test]
    fn test_excessive_length_errors() {
        // Test u64 overflow (all 10 bytes with continuation bit)
        let excessive_bytes = vec![0xFF; 10];
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(excessive_bytes.into_iter().map(Ok));
        match decode_uint64(&mut cursor) {
            Err(ProtoshimError::ExcessiveLength {
                data_type: "uint64",
                error_position: 9,
                at: 10,
                ..
            }) => {} // Expected
            other => panic!("Expected ExcessiveLength error, got: {:?}", other),
        }

        // Test u32 overflow (all 5 bytes with continuation bit)
        let excessive_bytes = vec![0xFF; 5];
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(excessive_bytes.into_iter().map(Ok));
        match decode_uint32(&mut cursor) {
            Err(ProtoshimError::ExcessiveLength {
                data_type: "uint32",
                error_position: 4,
                at: 5,
                ..
            }) => {} // Expected
            other => panic!("Expected ExcessiveLength error, got: {:?}", other),
        }
    }

    #[test]
    fn test_error_formatting() {
        let excessive_bytes = vec![0xFF; 10];
        let mut cursor: Box<dyn Iterator<Item = Result<u8, TestError>>> =
            Box::new(excessive_bytes.into_iter().map(Ok));
        let error = decode_uint64(&mut cursor).unwrap_err();
        let error_string = format!("{}", error);

        assert!(error_string.contains("Input sequence is excessive in length"));
        assert!(error_string.contains("0xFF 0xFF 0xFF"));
        assert!(error_string.contains("uint64 will overflow"));
    }
}
