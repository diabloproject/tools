use std::fmt::Display;

use thiserror::Error;

pub type YsonString = Vec<u8>;

#[derive(Debug, Clone, PartialEq)]
pub enum YsonValue {
    String(YsonString),
    Int64(i64),
    Uint64(u64),
    Double(f64),
    Boolean(bool),
    Entity,
    List(Vec<YsonNode>),
    Map(std::collections::BTreeMap<YsonString, YsonNode>),
}

impl From<YsonString> for YsonValue {
    fn from(v: YsonString) -> Self {
        YsonValue::String(v)
    }
}

impl From<&str> for YsonValue {
    fn from(v: &str) -> Self {
        YsonValue::String(v.bytes().collect())
    }
}

impl From<i64> for YsonValue {
    fn from(v: i64) -> Self {
        YsonValue::Int64(v)
    }
}

impl From<u64> for YsonValue {
    fn from(v: u64) -> Self {
        YsonValue::Uint64(v)
    }
}

impl From<f64> for YsonValue {
    fn from(v: f64) -> Self {
        YsonValue::Double(v)
    }
}

impl From<bool> for YsonValue {
    fn from(v: bool) -> Self {
        YsonValue::Boolean(v)
    }
}

impl From<Vec<YsonNode>> for YsonValue {
    fn from(v: Vec<YsonNode>) -> Self {
        YsonValue::List(v)
    }
}

impl From<std::collections::BTreeMap<YsonString, YsonNode>> for YsonValue {
    fn from(v: std::collections::BTreeMap<YsonString, YsonNode>) -> Self {
        YsonValue::Map(v)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct YsonNode {
    pub attributes: Option<std::collections::BTreeMap<String, YsonNode>>,
    pub value: YsonValue,
}

#[non_exhaustive]
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum YsonParseErrorVariant {
    #[error("Reached EOF before value was completed")]
    IncompleteValue,
    #[error("Parser was not implemented")]
    NotImplementedError,
    #[error("Binary representation overflow")]
    BinaryOverflow,
}

#[derive(Error, Debug)]
pub struct YsonParseError {
    pub variant: YsonParseErrorVariant,
    /// how many bytes since the start of the input
    pub at: usize,
}

impl Display for YsonParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reached error \"{}\" {} bytes into the sequence",
            self.variant, self.at
        )
    }
}

impl YsonParseError {
    pub fn incomplete(at: usize) -> Self {
        return Self {
            variant: YsonParseErrorVariant::IncompleteValue,
            at,
        };
    }

    pub fn todo(at: usize) -> Self {
        return Self {
            variant: YsonParseErrorVariant::NotImplementedError,
            at,
        };
    }
}

pub struct YsonParseItem<T: Sized> {
    pub at: usize,
    pub value: T,
}

macro_rules! item {
    ($item: expr, $at: expr) => {
        YsonParseItem {
            at: $at,
            value: $item,
        }
    };
}

type YsonParseResult<T> = Result<YsonParseItem<T>, YsonParseError>;

pub fn parse_once(value: &[u8]) -> YsonParseResult<YsonValue> {
    if value.len() == 0 {
        return Err(YsonParseError::incomplete(0));
    }
    // check if first byte means something special:
    match value[0] {
        // Binary string
        0x01 => return Err(YsonParseError::todo(1)),
        // sint64
        0x02 => {
            let mut result = 0u64;
            let mut shift = 0;
            let mut bytes_read = 0;

            // Varint decode
            for &byte in &value[1..] {
                bytes_read += 1;

                if bytes_read > 10 || shift >= 64 {
                    return Err(YsonParseError {
                        variant: YsonParseErrorVariant::BinaryOverflow,
                        at: 1 + bytes_read,
                    });
                }

                result |= ((byte & 0x7F) as u64) << shift;

                if (byte & 0x80) == 0 {
                    // ZigZag decode: (n >> 1) ^ -(n & 1)
                    let decoded = ((result >> 1) as i64) ^ (-((result & 1) as i64));
                    return Ok(item!(decoded.into(), bytes_read + 1));
                }

                shift += 7;
            }
            return Err(YsonParseError::incomplete(bytes_read + 1));
        }
        // uint64
        0x03 => return Err(YsonParseError::todo(1)),
        // true
        0x04 => return Ok(item!(true.into(), 1)),
        // false
        0x05 => return Ok(item!(false.into(), 1)),
        _ => return Err(YsonParseError::todo(0)),
    }
}

pub fn parse_complete_value(value: &[u8]) -> Result<YsonNode, YsonParseError> {
    return Ok(YsonNode {
        value: parse_once(value)?.value,
        attributes: None,
    });
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn test_parse_int64_textual() {
        assert!(
            parse_complete_value(&"0".bytes().collect::<Vec<u8>>()).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Int64(0)
                }
        );
        assert!(
            parse_complete_value(&"1".bytes().collect::<Vec<u8>>()).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Int64(1)
                }
        );
        assert!(
            parse_complete_value(&"-1".bytes().collect::<Vec<u8>>()).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Int64(-1)
                }
        );
    }

    #[test]
    fn test_parse_int64_binary() {
        assert!(
            parse_complete_value(&vec![0x02, 0x00]).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Int64(0)
                }
        );
        assert!(
            parse_complete_value(&vec![0x02, 0x02]).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Int64(1)
                }
        );
        assert!(
            parse_complete_value(&vec![0x02, 0x01]).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Int64(-1)
                }
        );
    }

    #[test]
    fn test_parse_double() {todo!()}
}
