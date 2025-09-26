mod protoshim;
mod string;
mod value;
mod error;
use crate::value::{ YsonValue, YsonNode };
use crate::error::YsonParseError;

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

pub fn parse_once(bytes: &[u8]) -> YsonParseResult<YsonValue> {
    if bytes.len() == 0 {
        return Err(YsonParseError::incomplete(0));
    }
    // check if first byte means something special:
    match bytes[0] {
        // Binary string
        0x01 => {
            let (buffer_length, bytes_read) = protoshim::decode_sint64(&bytes[1..])?;
            let buffer_length = buffer_length as usize;
            let contents = &bytes[bytes_read + 1..bytes_read + 1 + buffer_length];
            if contents.len() != buffer_length {
                return Err(YsonParseError::incomplete(1 + buffer_length + bytes_read));
            }
            return Ok(YsonParseItem {
                at: buffer_length + bytes_read + 1,
                value: contents.to_vec().into(),
            });
        }
        // sint64
        0x02 => {
            return Ok({
                let res = protoshim::decode_sint64(&bytes[1..])?;
                YsonParseItem {
                    at: res.1,
                    value: res.0.into(),
                }
            })
        }
        // double
        0x03 => {
            return Ok({
                let res = protoshim::decode_double(&bytes[1..])?;
                YsonParseItem {
                    at: res.1,
                    value: res.0.into(),
                }
            })
        }
        // true
        0x04 => return Ok(item!(true.into(), 1)),
        // false
        0x05 => return Ok(item!(false.into(), 1)),
        // uint64
        0x06 => {
            return Ok({
                let res = protoshim::decode_uint64(&bytes[1..])?;
                YsonParseItem {
                    at: res.1,
                    value: res.0.into(),
                }
            })
        }
        _ => {}
    }

    // None of byte patterns matched, so we conclude that this is text representation

    fn match_digits<'a>(bytes: &'a [u8]) -> (&'a str, usize) {
        let mut cur = 0;
        while bytes[cur].is_ascii_digit() {
            cur += 1;
        }
        return (unsafe { str::from_utf8_unchecked(&bytes[0..cur]) }, cur);
    }

    if match_digits(bytes).1 != 0 || bytes[0] == b'+' || bytes[0] == b'-' {
        // Numbers
    } else if bytes[0] == b'"' || bytes[0].is_ascii_alphabetic() || bytes[0] == b'_' {
        // String literals
    } else if bytes[0] == b'%' {
        // Booleans
    } else if bytes[0] == b'#' {
        // Entity
        return Ok(YsonParseItem {
            at: 1,
            value: YsonValue::Entity,
        });
    } else {
        // Erorr
    }

    return Err(YsonParseError::todo(0));
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
    fn test_parse_uint64_binary() {
        assert!(
            parse_complete_value(&vec![0x06, 0x00]).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Uint64(0)
                }
        );
        assert!(
            parse_complete_value(&vec![0x06, 0x01]).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Uint64(1)
                }
        );
        assert!(
            parse_complete_value(&vec![0x06, 0x02]).unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Uint64(2)
                }
        );
    }

    #[test]
    fn test_parse_double_binary() {
        assert!(
            parse_complete_value(&vec![0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
                .unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Double(0.0)
                }
        );
        assert!(
            parse_complete_value(&vec![0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F])
                .unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Double(1.0)
                }
        );
        assert!(
            parse_complete_value(&vec![0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xBF])
                .unwrap()
                == YsonNode {
                    attributes: None,
                    value: YsonValue::Double(-1.0)
                }
        );
    }
}
