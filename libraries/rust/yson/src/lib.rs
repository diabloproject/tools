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

impl From<std::collections::BTreeMap<String, YsonNode>> for YsonValue {
    fn from(v: std::collections::BTreeMap<String, YsonNode>) -> Self {
        YsonValue::Map(v)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct YsonNode {
    pub attributes: Option<std::collections::BTreeMap<String, YsonNode>>,
    pub value: YsonValue,
}

#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum YsonParseErrorVariant {
    IncompleteValue,
    NotImplementedError
}

pub struct YsonParseError {
    pub varitant: YsonParseErrorVariant,
    /// how many bytes since the start of the input
    pub at: usize
}

pub struct YsonParseItem<T: Sized> {
    pub at: usize,
    pub val: T
}

macro_rules! item {
    ($item: expr, $at: expr) => {YsonParseItem {at: $at, val: $item}}
}

type YsonParseResult<T> = Result<YsonParseItem<T>, YsonParseError>;


pub fn parse_once(value: &[u8]) -> YsonParseResult<YsonValue> {
    if value.len() == 0 {
        return Err(YsonParseError::IncompleteValue);
    }
    // check if first byte means something special:
    match value[0] {
        // Binary string
        0x01 => {},
        // sint64
        0x02 => {}
        // uint64
        0x03 => {}
        // %true
        0x04 => return Ok(item!(1, true.into())),
        // %false
        0x05 => return Ok(item!(1, false.into()))
    }
}


pub fn parse_complete_value(value: &[u8]) -> Result<YsonNode, YsonParseError> {
    return Err(YsonParseError::NotImplementedError)
}


#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn test_parse_int64_textual() {
        assert!(parse_complete_value(&"0".bytes().collect::<Vec<u8>>()).unwrap() == YsonNode {
            attributes: None,
            value: YsonValue::Int64(0)
        });
        assert!(parse_complete_value(&"1".bytes().collect::<Vec<u8>>()).unwrap() == YsonNode {
            attributes: None,
            value: YsonValue::Int64(1)
        });
        assert!(parse_complete_value(&"-1".bytes().collect::<Vec<u8>>()).unwrap() == YsonNode {
            attributes: None,
            value: YsonValue::Int64(-1)
        });
    }

    #[test]
    fn test_parse_int64_binary() {
        assert!(parse_complete_value(vec![0x02, 0x00]).unwrap() == YsonNode {
            attributes: None,
            value: YsonValue::Int64(0)
        });
        assert!(parse_complete_value(vec![0x02, 0x01]).unwrap() == YsonNode {
            attributes: None,
            value: YsonValue::Int64(1)
        });
        assert!(parse_complete_value(vec![0x02, 0x02]).unwrap() == YsonNode {
            attributes: None,
            value: YsonValue::Int64(-1)
        });
    }

    #[test]
    fn test_parse_double() {

    }
}
