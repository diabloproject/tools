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

trait YsonScalar {
    fn to_yson(self) -> YsonValue;
}

macro_rules! scalar {
    ($type: ty, $expr: expr) => {
        impl YsonScalar for $type {
            fn to_yson(self) -> YsonValue {
                let _f: fn($type) -> YsonValue = $expr;
                return _f(self)
            }
        }
    };
}

scalar!(i64, |item| YsonValue::Int64(item));
scalar!(u64, |item| YsonValue::Uint64(item));
scalar!(f64, |item| YsonValue::Double(item));
scalar!(bool, |item| YsonValue::Boolean(item));

impl<T: Into<Vec<u8>>> From<T> for YsonValue {
    fn from(v: T) -> Self {
        YsonValue::String(v.into())
    }
}

impl<T: YsonScalar> From<T> for YsonValue {
    fn from(v: T) -> Self {
        v.to_yson()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct YsonNode {
    pub attributes: Option<std::collections::BTreeMap<String, YsonNode>>,
    pub value: YsonValue,
}
