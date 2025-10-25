pub type YsonString = Vec<u8>;

#[derive(Debug, Clone, PartialEq)]
pub enum YsonValue {
    Array(Vec<YsonNode>),
    Map(Vec<(YsonString, YsonNode)>),
    Entity,
    Boolean(bool),
    SignedInteger(i64),
    UnsignedInteger(u64),
    Double(f64),
    String(YsonString),
}

#[derive(Debug, Clone, PartialEq)]
pub struct YsonNode {
    pub value: YsonValue,
    pub attributes: Vec<(YsonString, YsonNode)>,
}