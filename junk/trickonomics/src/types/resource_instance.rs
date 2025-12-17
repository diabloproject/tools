use std::fmt::{Debug, Display};
use crate::types::ResourceTypeId;

#[derive(Copy, Clone)]
pub struct ResourceInstanceId(usize);

impl Debug for ResourceInstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

impl Display for ResourceInstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

pub(crate) fn riid(n: usize) -> ResourceInstanceId {
    ResourceInstanceId(n)
}

pub struct ResourceInstance {
    pub id: ResourceInstanceId,
    pub rid: ResourceTypeId, // ResourceTypeId
}
