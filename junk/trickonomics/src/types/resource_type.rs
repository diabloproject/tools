use std::fmt::{Debug, Display};

#[derive(Copy, Clone)]
pub struct ResourceTypeId(usize);

impl Debug for ResourceTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

impl Display for ResourceTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

pub(crate) fn rid(n: usize) -> ResourceTypeId {
    ResourceTypeId(n)
}

pub struct ResourceType {
    pub id: ResourceTypeId,
}
