use crate::types::{EntityId, ResourceInstanceId};

#[derive(Copy, Clone)]
pub struct OperationId(usize);

impl std::fmt::Debug for OperationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

impl std::fmt::Display for OperationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

pub fn oid(n: usize) -> OperationId {
    OperationId(n)
}

pub struct Operation {
    pub id: OperationId,
    pub source_eid: EntityId,
    pub target_eid: EntityId,
    pub riid: ResourceInstanceId,
}