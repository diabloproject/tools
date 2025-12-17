#[derive(Copy, Clone)]
pub struct EntityId(usize);

impl std::fmt::Debug for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

impl EntityId {
    pub const SYSTEM: EntityId = EntityId(0);
}

pub(crate) fn eid(n: usize) -> EntityId {
    EntityId(n)
}

pub struct Entity {
    pub id: EntityId,
}
