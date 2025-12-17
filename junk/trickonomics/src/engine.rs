use crate::types::{
    Entity, EntityId, Operation, OperationId, ResourceInstance, ResourceInstanceId, ResourceType,
    ResourceTypeId, eid, oid, rid, riid,
};

pub struct Engine {
    entities: Vec<Entity>,
    resources: Vec<ResourceType>,
    resource_instances: Vec<ResourceInstance>,
    operations: Vec<Operation>,
    generation: u64,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            entities: vec![],
            resources: vec![],
            resource_instances: vec![],
            operations: vec![],
            generation: 0,
        }
    }

    /// Spawn a new entity and returns its unique `EntityId`.
    pub fn spawn(&mut self) -> EntityId {
        let eid = eid(self.entities.len() + 1);
        self.entities.push(Entity { id: eid });
        self.generation += 1;
        eid
    }

    /// Declare a new resource type.
    pub fn declare(&mut self) -> ResourceTypeId {
        let rid = rid(self.resources.len() + 1);
        self.resources.push(ResourceType { id: rid });
        self.generation += 1;
        rid
    }

    pub fn instantiate(&mut self, rid: ResourceTypeId) -> ResourceInstanceId {
        let riid = riid(self.resource_instances.len() + 1);
        self.resource_instances
            .push(ResourceInstance { id: riid, rid });
        self.generation += 1;
        riid
    }

    pub fn give(&mut self, eid: EntityId, riid: ResourceInstanceId) -> OperationId {
        let oid = oid(self.operations.len() + 1);
        self.operations.push(Operation {
            id: oid,
            source_eid: EntityId::SYSTEM,
            target_eid: eid,
            riid,
        });
        self.generation += 1;
        oid
    }

    pub fn take(&mut self, eid: EntityId, riid: ResourceInstanceId) {
        let oid = oid(self.operations.len() + 1);
        self.operations.push(Operation {
            id: oid,
            source_eid: eid,
            target_eid: EntityId::SYSTEM,
            riid,
        });
    }
}
