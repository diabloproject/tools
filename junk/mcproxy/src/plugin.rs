use bytes::Bytes;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketDirection {
    Forward,  // client -> upstream
    Backward, // upstream -> client
}

pub struct RawPacket {
    pub data: Bytes,
    pub packet_id: Option<i32>,
}

pub enum PacketAction {
    Forward(RawPacket),
    Drop,
}

pub trait PacketFilter: Send + Sync + Debug {
    fn name(&self) -> &str;
    
    fn filter(&self, packet: &RawPacket, direction: PacketDirection) -> bool;
}

pub trait PacketMapper: Send + Sync + Debug {
    fn name(&self) -> &str;
    
    fn map(&self, packet: RawPacket, direction: PacketDirection) -> PacketAction;
}

pub trait PacketInjector: Send + Sync + Debug {
    fn name(&self) -> &str;
    
    fn should_inject(&self, direction: PacketDirection) -> bool;
    
    fn inject(&self, direction: PacketDirection) -> Vec<RawPacket>;
}

pub struct PluginManager {
    filters: Vec<Box<dyn PacketFilter>>,
    mappers: Vec<Box<dyn PacketMapper>>,
    injectors: Vec<Box<dyn PacketInjector>>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            mappers: Vec::new(),
            injectors: Vec::new(),
        }
    }

    pub fn add_filter(&mut self, filter: Box<dyn PacketFilter>) {
        tracing::info!("Registered filter: {}", filter.name());
        self.filters.push(filter);
    }

    pub fn add_mapper(&mut self, mapper: Box<dyn PacketMapper>) {
        tracing::info!("Registered mapper: {}", mapper.name());
        self.mappers.push(mapper);
    }

    pub fn add_injector(&mut self, injector: Box<dyn PacketInjector>) {
        tracing::info!("Registered injector: {}", injector.name());
        self.injectors.push(injector);
    }

    pub fn process_packet(
        &self,
        mut packet: RawPacket,
        direction: PacketDirection,
    ) -> Option<RawPacket> {
        for filter in &self.filters {
            if !filter.filter(&packet, direction) {
                tracing::debug!(
                    "Packet filtered by {} (direction: {:?})",
                    filter.name(),
                    direction
                );
                return None;
            }
        }

        for mapper in &self.mappers {
            match mapper.map(packet, direction) {
                PacketAction::Forward(new_packet) => {
                    packet = new_packet;
                }
                PacketAction::Drop => {
                    tracing::debug!(
                        "Packet dropped by mapper {} (direction: {:?})",
                        mapper.name(),
                        direction
                    );
                    return None;
                }
            }
        }

        Some(packet)
    }

    pub fn get_injected_packets(&self, direction: PacketDirection) -> Vec<RawPacket> {
        let mut packets = Vec::new();
        for injector in &self.injectors {
            if injector.should_inject(direction) {
                packets.extend(injector.inject(direction));
            }
        }
        packets
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}
