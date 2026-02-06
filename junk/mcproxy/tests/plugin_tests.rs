use bytes::Bytes;
use mcproxy::plugin::{PacketAction, PacketDirection, PacketFilter, PacketMapper, RawPacket};
use mcproxy::PluginManager;

#[derive(Debug)]
struct TestFilter {
    allow: bool,
}

impl PacketFilter for TestFilter {
    fn name(&self) -> &str {
        "TestFilter"
    }

    fn filter(&self, _packet: &RawPacket, _direction: PacketDirection) -> bool {
        self.allow
    }
}

#[derive(Debug)]
struct TestMapper;

impl PacketMapper for TestMapper {
    fn name(&self) -> &str {
        "TestMapper"
    }

    fn map(&self, mut packet: RawPacket, _direction: PacketDirection) -> PacketAction {
        // Add a marker byte to the packet
        let mut new_data = packet.data.to_vec();
        new_data.push(0xFF);
        packet.data = Bytes::from(new_data);
        PacketAction::Forward(packet)
    }
}

#[test]
fn test_filter_allows_packet() {
    let mut manager = PluginManager::new();
    manager.add_filter(Box::new(TestFilter { allow: true }));

    let packet = RawPacket {
        data: Bytes::from_static(b"test"),
        packet_id: Some(0x00),
    };

    let result = manager.process_packet(packet, PacketDirection::Forward);
    assert!(result.is_some());
}

#[test]
fn test_filter_drops_packet() {
    let mut manager = PluginManager::new();
    manager.add_filter(Box::new(TestFilter { allow: false }));

    let packet = RawPacket {
        data: Bytes::from_static(b"test"),
        packet_id: Some(0x00),
    };

    let result = manager.process_packet(packet, PacketDirection::Forward);
    assert!(result.is_none());
}

#[test]
fn test_mapper_transforms_packet() {
    let mut manager = PluginManager::new();
    manager.add_mapper(Box::new(TestMapper));

    let packet = RawPacket {
        data: Bytes::from_static(b"test"),
        packet_id: Some(0x00),
    };

    let result = manager.process_packet(packet, PacketDirection::Forward);
    assert!(result.is_some());
    
    let result = result.unwrap();
    assert_eq!(result.data.len(), 5); // "test" + 0xFF
    assert_eq!(result.data[4], 0xFF);
}

#[test]
fn test_filter_then_mapper() {
    let mut manager = PluginManager::new();
    manager.add_filter(Box::new(TestFilter { allow: true }));
    manager.add_mapper(Box::new(TestMapper));

    let packet = RawPacket {
        data: Bytes::from_static(b"test"),
        packet_id: Some(0x00),
    };

    let result = manager.process_packet(packet, PacketDirection::Forward);
    assert!(result.is_some());
    
    let result = result.unwrap();
    assert_eq!(result.data.len(), 5);
}

#[test]
fn test_filter_blocks_mapper() {
    let mut manager = PluginManager::new();
    manager.add_filter(Box::new(TestFilter { allow: false }));
    manager.add_mapper(Box::new(TestMapper));

    let packet = RawPacket {
        data: Bytes::from_static(b"test"),
        packet_id: Some(0x00),
    };

    let result = manager.process_packet(packet, PacketDirection::Forward);
    // Filter drops packet before mapper runs
    assert!(result.is_none());
}

#[test]
fn test_packet_direction() {
    let forward = PacketDirection::Forward;
    let backward = PacketDirection::Backward;

    assert_eq!(forward, PacketDirection::Forward);
    assert_eq!(backward, PacketDirection::Backward);
    assert_ne!(forward, backward);
}
