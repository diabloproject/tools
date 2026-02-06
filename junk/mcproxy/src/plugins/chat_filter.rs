use crate::plugin::{PacketDirection, PacketFilter, RawPacket};

#[derive(Debug)]
pub struct ChatFilter {
    blocked_words: Vec<String>,
}

impl ChatFilter {
    pub fn new(blocked_words: Vec<String>) -> Self {
        Self { blocked_words }
    }
}

impl PacketFilter for ChatFilter {
    fn name(&self) -> &str {
        "ChatFilter"
    }

    fn filter(&self, _packet: &RawPacket, direction: PacketDirection) -> bool {
        if direction == PacketDirection::Backward {
            return true;
        }

        // Note: In a real implementation, you would check if the packet is a chat packet
        // and inspect its content. This is a simplified example.
        true
    }
}
