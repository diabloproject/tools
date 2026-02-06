use crate::plugin::{PacketDirection, PacketFilter, RawPacket};

#[derive(Debug)]
pub struct PacketLogger {
    log_forward: bool,
    log_backward: bool,
}

impl PacketLogger {
    pub fn new(log_forward: bool, log_backward: bool) -> Self {
        Self {
            log_forward,
            log_backward,
        }
    }
}

impl PacketFilter for PacketLogger {
    fn name(&self) -> &str {
        "PacketLogger"
    }

    fn filter(&self, packet: &RawPacket, direction: PacketDirection) -> bool {
        let should_log = match direction {
            PacketDirection::Forward => self.log_forward,
            PacketDirection::Backward => self.log_backward,
        };

        if should_log {
            tracing::info!(
                "{:?} packet (id: {:?}, size: {} bytes)",
                direction,
                packet.packet_id,
                packet.data.len()
            );
        }

        true
    }
}
