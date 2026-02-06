use crate::plugin::{PacketAction, PacketDirection, PacketMapper, RawPacket};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub struct BandwidthLimiter {
    bytes_sent: Arc<AtomicUsize>,
    max_bytes_per_packet: usize,
}

impl BandwidthLimiter {
    pub fn new(max_bytes_per_packet: usize) -> Self {
        Self {
            bytes_sent: Arc::new(AtomicUsize::new(0)),
            max_bytes_per_packet,
        }
    }
}

impl PacketMapper for BandwidthLimiter {
    fn name(&self) -> &str {
        "BandwidthLimiter"
    }

    fn map(&self, packet: RawPacket, direction: PacketDirection) -> PacketAction {
        if direction == PacketDirection::Forward {
            let size = packet.data.len();
            let total = self.bytes_sent.fetch_add(size, Ordering::Relaxed);
            
            if size > self.max_bytes_per_packet {
                tracing::warn!(
                    "Dropping oversized packet: {} bytes (limit: {})",
                    size,
                    self.max_bytes_per_packet
                );
                return PacketAction::Drop;
            }
            
            tracing::trace!("Total bytes forwarded: {}", total + size);
        }
        
        PacketAction::Forward(packet)
    }
}
