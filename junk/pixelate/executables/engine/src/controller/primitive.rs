use crate::backing_store::BackingStore;
use crate::controller::{PixelInfo, StoreController};
use std::sync::Arc;

pub struct PrimitiveStoreController {
    pub backing: Arc<dyn BackingStore>,
}

impl PrimitiveStoreController {
    pub fn new(backing_store: Arc<dyn BackingStore>) -> Self {
        Self {
            backing: backing_store,
        }
    }
}

impl StoreController for PrimitiveStoreController {
    fn push_pixel(&self, x: i64, y: i64, color: u8, user_id: u64, timestamp: u64) {
        self.backing.append_change(x, y, color, user_id, timestamp);
    }

    fn pixel_info_at(&self, x: i64, y: i64, timestamp: u64) -> Option<PixelInfo> {
        let node = self.backing.get_changelog(x, y)?;
        if node.side.timestamp < timestamp || timestamp == 0 {
            return Some(node.side);
        }
        while let Some(ref node) = node.next {
            if node.side.timestamp < timestamp {
                return Some(node.side);
            }
        }
        None
    }

    fn snapshot(&self, timestamp: u64) -> Vec<PixelInfo> {
        let frame = self.backing.describe().frame;
        let mut infos = Vec::with_capacity(frame.width as usize * frame.height as usize);
        for x in (0 - frame.center_x as i64)..(frame.width as i64 - frame.center_x as i64) {
            for y in (0 - frame.center_y as i64)..(frame.height as i64 - frame.center_y as i64) {
                infos.push(
                    self.pixel_info_at(x, y, timestamp)
                        .map(|pi| pi)
                        .expect("backing store lied about accessible pixels"),
                )
            }
        }
        infos
    }
}
