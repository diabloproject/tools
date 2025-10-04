use crate::backing_store::{BackingStore, BackingStoreDescriptor, Frame, VNode};
use crate::types::PixelInfo;
use std::borrow::Cow;
use std::sync::RwLock;
use tracing::{debug, trace};

pub struct RamBackingStore {
    changelogs: RwLock<Vec<Vec<VNode>>>,
    frame: Frame,
}

impl RamBackingStore {
    pub fn new(frame: Frame, initializer: fn(i64, i64) -> PixelInfo) -> Self {
        debug!(
            "Initializing RAM backing store with size: {}x{} (offset x={};y={})",
            frame.width, frame.height, frame.center_x, frame.center_y
        );
        let mut changelogs = Vec::with_capacity(frame.width as usize);
        for x in (0 - frame.center_x)..(frame.width as i64 - frame.center_x) {
            let mut column = Vec::with_capacity(frame.height as usize);
            for y in (0 - frame.center_y)..(frame.height as i64 - frame.center_y) {
                let init = initializer(x, y);
                if init.x != x || init.y != y {
                    panic!("Initializer returned a wrong pixel location");
                }
                column.push(VNode {
                    side: init,
                    next: None,
                });
            }
            changelogs.push(column);
        }
        debug!("RAM backing store initialized");
        Self {
            changelogs: RwLock::new(changelogs),
            frame,
        }
    }
}

impl BackingStore for RamBackingStore {
    fn describe(&self) -> BackingStoreDescriptor {
        BackingStoreDescriptor { frame: self.frame }
    }

    fn get_changelog(&self, x: i64, y: i64) -> Option<VNode> {
        trace!("Looking for pixel ({}, {}) in RAM backing store", x, y);
        if x < 0 - self.frame.center_x
            || x > self.frame.width as i64 - self.frame.center_x
        {
            trace!("Pixel ({}, {}) is out of bounds", x, y);
            return None;
        }
        if y < 0 - self.frame.center_y
            || y > self.frame.height as i64 - self.frame.center_y
        {
            trace!("Pixel ({}, {}) is out of bounds", x, y);
            return None;
        }
        let changelogs = self.changelogs.read().unwrap();
        let res = changelogs[(x + self.frame.center_x) as usize]
            [(y + self.frame.center_y) as usize]
            .clone();
        trace!("Pixel ({}, {}) found (color {})", x, y, res.side.color);
        Some(res)
    }

    fn append_change(&self, x: i64, y: i64, color: u8, user_id: u64, timestamp: u64) {
        trace!("Appending change to pixel ({}, {}) in RAM backing store", x, y);
        let Some(old_changelog) = self.get_changelog(x, y) else {
            return;
        };

        let pixel_info = PixelInfo {
            x,
            y,
            color,
            timestamp,
            user_id,
            generation: old_changelog.side.generation + 1,
        };
        let mut changelogs = self.changelogs.write().unwrap();
        changelogs[(x + self.frame.center_x) as usize]
            [(y + self.frame.center_y) as usize] = VNode {
            side: pixel_info,
            next: Some(Box::new(old_changelog)),
        };
        trace!("Pixel ({}, {}) appended", x, y);
    }
}
