use crate::backing_store::{BackingStore, BackingStoreDescriptor, Frame, VNode};
use crate::types::PixelInfo;
use std::sync::RwLock;
use tracing::{debug, trace};

pub struct RamBackingStore {
    // Shard changelogs by column with interleaved sharding for better clustering
    changelog_shards: Vec<RwLock<Vec<Vec<VNode>>>>,
    shard_count: usize,
    frame: Frame,
}

impl RamBackingStore {
    pub fn new(frame: Frame, initializer: fn(i64, i64) -> PixelInfo) -> Self {
        debug!(
            "Initializing RAM backing store with size: {}x{} (offset x={};y={})",
            frame.width, frame.height, frame.center_x, frame.center_y
        );
        
        // Use 16 shards for good parallelism
        let shard_count = 16;
        
        debug!("Using {} shards with interleaved column distribution", shard_count);
        
        let mut changelog_shards = Vec::with_capacity(shard_count);
        
        // Initialize shards with interleaved columns
        // Shard 0: cols 0, 16, 32, 48...
        // Shard 1: cols 1, 17, 33, 49...
        // etc.
        for shard_idx in 0..shard_count {
            let mut shard_changelogs = Vec::new();
            
            for col in (shard_idx..(frame.width as usize)).step_by(shard_count) {
                let x = col as i64 + (0 - frame.center_x);
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
                shard_changelogs.push(column);
            }
            
            changelog_shards.push(RwLock::new(shard_changelogs));
        }
        
        debug!("RAM backing store initialized with {} shards (interleaved)", changelog_shards.len());
        Self {
            changelog_shards,
            shard_count,
            frame,
        }
    }
    
    fn get_shard_and_column(&self, x: i64) -> Option<(usize, usize)> {
        let adjusted_x = (x + self.frame.center_x) as usize;
        if adjusted_x >= self.frame.width as usize {
            return None;
        }
        
        // Interleaved sharding: consecutive columns go to different shards
        let shard_idx = adjusted_x % self.shard_count;
        let col_in_shard = adjusted_x / self.shard_count;
        
        Some((shard_idx, col_in_shard))
    }
}

impl BackingStore for RamBackingStore {
    fn describe(&self) -> BackingStoreDescriptor {
        BackingStoreDescriptor { frame: self.frame }
    }

    fn get_changelog(&self, x: i64, y: i64) -> Option<VNode> {
        trace!("Looking for pixel ({}, {}) in RAM backing store", x, y);
        
        if x < 0 - self.frame.center_x || x >= self.frame.width as i64 - self.frame.center_x {
            trace!("Pixel ({}, {}) is out of bounds (x)", x, y);
            return None;
        }
        if y < 0 - self.frame.center_y || y >= self.frame.height as i64 - self.frame.center_y {
            trace!("Pixel ({}, {}) is out of bounds (y)", x, y);
            return None;
        }
        
        let (shard_idx, col_in_shard) = self.get_shard_and_column(x)?;
        let adjusted_y = (y + self.frame.center_y) as usize;
        
        let shard = self.changelog_shards[shard_idx].read().unwrap();
        let res = shard[col_in_shard][adjusted_y].clone();
        
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
        
        let (shard_idx, col_in_shard) = self.get_shard_and_column(x).unwrap();
        let adjusted_y = (y + self.frame.center_y) as usize;
        
        let mut shard = self.changelog_shards[shard_idx].write().unwrap();
        shard[col_in_shard][adjusted_y] = VNode {
            side: pixel_info,
            next: Some(Box::new(old_changelog)),
        };
        
        trace!("Pixel ({}, {}) appended", x, y);
    }
}
