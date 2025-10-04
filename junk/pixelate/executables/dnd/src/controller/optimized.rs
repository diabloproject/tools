use crate::backing_store::BackingStore;
use crate::controller::{PixelInfo, StoreController};
use std::sync::Arc;
use std::sync::RwLock;
use tracing::{debug, trace};

/// Optimized controller with cached current snapshot and sharded locks
///
/// Strategy: Keep the entire current state of the canvas in memory.
/// - Reads are instant (just index into the cached snapshot)
/// - Writes update both the backing store AND the cache
/// - Snapshot operations are O(1) - just clone the cache
/// - Uses sharded RwLocks to reduce contention on multi-core systems
pub struct OptimizedStoreController {
    pub backing: Arc<dyn BackingStore>,
    // Cached current snapshot - sharded by row to reduce lock contention
    // Each shard covers multiple rows
    snapshot_shards: Vec<RwLock<Vec<PixelInfo>>>,
    shard_count: usize,
    width: usize,
    height: usize,
    center_x: i64,
    center_y: i64,
}

impl OptimizedStoreController {
    pub fn new(backing_store: Arc<dyn BackingStore>) -> Self {
        debug!("Initializing optimized store controller with snapshot caching");

        let descriptor = backing_store.describe();
        let frame = descriptor.frame;
        let width = frame.width as usize;
        let height = frame.height as usize;

        // Use 16 shards for good parallelism on multicore systems
        // This reduces lock contention significantly
        let shard_count = 16;

        debug!(
            "Building an initial snapshot cache for {}x{} pixels ({} total)",
            width,
            height,
            width * height
        );
        debug!("Using {} shards for lock-free parallelism", shard_count);

        let mut snapshot_shards = Vec::with_capacity(shard_count);

        for shard_idx in 0..shard_count {
            let mut shard_data = Vec::with_capacity((height * width / shard_count) + 1);

            for y in 0..height {
                if y % shard_count != shard_idx {
                    continue;
                }
                for x in 0..width {
                    let global_x = x as i64 + (0 - frame.center_x);
                    let global_y = y as i64 + (0 - frame.center_y);

                    let node = backing_store
                        .get_changelog(global_x, global_y)
                        .expect("backing store missing pixel during initialization");
                    shard_data.push(node.side);
                }
            }

            snapshot_shards.push(RwLock::new(shard_data));
        }

        debug!(
            "Store controller initialized with {} shards, {} cached pixels",
            snapshot_shards.len(),
            width * height
        );

        Self {
            backing: backing_store,
            snapshot_shards,
            shard_count,
            width,
            height,
            center_x: frame.center_x,
            center_y: frame.center_y,
        }
    }

    fn coords_to_shard_and_index(&self, x: i64, y: i64) -> Option<(usize, usize)> {
        let adjusted_x = x + self.center_x;
        let adjusted_y = y + self.center_y;

        if adjusted_x < 0 || adjusted_x >= self.width as i64 {
            return None;
        }
        if adjusted_y < 0 || adjusted_y >= self.height as i64 {
            return None;
        }

        let row = adjusted_y as usize;
        let col = adjusted_x as usize;

        let shard_idx = row % self.shard_count;
        let row_in_shard = row / self.shard_count;
        let index_in_shard = row_in_shard * self.width + col;

        Some((shard_idx, index_in_shard))
    }
}

impl StoreController for OptimizedStoreController {
    fn push_pixel(&self, x: i64, y: i64, color: u8, user_id: u64, timestamp: u64) {
        trace!("Pushing pixel ({}, {}) with color {}", x, y, color);

        // Write to backing store first
        self.backing.append_change(x, y, color, user_id, timestamp);

        // Update the cached snapshot in the appropriate shard
        if let Some((shard_idx, index)) = self.coords_to_shard_and_index(x, y) {
            let mut shard = self.snapshot_shards[shard_idx].write().unwrap();
            let old_generation = shard[index].generation;
            shard[index] = PixelInfo {
                x,
                y,
                color,
                timestamp,
                user_id,
                generation: old_generation + 1,
            };
            trace!(
                "Updated cache shard {} index {} for pixel ({}, {})",
                shard_idx, index, x, y
            );
        }
    }

    fn pixel_info_at(&self, x: i64, y: i64, timestamp: u64) -> Option<PixelInfo> {
        // For current state (timestamp == 0), use cache
        if timestamp == 0 {
            let (shard_idx, index) = self.coords_to_shard_and_index(x, y)?;
            let shard = self.snapshot_shards[shard_idx].read().unwrap();
            return Some(shard[index]);
        }

        // For historical queries, traverse backing store
        let mut node = self.backing.get_changelog(x, y)?;

        if node.side.timestamp <= timestamp {
            return Some(node.side);
        }

        while let Some(ref next_node) = node.next {
            if next_node.side.timestamp <= timestamp {
                return Some(next_node.side);
            }
            node = *next_node.clone();
        }

        None
    }

    fn snapshot(&self, timestamp: u64) -> Vec<PixelInfo> {
        // For current state (timestamp == 0), just collect from all shards - O(1) operation!
        if timestamp == 0 {
            let mut result = Vec::with_capacity(self.width * self.height);

            for shard in &self.snapshot_shards {
                let shard_data = shard.read().unwrap();
                result.extend_from_slice(&shard_data);
            }

            debug!(
                "Returning cached snapshot with {} pixels (instant!)",
                result.len()
            );
            return result;
        }

        // For historical snapshots, we need to query the backing store
        debug!("Generating historical snapshot at timestamp {}", timestamp);
        let mut infos = Vec::with_capacity(self.width * self.height);

        for y in (0 - self.center_y)..(self.height as i64 - self.center_y) {
            for x in (0 - self.center_x)..(self.width as i64 - self.center_x) {
                infos.push(
                    self.pixel_info_at(x, y, timestamp)
                        .expect("backing store lied about accessible pixels"),
                )
            }
        }

        debug!("Historical snapshot generated with {} pixels", infos.len());
        infos
    }
}
