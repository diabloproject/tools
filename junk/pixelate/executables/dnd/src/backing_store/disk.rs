use crate::backing_store::{BackingStore, BackingStoreDescriptor, Frame, VNode};
use crate::types::PixelInfo;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use tracing::{debug, error, trace, warn};

/// Disk-backed storage for pixel changelogs using chunked storage
/// 
/// Storage strategy:
/// - Pixels are organized into chunks (e.g., 256x256 tiles)
/// - Each chunk has its own file
/// - Within a chunk file, pixels are stored with fixed offsets
/// - Each pixel slot stores: [exists:u8][data_length:u32][serialized_vnode]
/// - This keeps file count manageable even for large canvases
pub struct DiskBackingStore {
    data_dir: PathBuf,
    frame: Frame,
    chunk_size: u64,
    // Cache recently accessed changelogs in memory
    cache: RwLock<HashMap<(i64, i64), VNode>>,
    // Cache for chunk file handles to avoid constant reopening
    chunk_handles: RwLock<HashMap<(i64, i64), ()>>,
    cache_size: usize,
}

impl DiskBackingStore {
    pub fn new(frame: Frame, data_dir: impl AsRef<Path>, _initializer: fn(i64, i64) -> PixelInfo) -> io::Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let chunk_size = 256; // 256x256 pixels per chunk = 65,536 pixels per file
        
        debug!(
            "Initializing disk backing store at {:?} with size: {}x{} (offset x={};y={})",
            data_dir, frame.width, frame.height, frame.center_x, frame.center_y
        );
        debug!("Using chunk size: {}x{}", chunk_size, chunk_size);

        // Calculate number of chunks
        let chunks_x = (frame.width + chunk_size - 1) / chunk_size;
        let chunks_y = (frame.height + chunk_size - 1) / chunk_size;
        debug!("Total chunks: {}x{} = {}", chunks_x, chunks_y, chunks_x * chunks_y);

        // Create directory if it doesn't exist
        fs::create_dir_all(&data_dir)?;

        let store = Self {
            data_dir,
            frame,
            chunk_size,
            cache: RwLock::new(HashMap::new()),
            chunk_handles: RwLock::new(HashMap::new()),
            cache_size: 10000, // Keep 10000 pixels in cache
        };

        debug!("Disk backing store initialized (using lazy chunk creation)");
        Ok(store)
    }

    fn get_chunk_coords(&self, x: i64, y: i64) -> (i64, i64) {
        let chunk_x = (x + self.frame.center_x).div_euclid(self.chunk_size as i64);
        let chunk_y = (y + self.frame.center_y).div_euclid(self.chunk_size as i64);
        (chunk_x, chunk_y)
    }

    fn get_pixel_offset_in_chunk(&self, x: i64, y: i64) -> usize {
        let local_x = (x + self.frame.center_x).rem_euclid(self.chunk_size as i64) as usize;
        let local_y = (y + self.frame.center_y).rem_euclid(self.chunk_size as i64) as usize;
        local_y * self.chunk_size as usize + local_x
    }

    fn chunk_file_path(&self, chunk_x: i64, chunk_y: i64) -> PathBuf {
        self.data_dir.join(format!("chunk_{}_{}.dat", chunk_x, chunk_y))
    }

    fn serialize_node(node: &VNode) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Serialize current node
        data.extend_from_slice(&node.side.generation.to_le_bytes());
        data.extend_from_slice(&node.side.timestamp.to_le_bytes());
        data.push(node.side.color);
        data.extend_from_slice(&node.side.user_id.to_le_bytes());
        data.extend_from_slice(&node.side.x.to_le_bytes());
        data.extend_from_slice(&node.side.y.to_le_bytes());
        
        // Has next?
        if let Some(ref next) = node.next {
            data.push(1);
            data.extend_from_slice(&Self::serialize_node(next));
        } else {
            data.push(0);
        }
        
        data
    }

    fn deserialize_node(data: &[u8]) -> io::Result<VNode> {
        if data.len() < 30 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Data too short to deserialize VNode",
            ));
        }

        let generation = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let timestamp = u64::from_le_bytes([
            data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11],
        ]);
        let color = data[12];
        let user_id = u64::from_le_bytes([
            data[13], data[14], data[15], data[16],
            data[17], data[18], data[19], data[20],
        ]);
        let x = i64::from_le_bytes([
            data[21], data[22], data[23], data[24],
            data[25], data[26], data[27], data[28],
        ]);
        let y = i64::from_le_bytes([
            data[29], data[30], data[31], data[32],
            data[33], data[34], data[35], data[36],
        ]);
        let has_next = data[37];

        let next = if has_next == 1 {
            Some(Box::new(Self::deserialize_node(&data[38..])?))
        } else {
            None
        };

        Ok(VNode {
            side: PixelInfo {
                x,
                y,
                color,
                timestamp,
                user_id,
                generation,
            },
            next,
        })
    }

    fn read_pixel_from_chunk(&self, x: i64, y: i64) -> io::Result<VNode> {
        let (chunk_x, chunk_y) = self.get_chunk_coords(x, y);
        let chunk_path = self.chunk_file_path(chunk_x, chunk_y);
        
        if !chunk_path.exists() {
            // Chunk doesn't exist, return default initialized pixel
            return Ok(VNode {
                side: PixelInfo {
                    x,
                    y,
                    color: 255,
                    timestamp: 0,
                    user_id: 0,
                    generation: 0,
                },
                next: None,
            });
        }

        let mut file = File::open(&chunk_path)?;
        let pixel_offset = self.get_pixel_offset_in_chunk(x, y);
        
        // Each pixel entry: [exists:u8][data_length:u32][data...]
        let entry_start = pixel_offset * (1 + 4 + 8192); // Max 8KB per pixel entry
        file.seek(SeekFrom::Start(entry_start as u64))?;
        
        let mut exists = [0u8; 1];
        file.read_exact(&mut exists)?;
        
        if exists[0] == 0 {
            // Pixel not initialized in this chunk
            return Ok(VNode {
                side: PixelInfo {
                    x,
                    y,
                    color: 255,
                    timestamp: 0,
                    user_id: 0,
                    generation: 0,
                },
                next: None,
            });
        }
        
        let mut length_bytes = [0u8; 4];
        file.read_exact(&mut length_bytes)?;
        let data_length = u32::from_le_bytes(length_bytes) as usize;
        
        if data_length > 8192 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Pixel data exceeds maximum size",
            ));
        }
        
        let mut data = vec![0u8; data_length];
        file.read_exact(&mut data)?;
        
        Self::deserialize_node(&data)
    }

    fn write_pixel_to_chunk(&self, x: i64, y: i64, node: &VNode) -> io::Result<()> {
        let (chunk_x, chunk_y) = self.get_chunk_coords(x, y);
        let chunk_path = self.chunk_file_path(chunk_x, chunk_y);
        
        // Ensure chunk file exists
        if !chunk_path.exists() {
            let pixels_per_chunk = self.chunk_size * self.chunk_size;
            let entry_size = 1 + 4 + 8192; // exists + length + max data
            let chunk_file_size = pixels_per_chunk * entry_size;
            
            let file = File::create(&chunk_path)?;
            file.set_len(chunk_file_size)?;
            trace!("Created chunk file {:?} with size {}", chunk_path, chunk_file_size);
        }
        
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&chunk_path)?;
        
        let pixel_offset = self.get_pixel_offset_in_chunk(x, y);
        let entry_start = pixel_offset * (1 + 4 + 8192);
        
        file.seek(SeekFrom::Start(entry_start as u64))?;
        
        let data = Self::serialize_node(node);
        if data.len() > 8192 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Pixel data exceeds maximum size",
            ));
        }
        
        // Write: exists flag + data length + data
        file.write_all(&[1u8])?;
        file.write_all(&(data.len() as u32).to_le_bytes())?;
        file.write_all(&data)?;
        file.sync_data()?;
        
        Ok(())
    }

    fn evict_cache_if_needed(&self) {
        let mut cache = self.cache.write().unwrap();
        if cache.len() > self.cache_size {
            // Remove 10% of entries
            let to_remove = cache.len() / 10;
            let keys: Vec<_> = cache.keys().take(to_remove).cloned().collect();
            for key in keys {
                cache.remove(&key);
            }
            trace!("Evicted {} entries from cache", to_remove);
        }
    }
}

impl BackingStore for DiskBackingStore {
    fn describe(&self) -> BackingStoreDescriptor {
        BackingStoreDescriptor { frame: self.frame }
    }

    fn get_changelog(&self, x: i64, y: i64) -> Option<VNode> {
        trace!("Looking for pixel ({}, {}) in disk backing store", x, y);
        
        // Check bounds
        if x < 0 - self.frame.center_x || x >= self.frame.width as i64 - self.frame.center_x {
            trace!("Pixel ({}, {}) is out of bounds", x, y);
            return None;
        }
        if y < 0 - self.frame.center_y || y >= self.frame.height as i64 - self.frame.center_y {
            trace!("Pixel ({}, {}) is out of bounds", y, y);
            return None;
        }

        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(node) = cache.get(&(x, y)) {
                trace!("Pixel ({}, {}) found in cache", x, y);
                return Some(node.clone());
            }
        }

        // Read from disk
        match self.read_pixel_from_chunk(x, y) {
            Ok(node) => {
                trace!("Pixel ({}, {}) loaded from disk (color {})", x, y, node.side.color);
                
                // Update cache
                {
                    let mut cache = self.cache.write().unwrap();
                    cache.insert((x, y), node.clone());
                }
                self.evict_cache_if_needed();
                
                Some(node)
            }
            Err(e) => {
                error!("Failed to read pixel ({}, {}) from disk: {}", x, y, e);
                None
            }
        }
    }

    fn append_change(&self, x: i64, y: i64, color: u8, user_id: u64, timestamp: u64) {
        trace!("Appending change to pixel ({}, {}) in disk backing store", x, y);
        
        let Some(old_changelog) = self.get_changelog(x, y) else {
            warn!("Cannot append change to non-existent pixel ({}, {})", x, y);
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

        let new_node = VNode {
            side: pixel_info,
            next: Some(Box::new(old_changelog)),
        };

        // Write to disk
        if let Err(e) = self.write_pixel_to_chunk(x, y, &new_node) {
            error!("Failed to write pixel ({}, {}) to disk: {}", x, y, e);
            return;
        }

        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert((x, y), new_node);
        }

        trace!("Pixel ({}, {}) appended", x, y);
    }
}
