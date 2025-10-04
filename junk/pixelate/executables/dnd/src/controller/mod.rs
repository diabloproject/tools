pub mod primitive;
pub mod optimized;

use crate::types::PixelInfo;

pub trait StoreController: Send + Sync {
    fn push_pixel(&self, x: i64, y: i64, color: u8, user_id: u64, timestamp: u64);
    fn pixel_info_at(&self, x: i64, y: i64, timestamp: u64) -> Option<PixelInfo>;
    fn snapshot(&self, timestamp: u64) -> Vec<PixelInfo>;
}
