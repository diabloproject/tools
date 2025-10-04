pub mod ram;
pub mod disk;

use crate::types::PixelInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VNode {
    pub next: Option<Box<VNode>>,
    pub side: PixelInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    pub width: u64,
    pub height: u64,
    pub center_x: i64,
    pub center_y: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackingStoreDescriptor {
    pub frame: Frame,
}

pub trait BackingStore: Send + Sync {
    fn describe(&self) -> BackingStoreDescriptor;
    fn get_changelog(&self, x: i64, y: i64) -> Option<VNode>;
    fn append_change(&self, x: i64, y: i64, color: u8, user_id: u64, timestamp: u64);
}
