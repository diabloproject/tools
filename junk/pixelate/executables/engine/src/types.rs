#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelInfo {
    pub x: i64,
    pub y: i64,
    pub color: u8,
    pub timestamp: u64,
    pub user_id: u64,
    pub generation: u32
}
