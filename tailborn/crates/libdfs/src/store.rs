pub trait Store
where
    Self: Send + Sync,
{
    /// Allocate a block in the store.
    /// Returns the block ID of the allocated block.
    fn allocate_block(&self) -> Result<u64, std::io::Error>;
    /// Deallocate a block in the store.
    /// Returns an error if the block could not be deallocated.
    fn deallocate_block(&self, block_id: u64) -> Result<(), std::io::Error>;
    /// Write ids of allocated blocks to the buffer. Returns the number of blocks written.
    fn allocated_blocks(&self, buf: &mut [u64]) -> Result<usize, std::io::Error>;
    /// Returns the number of allocated blocks.
    fn allocated_blocks_count(&self) -> Result<usize, std::io::Error>;
    /// Read a block from the store.
    /// Returns the number of bytes read.
    fn read_block(&self, block_id: u64, buf: &mut [u8; 4096]) -> Result<usize, std::io::Error>;
    /// Write a block to the store.
    /// Returns an error if the block could not be written.
    fn write_block(&self, block_id: u64, data: &[u8; 4096]) -> Result<(), std::io::Error>;
}
