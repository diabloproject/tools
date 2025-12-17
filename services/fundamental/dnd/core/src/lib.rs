const BLOCK_SIZE: usize = 4096;


struct Block {
    data: [u8; BLOCK_SIZE]
}


impl Block {
    fn new() -> Self {
        Block {
            data: [0; BLOCK_SIZE]
        }
    }
}


struct DeferredBlockArray {
    data_fd: (),
}

impl DeferredBlockArray {
    fn new() -> Self {
        DeferredBlockArray {}
    }

    fn with<'a, R>(&self, idx: usize, f: &'a dyn FnOnce(&'a Block) -> R) -> R {
        unimplemented!()
    }
}


struct Engine {
    blocks: DeferredBlockArray
}