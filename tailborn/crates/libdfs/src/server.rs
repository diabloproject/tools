//! The `Server` component acts as a translator between high-level client commands
//! (defined in `DfsClient`) and low-level storage operations. It implements the
//! core storage logic and business rules of the distributed file system, ensuring
//! that client requests are properly transformed into operations compatible with
//! the underlying storage infrastructure.
//!
//! ## Storage model
//! Store manipulates constructs called blocks: each block has a unique identifier and a fixed size.
//! Blocks are used to store data in a distributed file system. But at the high level, dfs exposes an s3-like object storage system.
//! To translate high-level client commands into low-level storage operations, the `Server` will try to create kind of a file system.
//!
//! ## Internal file system
//! Since a single server can manage multiple stores, as well as single file system can be assembled of multiple servers,
//! there are certain rules that we must enforce while using the dfs:
//! 1. Files have "modes": distributed, replicated and localized.
//!   - localized mode will store the data on every server identically.
//!   - replicated mode will ensure that each chunk is stored at at least k nodes, where k is fs' replication factor.
//!   - distributed mode will try to store chunks as evenly as possible, allowing to evenly distribute the network load, as such increasing the chance that compute nodes will never be underloaded.
//! 2. For ease of disposal and isolation, `Server` provides namespaces, and each namespace should have independent file system headers and journals.
//!    There should be no event which results in multiple namespaces pointing to the same block. The only exception is the 0th block, which is used to store metainfo about the current host,
//!    like present namespaces and references to the corresponding blocks.
//! 3. Server can never assume that it owns the namespace. Any operation that modifies namespace content
//!    must first be coordinated with the client, which maintains awareness of all servers participating
//!    in the namespace. This coordination ensures data consistency across distributed nodes and prevents
//!    conflicting operations that could corrupt the namespace structure.
//!
use crate::{client::DfsClient, store::Store};

pub struct Server {
    stores: Vec<Box<dyn Store>>,
}

impl Server {
    pub fn new(stores: Vec<Box<dyn Store>>) -> Self {
        Self { stores }
    }
}
