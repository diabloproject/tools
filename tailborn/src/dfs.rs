#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

/// Errors that can occur when interacting with the Tailborn DFS.
#[derive(Debug, Error)]
pub enum DfsClientError {
    /// The requested object ID does not exist.
    #[error("object does not exist")]
    ObjectDoesNotExist,
    /// Provided file path was invalid.
    #[error("invalid path")]
    InvalidPath,
    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// An opaque identifier for a DFS object.
pub type ObjectID = u64;

/// A client for interacting with the Tailborn distributed file store.
/// Implementors must be Send + Sync to allow parallel usage.
pub trait DfsClient: Send + Sync + 'static {
    /// Get the content of an object by its ID.
    async fn get(&self, id: ObjectID) -> Result<Vec<u8>, DfsClientError>;
    /// Create a new empty object, returning its new ID.
    async fn create(&self) -> Result<ObjectID, DfsClientError>;
    /// Store the content of an existing object.
    async fn store(&self, id: ObjectID, content: Vec<u8>) -> Result<(), DfsClientError>;
    /// Delete an object by its ID.
    async fn delete(&self, id: ObjectID) -> Result<(), DfsClientError>;
}

/// A trivial in-memory DFS client for local testing.
/// Stores objects in an in-memory hash map.
#[derive(Clone)]
pub struct InMemoryDfsClient {
    inner: Arc<Mutex<InnerStore>>,
}

struct InnerStore {
    next_id: ObjectID,
    data: HashMap<ObjectID, Vec<u8>>,
}

impl InMemoryDfsClient {
    /// Construct a new in-memory DFS client.
    pub fn new() -> Self {
        InMemoryDfsClient {
            inner: Arc::new(Mutex::new(InnerStore {
                next_id: 1,
                data: HashMap::new(),
            })),
        }
    }
}

impl DfsClient for InMemoryDfsClient {
    async fn get(&self, id: ObjectID) -> Result<Vec<u8>, DfsClientError> {
        let guard = self.inner.lock().await;
        if let Some(buf) = guard.data.get(&id) {
            Ok(buf.clone())
        } else {
            Err(DfsClientError::ObjectDoesNotExist)
        }
    }

    async fn create(&self) -> Result<ObjectID, DfsClientError> {
        let mut guard = self.inner.lock().await;
        let id = guard.next_id;
        guard.next_id += 1;
        guard.data.insert(id, Vec::new());
        Ok(id)
    }

    async fn store(&self, id: ObjectID, content: Vec<u8>) -> Result<(), DfsClientError> {
        let mut guard = self.inner.lock().await;
        if let Some(entry) = guard.data.get_mut(&id) {
            *entry = content;
            Ok(())
        } else {
            Err(DfsClientError::ObjectDoesNotExist)
        }
    }

    async fn delete(&self, id: ObjectID) -> Result<(), DfsClientError> {
        let mut guard = self.inner.lock().await;
        if guard.data.remove(&id).is_some() {
            Ok(())
        } else {
            Err(DfsClientError::ObjectDoesNotExist)
        }
    }
}