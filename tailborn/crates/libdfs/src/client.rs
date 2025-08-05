use thiserror::Error;

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

pub type DfsFuture<T> = Box<dyn Future<Output = Result<T, DfsClientError>> + Send>;

/// A client for interacting with the Tailborn distributed file store.
/// Implementors must be Send + Sync to allow parallel usage.
pub trait DfsClient: Send + Sync + 'static {
    /// Get the content of an object by its ID.
    fn get(&self, id: ObjectID) -> DfsFuture<Vec<u8>>;
    /// Create a new empty object, returning its new ID.
    fn create(&self) -> DfsFuture<ObjectID>;
    /// Store the content of an existing object.
    fn store(&self, id: ObjectID, content: Vec<u8>) -> DfsFuture<()>;
    /// Delete an object by its ID.
    fn delete(&self, id: ObjectID) -> DfsFuture<()>;
}
