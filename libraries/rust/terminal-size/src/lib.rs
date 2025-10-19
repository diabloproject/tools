#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Width(pub u16);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Height(pub u16);

#[cfg(unix)]
mod unix;
#[cfg(unix)]
#[allow(deprecated)]
pub use crate::unix::{terminal_size, terminal_size_of};

#[cfg(windows)]
mod windows;
#[cfg(windows)]
#[allow(deprecated)]
pub use crate::windows::{terminal_size, terminal_size_of};

#[cfg(not(any(unix, windows)))]
pub fn terminal_size() -> Option<(Width, Height)> {
    None
}
