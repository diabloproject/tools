use std::hash::Hash;

#[derive(Clone, Copy)]
pub struct StaticString<const CAPACITY: usize> {
    data: [u8; CAPACITY],
    len: usize,
}

impl<const CAPACITY: usize> StaticString<CAPACITY> {
    pub const fn new() -> Self {
        Self {
            data: [0; CAPACITY],
            len: 0,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    pub const fn clear(&mut self) {
        self.len = 0;
    }

    pub const fn push_str(&mut self, s: &str) -> bool {
        let bytes = s.as_bytes();
        if self.len + bytes.len() > CAPACITY {
            return false;
        }
        self.data[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        true
    }

    pub const fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) }
    }

    pub const fn as_mut_str(&mut self) -> &mut str {
        unsafe { std::str::from_utf8_unchecked_mut(&mut self.data[..self.len]) }
    }
}

impl<const CAPACITY: usize> Default for StaticString<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> std::fmt::Debug for StaticString<CAPACITY> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticString")
            .field("data", &&self.data[..self.len])
            .field("len", &self.len)
            .field("capacity", &CAPACITY)
            .finish()
    }
}

impl<const CAPACITY: usize> std::fmt::Display for StaticString<CAPACITY> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl<const CAPACITY: usize> const std::ops::Deref for StaticString<CAPACITY> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl<const CAPACITY: usize> const std::ops::DerefMut for StaticString<CAPACITY> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str()
    }
}

#[derive(Debug)]
pub enum StaticStringConversionError {
    StrTooLong,
}

impl std::fmt::Display for StaticStringConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticStringConversionError::StrTooLong => {
                write!(f, "String is too long to fit in static string")
            }
        }
    }
}

impl std::error::Error for StaticStringConversionError {}

impl<const CAPACITY: usize> const std::str::FromStr for StaticString<CAPACITY> {
    type Err = StaticStringConversionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut result = Self::new();
        if !result.push_str(s) {
            return Err(StaticStringConversionError::StrTooLong);
        }
        Ok(result)
    }
}

impl<const CAPACITY: usize> Hash for StaticString<CAPACITY> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl<const CAPACITY: usize> const PartialEq for StaticString<CAPACITY> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<const CAPACITY: usize> const Eq for StaticString<CAPACITY> {}
