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

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn push_str(&mut self, s: &str) -> bool {
        let bytes = s.as_bytes();
        if self.len + bytes.len() > CAPACITY {
            return false;
        }
        self.data[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        true
    }

    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) }
    }

    pub fn as_mut_str(&mut self) -> &mut str {
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

impl<const CAPACITY: usize> std::ops::Deref for StaticString<CAPACITY> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl<const CAPACITY: usize> std::ops::DerefMut for StaticString<CAPACITY> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str()
    }
}

pub enum StaticStringConversionError {
    StrTooLong,
}

impl<const CAPACITY: usize> std::str::FromStr for StaticString<CAPACITY> {
    type Err = StaticStringConversionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut result = Self::new();
        if !result.push_str(s) {
            return Err(StaticStringConversionError::StrTooLong);
        }
        Ok(result)
    }
}
