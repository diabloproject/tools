#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct StaticVec<T, const CAPACITY: usize> {
    data: [T; CAPACITY],
    len: usize,
}

impl<T, const CAPACITY: usize> StaticVec<T, CAPACITY> {
    pub const fn new() -> Self {
        unsafe {
            Self {
                data: std::mem::zeroed(),
                len: 0,
            }
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

    pub fn push(&mut self, value: T) -> bool {
        if self.len >= CAPACITY {
            return false;
        }
        self.data[self.len] = value;
        self.len += 1;
        true
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }
}

impl<T, const CAPACITY: usize> Default for StaticVec<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAPACITY: usize> std::fmt::Debug for StaticVec<T, CAPACITY>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticVec")
            .field("data", &self.as_slice())
            .field("len", &self.len)
            .field("capacity", &CAPACITY)
            .finish()
    }
}

impl<T, const CAPACITY: usize> std::ops::Deref for StaticVec<T, CAPACITY> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const CAPACITY: usize> std::ops::DerefMut for StaticVec<T, CAPACITY> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[derive(Debug)]
pub enum StaticVecError {
    TooLong,
}

impl std::fmt::Display for StaticVecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticVecError::TooLong => write!(f, "Input is too long for static vector capacity"),
        }
    }
}

impl std::error::Error for StaticVecError {}

impl<T: Clone, const CAPACITY: usize> TryFrom<&[T]> for StaticVec<T, CAPACITY> {
    type Error = StaticVecError;

    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        if slice.len() > CAPACITY {
            return Err(StaticVecError::TooLong);
        }

        let mut vec = Self::new();
        for item in slice {
            vec.push(item.clone());
        }
        Ok(vec)
    }
}

impl<T: Clone, const CAPACITY: usize> TryFrom<Vec<T>> for StaticVec<T, CAPACITY> {
    type Error = StaticVecError;
    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_from(vec.as_slice())
    }
}

impl<T: Clone, const CAPACITY: usize> From<StaticVec<T, CAPACITY>> for Vec<T> {
    fn from(vec: StaticVec<T, CAPACITY>) -> Self {
        vec.as_slice().to_vec()
    }
}
