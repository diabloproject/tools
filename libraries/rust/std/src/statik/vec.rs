//! # Static vector
//! This is an implementation of the static vector — vector that can be:
//! 1. Allocated on the stack
//! 2. Used in const expressions
//!
//! # Why `Copy` requirement?
//! Unlike its runtime counterpart, StaticVec requires the type
//! you are going to store in it to implement Copy. There are
//! a couple of reasons for that, namely:
//! 1. `::new()` becomes un-const, since constructing `[T]` is un-const.
//! 2. I am too lazy to implement the vector in way that allows `Drop` to happen =)

use std::hash::{Hash, Hasher};
#[derive(Clone, Copy)]
pub struct StaticVec<T: Copy, const CAPACITY: usize> {
    data: [T; CAPACITY],
    len: usize,
}

impl<T: Copy + PartialEq, const CAPACITY: usize> PartialEq for StaticVec<T, CAPACITY> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        for i in 0..self.len {
            if self.data[i] != other.data[i] {
                return false;
            }
        }
        true
    }
}

impl<T: Copy + Eq, const CAPACITY: usize> Eq for StaticVec<T, CAPACITY> {}

impl<T: Copy + Hash, const CAPACITY: usize> Hash for StaticVec<T, CAPACITY> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for item in self.as_slice() {
            item.hash(state);
        }
    }
}

impl<T: Copy, const CAPACITY: usize> StaticVec<T, CAPACITY> {
    pub const fn new() -> Self {
        unsafe {
            Self {
                data: std::mem::zeroed(),
                len: 0,
            }
        }
    }

    pub const fn from_const(data: &[T]) -> Self {
        let mut mem: [T; CAPACITY] = unsafe { std::mem::zeroed() };
        // Copy memory from the const slice to the mutable slice
        let view =
            unsafe { std::slice::from_raw_parts_mut(mem.as_mut_ptr(), data.len().min(CAPACITY)) };
        view.copy_from_slice(data);
        Self {
            data: mem,
            len: data.len().min(CAPACITY),
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

    pub const fn push(&mut self, value: T) -> bool {
        if self.len >= CAPACITY {
            return false;
        }
        self.data[self.len] = value;
        self.len += 1;
        true
    }

    pub const fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }
}

impl<T: Copy, const CAPACITY: usize> Default for StaticVec<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const CAPACITY: usize> std::fmt::Debug for StaticVec<T, CAPACITY>
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

impl<T: Copy, const CAPACITY: usize> const std::ops::Deref for StaticVec<T, CAPACITY> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy, const CAPACITY: usize> const std::ops::DerefMut for StaticVec<T, CAPACITY> {
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

impl<T: Copy, const CAPACITY: usize> const TryFrom<&[T]> for StaticVec<T, CAPACITY> {
    type Error = StaticVecError;

    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        if slice.len() > CAPACITY {
            return Err(StaticVecError::TooLong);
        }

        let mut vec = Self::new();
        let mut idx = 0;
        // For loop is not const, so we have to use a while loop
        while idx < slice.len() {
            vec.push(slice[idx]);
            idx += 1;
        }
        Ok(vec)
    }
}

impl<T: Copy, const CAPACITY: usize> TryFrom<Vec<T>> for StaticVec<T, CAPACITY> {
    type Error = StaticVecError;
    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_from(vec.as_slice())
    }
}

impl<T: Copy, const CAPACITY: usize> From<StaticVec<T, CAPACITY>> for Vec<T> {
    fn from(vec: StaticVec<T, CAPACITY>) -> Self {
        vec.as_slice().to_vec()
    }
}

#[macro_export]
macro_rules! static_vec {
    [$( $arg: expr ), *] => {
        const {
            let mut vec = StaticVec::<_, _>::new();
            $(
                vec.push($arg);
            )*
            vec
        }
    };
}
