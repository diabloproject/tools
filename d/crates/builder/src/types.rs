use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Copy, Clone)]
pub(crate) struct StaticString<const N: usize>(usize, [u8; N]);

impl<const N: usize> StaticString<N> {
    pub unsafe fn new_raw_unchecked(data: [u8; N], length: usize) -> Self {
        Self(length, data)
    }

    pub fn new() -> Self {
        Self(0, [0; N])
    }
}

impl<const N: usize> Debug for StaticString<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // SAFETY: Unless person used unchecked input, it is just rust internal string rep, so...
        let check = unsafe { std::str::from_utf8_unchecked(&self.1[..self.0]) };
        write!(f, "ss<{N}>{check:?}")
    }
}

impl<const N: usize> PartialEq for StaticString<N> {
    fn eq(&self, other: &Self) -> bool {
        // We are not interested in garbage
        &self.1[..self.0] == &other.1[..other.0]
    }
}

impl<const N: usize> Eq for StaticString<N> {}
impl<const N: usize> PartialOrd for StaticString<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let left = unsafe { std::str::from_utf8_unchecked(&self.1[..self.0]) };
        let right = unsafe { std::str::from_utf8_unchecked(&other.1[..other.0]) };
        left.partial_cmp(right)
    }
}
impl<const N: usize> Ord for StaticString<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        let left = unsafe { std::str::from_utf8_unchecked(&self.1[..self.0]) };
        let right = unsafe { std::str::from_utf8_unchecked(&other.1[..other.0]) };
        left.cmp(right)
    }
}

impl<const N: usize> Hash for StaticString<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let vale = unsafe { std::str::from_utf8_unchecked(&self.1[..self.0]) };
        vale.hash(state);
    }
}

#[derive(Error, Debug)]
pub(crate) enum StaticError {
    #[error("Sequence too long")]
    TooLong,
}

impl<const N: usize> TryFrom<&str> for StaticString<N> {
    type Error = StaticError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.as_bytes().len() > N {
            return Err(StaticError::TooLong);
        }
        let mut bytes = [0; N];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        Ok(Self(value.len(), bytes))
    }
}

#[derive(Copy, Clone)]
pub(crate) struct StaticVec<const N: usize, T: Copy + Clone>(usize, [T; N]);
impl<const N: usize, T: Copy + Clone + Debug> Debug for StaticVec<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self)
            .finish()
    }
}

impl<const N: usize, T: Copy + Clone> TryFrom<&[T]> for StaticVec<N, T> {
    type Error = StaticError;

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        if value.len() > N {
            Err(StaticError::TooLong)
        } else {
            let mut arr = [unsafe { std::mem::zeroed() }; N];
            arr[..value.len()].copy_from_slice(value);
            Ok(Self(value.len(), arr))
        }
    }
}

impl<const N: usize, T: Copy + Clone> Index<usize> for StaticVec<N, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.1[index]
    }
}

pub(crate) struct StaticVecIter<'a, const N: usize, T: Copy + Clone> {
    contents: &'a StaticVec<N, T>,
    index: usize,
}

impl<'a, const N: usize, T: Copy + Clone> IntoIterator for &'a StaticVec<N, T> {
    type Item = &'a T;
    type IntoIter = StaticVecIter<'a, N, T>;

    fn into_iter(self) -> Self::IntoIter {
        StaticVecIter {
            contents: self,
            index: 0,
        }
    }
}

impl<'a, const N: usize, T: Copy + Clone> Iterator for StaticVecIter<'a, N, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        if self.contents.0 < self.index {
            None
        } else {
            Some(&self.contents.1[self.index - 1])
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Spec {
    pub(crate) name: StaticString<64>,
    pub(crate) artifacts: StaticVec<16, Artifact>,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Artifact {
    pub(crate) name: StaticString<64>,
    pub(crate) r#type: ArtifactType,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum ArtifactType {
    Executable,
    Library,
    Resource,
}


#[derive(Clone)]
pub(crate) struct  Recipe {
    pub(crate) name: StaticString<64>,
    pub(crate) function: rust_lisp::model::Lambda
}

impl Debug for Recipe {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        struct _fake;
        impl Debug for _fake {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                f.write_str("<lisp_function>")
            }
        }
        f.debug_struct("Recipe")
            .field("name", &self.name)
            .field("function", &_fake)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ArtifactPath {
    pub(crate) path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ArtifactInstance {
    pub(crate) path: PathBuf,
    pub(crate) artifact: Artifact,
}
