use super::hashmap::StaticHash;

#[derive(Clone, Copy)]
pub struct StaticSet<T, const CAPACITY: usize> {
    data: [Option<T>; CAPACITY],
}

impl<T, const CAPACITY: usize> StaticSet<T, CAPACITY> {
    pub const fn new() -> Self {
        unsafe {
            Self {
                data: std::mem::zeroed(),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.data.iter().filter(|x| x.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    pub fn clear(&mut self) {
        for item in &mut self.data {
            *item = None;
        }
    }
}

impl<T, const CAPACITY: usize> StaticSet<T, CAPACITY>
where
    T: StaticHash + Eq + Clone,
{
    pub fn insert(&mut self, value: T) -> bool {
        if self.contains(&value) {
            return false;
        }
        let hash = value.static_hash() % CAPACITY;
        for i in 0..CAPACITY {
            let idx = (hash + i) % CAPACITY;
            if self.data[idx].is_none() {
                self.data[idx] = Some(value);
                return true;
            }
        }
        false // full
    }

    pub fn contains(&self, value: &T) -> bool {
        let hash = value.static_hash() % CAPACITY;
        for i in 0..CAPACITY {
            let idx = (hash + i) % CAPACITY;
            match &self.data[idx] {
                Some(v) if v == value => return true,
                None => return false,
                _ => {}
            }
        }
        false
    }

    pub fn remove(&mut self, value: &T) -> bool {
        let hash = value.static_hash() % CAPACITY;
        for i in 0..CAPACITY {
            let idx = (hash + i) % CAPACITY;
            if let Some(v) = &self.data[idx] {
                if v == value {
                    self.data[idx] = None;
                    return true;
                }
            } else {
                return false;
            }
        }
        false
    }
}

impl<T, const CAPACITY: usize> Default for StaticSet<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAPACITY: usize> std::fmt::Debug for StaticSet<T, CAPACITY>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set()
            .entries(self.data.iter().filter_map(|x| x.as_ref()))
            .finish()
    }
}
