pub trait StaticHash {
    fn static_hash(&self) -> usize;
}

impl StaticHash for u32 {
    fn static_hash(&self) -> usize {
        *self as usize
    }
}

impl StaticHash for i32 {
    fn static_hash(&self) -> usize {
        *self as usize
    }
}

impl StaticHash for &str {
    fn static_hash(&self) -> usize {
        self.as_bytes().iter().map(|&b| b as usize).sum()
    }
}

impl StaticHash for String {
    fn static_hash(&self) -> usize {
        self.as_str().static_hash()
    }
}

#[derive(Clone, Copy)]
pub struct StaticHashMap<K, V, const CAPACITY: usize> {
    data: [Option<(K, V)>; CAPACITY],
}

impl<K: Copy, V: Copy, const CAPACITY: usize> StaticHashMap<K, V, CAPACITY> {
    pub const fn new() -> Self {
        Self {
            data: [None; CAPACITY],
        }
    }
}

impl<K, V, const CAPACITY: usize> StaticHashMap<K, V, CAPACITY> {
    pub fn new_iter() -> Self {
        let data: [Option<(K, V)>; CAPACITY] = std::array::from_fn(|_| None);
        Self { data }
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

impl<K, V, const CAPACITY: usize> StaticHashMap<K, V, CAPACITY>
where
    K: StaticHash + Eq + Clone,
    V: Clone,
{
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = key.static_hash() % CAPACITY;
        let mut existing = None;
        let mut empty = None;
        for i in 0..CAPACITY {
            let idx = (hash + i) % CAPACITY;
            if let Some((k, _)) = &self.data[idx] {
                if k == &key {
                    existing = Some(idx);
                    break;
                }
            } else if empty.is_none() {
                empty = Some(idx);
            }
        }
        if let Some(idx) = existing {
            let old = self.data[idx].as_ref().unwrap().1.clone();
            self.data[idx] = Some((key, value));
            Some(old)
        } else if let Some(idx) = empty {
            self.data[idx] = Some((key, value));
            None
        } else {
            None
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = key.static_hash() % CAPACITY;
        for i in 0..CAPACITY {
            let idx = (hash + i) % CAPACITY;
            if let Some((k, v)) = &self.data[idx] {
                if k == key {
                    return Some(v);
                }
            } else {
                return None;
            }
        }
        None
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = key.static_hash() % CAPACITY;
        let mut found = None;
        for i in 0..CAPACITY {
            let idx = (hash + i) % CAPACITY;
            if let Some((k, _)) = &self.data[idx] {
                if k == key {
                    found = Some(idx);
                    break;
                }
            } else {
                break;
            }
        }
        if let Some(idx) = found {
            if let Some(Some((_, v))) = self.data.get_mut(idx) {
                Some(v)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = key.static_hash() % CAPACITY;
        let mut found = None;
        for i in 0..CAPACITY {
            let idx = (hash + i) % CAPACITY;
            if let Some((k, _)) = &self.data[idx] {
                if k == key {
                    found = Some(idx);
                    break;
                }
            } else {
                break;
            }
        }
        if let Some(idx) = found {
            if let Some((_, v)) = self.data[idx].take() {
                Some(v)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
}

impl<K, V, const CAPACITY: usize> Default for StaticHashMap<K, V, CAPACITY> {
    fn default() -> Self {
        Self::new_iter()
    }
}

impl<K, V, const CAPACITY: usize> std::fmt::Debug for StaticHashMap<K, V, CAPACITY> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticHashMap")
            .field("capacity", &CAPACITY)
            .field("len", &self.len())
            .finish()
    }
}
