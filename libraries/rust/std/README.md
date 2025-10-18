# static-std

A library providing static (fixed-capacity) versions of standard library types.

## Types

- `StaticString<const CAPACITY: usize>`: A string with fixed capacity.
- `StaticVec<T, const CAPACITY: usize>`: A vector with fixed capacity.
- `StaticHashMap<K, V, const CAPACITY: usize>`: A hash map with fixed capacity.
- `StaticSet<T, const CAPACITY: usize>`: A hash set with fixed capacity.

All types are `Copy` and do not allocate on the heap.