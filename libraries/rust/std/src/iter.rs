pub struct WithLastMarker<T: Iterator + Sized + ExactSizeIterator> {
    iterator: T,
}

impl<T: Iterator + Sized + ExactSizeIterator> Iterator for WithLastMarker<T> {
    type Item = (T::Item, bool);

    fn next(&mut self) -> Option<Self::Item> {
        let is_last = self.iterator.len() <= 1;
        Some((self.iterator.next()?, is_last))
    }
}

pub trait LastValueIteratorExt: Iterator + Sized + ExactSizeIterator {
    fn with_last_marker(self) -> WithLastMarker<Self> {
        WithLastMarker { iterator: self }
    }
}

impl<T: ExactSizeIterator + Sized> LastValueIteratorExt for T {}
