pub struct RowIterator<'src> {
    source: &'src [u8],
    delimiter: u8,
    pos: usize,
}

impl<'src> RowIterator<'src> {
    pub fn new(source: &'src [u8], delimiter: u8) -> Self {
        RowIterator {
            delimiter,
            source,
            pos: 0,
        }
    }
}

impl<'src> Iterator for RowIterator<'src> {
    type Item = Vec<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut row = Vec::new();
        let mut in_literal = false;
        let mut start = self.pos;
        let mut escaped = false;
        let mut last_cell = Vec::new();

        while self.pos < self.source.len() {
            let c = self.source[self.pos];

            if escaped {
                self.pos += 1;
                escaped = false;
                last_cell.push(c);
                continue;
            } else if c == b'\\' {
                escaped = true;
                self.pos += 1;
            } else if c == b'"' {
                in_literal = !in_literal;
                self.pos += 1;
            } else if in_literal {
                last_cell.push(c);
                self.pos += 1;
            } else if c == self.delimiter {
                self.pos += 1;
                start = self.pos;
                row.push(last_cell);
                last_cell = Vec::new();
            } else if c == b'\n' {
                row.push(last_cell);
                let val = Some(row);
                self.pos += 1;
                return val;
            } else {
                last_cell.push(c);
                self.pos += 1;
            }
        }

        if !row.is_empty() { Some(row) } else { None }
    }
}

pub struct RowSerializer {
    delimiter: u8,
}

impl<'src> RowSerializer {
    pub fn new(delimiter: u8) -> Self {
        RowSerializer { delimiter }
    }

    pub fn serialize(&self, rows: Vec<Vec<&'src [u8]>>) -> Vec<u8> {
        let mut result = Vec::new();
        for row in rows.iter() {
            for (i, cell) in row.iter().enumerate() {
                if i > 0 {
                    result.push(self.delimiter);
                }
                if cell.contains(&self.delimiter) || cell.contains(&b'"') {
                    result.push(b'"');
                    result.extend_from_slice(cell);
                    result.push(b'"');
                } else {
                    result.extend_from_slice(cell);
                }
            }
            result.push(b'\n');
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_serializer_basic() {
        let serializer = RowSerializer::new(b',');
        let rows: Vec<Vec<&[u8]>> = vec![vec![b"a", b"b", b"c"], vec![b"d", b"e", b"f"]];

        assert_eq!(serializer.serialize(rows), b"a,b,c\nd,e,f\n");
    }

    #[test]
    fn test_row_serializer_with_delimiter() {
        let serializer = RowSerializer::new(b',');
        let rows: Vec<Vec<&[u8]>> = vec![vec![b"a,b", b"c"], vec![b"x,y", b"z"]];

        assert_eq!(serializer.serialize(rows), b"\"a,b\",c\n\"x,y\",z\n");
    }

    #[test]
    fn test_row_iterator_basic() {
        let source = b"a,b,c\nd,e,f\n";
        let mut iter = RowIterator::new(source, b',');

        assert_eq!(iter.next(), Some(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]));
        assert_eq!(iter.next(), Some(vec![b"d".to_vec(), b"e".to_vec(), b"f".to_vec()]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_row_iterator_literal() {
        let source = b"\"a,b\",c\n\"x,y\",z\n";
        let mut iter = RowIterator::new(source, b',');

        assert_eq!(iter.next(), Some(vec![b"a,b".to_vec(), b"c".to_vec()]));
        assert_eq!(iter.next(), Some(vec![b"x,y".to_vec(), b"z".to_vec()]));
        assert_eq!(iter.next(), None);
    }
    
    #[test]
    fn test_escaping() {
        let source = b"a\\,b,c\n\"x\\\"y\",z\n";
        let mut iter = RowIterator::new(source, b',');

        assert_eq!(iter.next(), Some(vec![b"a,b".to_vec(), b"c".to_vec()]));
        assert_eq!(iter.next(), Some(vec![b"x\"y".to_vec(), b"z".to_vec()]));
        assert_eq!(iter.next(), None);
    }
}