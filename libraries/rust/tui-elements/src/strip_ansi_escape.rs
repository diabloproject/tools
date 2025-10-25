use std::io::{self, IntoInnerError, LineWriter, Write};
use vte::{Parser, Perform};

pub struct Writer<W>
where
    W: Write,
{
    performer: Performer<W>,
    parser: Parser,
}

/// Strip ANSI escapes from `data` and return the remaining bytes as a `Vec<u8>`.
///
/// See [the module documentation][mod] for an example.
///
/// [mod]: index.html
pub fn strip<T>(data: T) -> Vec<u8>
where
    T: AsRef<[u8]>,
{
    fn strip_impl(data: &[u8]) -> io::Result<Vec<u8>> {
        let mut writer = Writer::new(Vec::new());
        writer.write_all(data)?;
        Ok(writer.into_inner()?)
    }

    strip_impl(data.as_ref()).expect("writing to a Vec<u8> cannot fail")
}

/// Strip ANSI escapes from `data` and return the remaining contents as a `String`.
///
/// # Example
///
/// ```
/// let str_with_colors = "\x1b[32mfoo\x1b[m bar";
/// let string_without_colors = strip_ansi_escapes::strip_str(str_with_colors);
/// assert_eq!(string_without_colors, "foo bar");
/// ```
pub fn strip_str<T>(data: T) -> String
where
    T: AsRef<str>,
{
    let bytes = strip(data.as_ref());
    String::from_utf8(bytes)
        .expect("stripping ANSI escapes from a UTF-8 string always results in UTF-8")
}

struct Performer<W>
where
    W: Write,
{
    writer: LineWriter<W>,
    err: Option<io::Error>,
}

impl<W> Writer<W>
where
    W: Write,
{
    /// Create a new `Writer` that writes to `inner`.
    pub fn new(inner: W) -> Writer<W> {
        Writer {
            performer: Performer {
                writer: LineWriter::new(inner),
                err: None,
            },
            parser: Parser::new(),
        }
    }

    /// Unwraps this `Writer`, returning the underlying writer.
    ///
    /// The internal buffer is written out before returning the writer, which
    /// may produce an [`IntoInnerError`].
    ///
    /// [IntoInnerError]: https://doc.rust-lang.org/std/io/struct.IntoInnerError.html
    pub fn into_inner(self) -> Result<W, IntoInnerError<LineWriter<W>>> {
        self.performer.into_inner()
    }
}

impl<W> Write for Writer<W>
where
    W: Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.parser.advance(&mut self.performer, buf);
        match self.performer.err.take() {
            Some(e) => Err(e),
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.performer.flush()
    }
}

impl<W> Performer<W>
where
    W: Write,
{
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn into_inner(self) -> Result<W, IntoInnerError<LineWriter<W>>> {
        self.writer.into_inner()
    }
}

impl<W> Perform for Performer<W>
where
    W: Write,
{
    fn print(&mut self, c: char) {
        // Just print bytes to the inner writer.
        self.err = write!(self.writer, "{}", c).err();
    }
    fn execute(&mut self, byte: u8) {
        // We only care about executing linefeeds.
        if byte == b'\n' {
            self.err = writeln!(self.writer).err();
        }
    }
}

#[cfg(doctest)]
extern crate doc_comment;

#[cfg(doctest)]
doc_comment::doctest!("../README.md", readme);

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_parsed(input: &[u8], expected: &[u8]) {
        let bytes = strip(input);
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_simple() {
        assert_parsed(b"\x1b[m\x1b[m\x1b[32m\x1b[1m    Finished\x1b[m dev [unoptimized + debuginfo] target(s) in 0.0 secs",
                      b"    Finished dev [unoptimized + debuginfo] target(s) in 0.0 secs");
    }

    #[test]
    fn test_newlines() {
        assert_parsed(b"foo\nbar\n", b"foo\nbar\n");
    }

    #[test]
    fn test_escapes_newlines() {
        assert_parsed(b"\x1b[m\x1b[m\x1b[32m\x1b[1m   Compiling\x1b[m utf8parse v0.1.0
\x1b[m\x1b[m\x1b[32m\x1b[1m   Compiling\x1b[m vte v0.3.2
\x1b[m\x1b[m\x1b[32m\x1b[1m   Compiling\x1b[m strip-ansi-escapes v0.1.0-pre (file:///build/strip-ansi-escapes)
\x1b[m\x1b[m\x1b[32m\x1b[1m    Finished\x1b[m dev [unoptimized + debuginfo] target(s) in 0.66 secs
",
                      b"   Compiling utf8parse v0.1.0
   Compiling vte v0.3.2
   Compiling strip-ansi-escapes v0.1.0-pre (file:///build/strip-ansi-escapes)
    Finished dev [unoptimized + debuginfo] target(s) in 0.66 secs
");
    }
}