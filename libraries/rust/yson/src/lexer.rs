use std::error::Error;
use std::fmt::Display;
use std::io::{self};
use std::iter::Peekable;
use std::str::FromStr;

use crate::protoshim::{self, ProtoshimError};

#[derive(Debug, Clone, PartialEq)]
pub enum YsonToken {
    // Syntax
    LeftBracket,  // [
    RightBracket, // ]
    LeftBrace,    // {
    RightBrace,   // }
    LeftAngle,    // <
    RightAngle,   // >
    Semicolon,    // ;
    EqualSign,    // =

    // Literals
    SignedInteger(i64),
    UnsignedInteger(u64),
    Float(f64),
    String(Vec<u8>),
    Entity,
    Boolean(bool),
}

#[derive(Debug)]
pub enum YsonLexError<E: Error> {
    Io(io::Error),
    UnexpectedByte(u8),
    UnexpectedEof,
    IteratorError(E),
    ProtoshimError(ProtoshimError<E>),
}

impl<E: Error> Display for YsonLexError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            YsonLexError::Io(err) => write!(f, "IO error: {}", err),
            YsonLexError::UnexpectedByte(byte) => write!(f, "Unexpected byte: {}", byte),
            YsonLexError::UnexpectedEof => write!(f, "Unexpected end of file"),
            YsonLexError::IteratorError(err) => write!(f, "Iterator error: {}", err),
            YsonLexError::ProtoshimError(protoshim_error) => {
                write!(f, "Protoshim error: {}", protoshim_error)
            }
        }
    }
}

impl<E: Error> std::error::Error for YsonLexError<E> {}

impl<E: Error> From<ProtoshimError<E>> for YsonLexError<E> {
    fn from(err: ProtoshimError<E>) -> Self {
        match err {
            ProtoshimError::IteratorError(err) => YsonLexError::IteratorError(err),
            err => YsonLexError::ProtoshimError(err),
        }
    }
}

impl<E: Error> From<E> for YsonLexError<E> {
    fn from(err: E) -> Self {
        YsonLexError::IteratorError(err)
    }
}

/// Streaming lexer over a byte source
pub struct YsonLexer<E: Error, R: Iterator<Item = Result<u8, E>>> {
    reader: Peekable<R>,
}

impl<E: Error, R: Iterator<Item = Result<u8, E>>> YsonLexer<E, R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: reader.peekable(),
        }
    }

    fn next_token(&mut self) -> Result<Option<YsonToken>, YsonLexError<E>> {
        while let Some(res) = self.reader.peek() {
            if let Ok(byte) = res {
                if byte.is_ascii_whitespace() {
                    self.reader.next();
                } else {
                    break;
                }
            } else {
                let res = self.reader.next().unwrap();
                return Err(YsonLexError::IteratorError(res.unwrap_err()));
            }
        }
        match self.reader.next().ok_or(YsonLexError::UnexpectedEof)?? {
            0x01 => {
                let buffer_length = protoshim::decode_sint64(&mut self.reader)? as usize;
                let mut buffer = Vec::with_capacity(buffer_length);
                for _ in 0..buffer_length {
                    buffer.push(self.reader.next().ok_or(YsonLexError::UnexpectedEof)??);
                }
                Ok(Some(YsonToken::String(buffer)))
            }
            0x02 => {
                let value = protoshim::decode_sint64(&mut self.reader)?;
                Ok(Some(YsonToken::SignedInteger(value)))
            }
            0x03 => {
                let value = protoshim::decode_double(&mut self.reader)?;
                Ok(Some(YsonToken::Float(value)))
            }
            0x04 => Ok(Some(YsonToken::Boolean(true))),
            0x05 => Ok(Some(YsonToken::Boolean(false))),
            0x06 => {
                let value = protoshim::decode_uint64(&mut self.reader)?;
                Ok(Some(YsonToken::UnsignedInteger(value)))
            }
            b'"' => {
                let mut in_escape = false;
                let mut buffer = Vec::new();
                loop {
                    match self
                        .reader
                        .next()
                        .ok_or_else(|| YsonLexError::UnexpectedEof)??
                    {
                        b'"' if !in_escape => break,
                        b'\\' => in_escape = !in_escape,
                        byte => {
                            if in_escape {
                                assert!(
                                    byte == b'"',
                                    "Only \" is allowed in escape sequences for now"
                                );
                            }
                            in_escape = false;
                            buffer.push(byte);
                        }
                    }
                }
                Ok(Some(YsonToken::String(Vec::new())))
            }
            byte if byte.is_ascii_alphabetic() || byte == b'_' => {
                let mut buffer = Vec::new();
                buffer.push(byte);
                loop {
                    let byte = self
                        .reader
                        .next()
                        .ok_or_else(|| YsonLexError::UnexpectedEof)??;
                    match byte {
                        b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => buffer.push(byte),
                        _ => break,
                    }
                }
                Ok(Some(YsonToken::String(buffer)))
            }
            byte if byte.is_ascii_digit() => {
                fn parse<T: FromStr, E: Error>(val: &[u8]) -> Result<T, YsonLexError<E>> {
                    unsafe { std::str::from_utf8_unchecked(val) }
                        .parse()
                        .map_err(|_| todo!("proper error"))
                }

                let mut buffer = Vec::new();
                buffer.push(byte);
                loop {
                    let res = self.reader.peek();
                    if res.is_none() {
                        if buffer.contains(&b'.') {
                            return Ok(Some(YsonToken::Float(parse(&buffer)?)));
                        } else {
                            return Ok(Some(YsonToken::SignedInteger(parse(&buffer)?)));
                        }
                    }
                    let byte = match res.unwrap() {
                        Ok(byte) => *byte,
                        Err(_) => {
                            return Err(YsonLexError::IteratorError(
                                self.reader.next().unwrap().err().unwrap(),
                            ));
                        }
                    };
                    match byte {
                        b'0'..=b'9' => {
                            buffer.push(byte);
                            self.reader.next();
                        }
                        b'.' if !buffer.contains(&b'.') => {
                            buffer.push(byte);
                            self.reader.next();
                        }
                        _ => break,
                    }
                }

                if buffer.contains(&b'.') {
                    Ok(Some(YsonToken::Float(parse(&buffer)?)))
                } else if buffer.ends_with(b"u") {
                    Ok(Some(YsonToken::UnsignedInteger(parse(
                        &buffer[..buffer.len() - 1],
                    )?)))
                } else {
                    Ok(Some(YsonToken::SignedInteger(parse(&buffer)?)))
                }
            }
            b'%' => match self.reader.next().ok_or(YsonLexError::UnexpectedEof)?? {
                b't' => {
                    let rest = "rue".to_string().as_bytes().to_vec();
                    for byte in rest {
                        if byte != self.reader.next().ok_or(YsonLexError::UnexpectedEof)?? {
                            return Err(YsonLexError::UnexpectedByte(byte));
                        }
                    }
                    Ok(Some(YsonToken::Boolean(true)))
                }
                b'f' => {
                    let rest = "alse".to_string().as_bytes().to_vec();
                    for byte in rest {
                        if byte != self.reader.next().ok_or(YsonLexError::UnexpectedEof)?? {
                            return Err(YsonLexError::UnexpectedByte(byte));
                        }
                    }
                    Ok(Some(YsonToken::Boolean(false)))
                }
                byte => Err(YsonLexError::UnexpectedByte(byte)),
            },
            b'#' => Ok(Some(YsonToken::Entity)),
            b'<' => Ok(Some(YsonToken::LeftAngle)),
            b'>' => Ok(Some(YsonToken::RightAngle)),
            b'{' => Ok(Some(YsonToken::LeftBrace)),
            b'}' => Ok(Some(YsonToken::RightBrace)),
            b'[' => Ok(Some(YsonToken::LeftBracket)),
            b']' => Ok(Some(YsonToken::RightBracket)),
            b'=' => Ok(Some(YsonToken::EqualSign)),
            b';' => Ok(Some(YsonToken::Semicolon)),
            b' ' | b'\t' | b'\n' | b'\r' => self.next_token(),
            _ => todo!("Proper error"),
        }
    }
}

impl<E: Error, R: Iterator<Item = Result<u8, E>>> Iterator for YsonLexer<E, R> {
    type Item = Result<YsonToken, YsonLexError<E>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Ok(Some(tok)) => Some(Ok(tok)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn lex_bytes(input: &[u8]) -> Result<Vec<YsonToken>, YsonLexError<std::convert::Infallible>> {
        let iter = input.iter().map(|&b| Ok(b));
        let lexer = YsonLexer::new(iter);
        let mut tokens = Vec::new();
        for token in lexer {
            tokens.push(token?);
        }
        Ok(tokens)
    }

    #[test]
    fn test_syntax_tokens() {
        let input = b"[]{}<>();=";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                YsonToken::LeftBracket,
                YsonToken::RightBracket,
                YsonToken::LeftBrace,
                YsonToken::RightBrace,
                YsonToken::LeftAngle,
                YsonToken::RightAngle,
                YsonToken::Semicolon,
                YsonToken::EqualSign,
            ]
        );
    }

    #[test]
    fn test_entity() {
        let input = b"#";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(tokens, vec![YsonToken::Entity]);
    }

    #[test]
    fn test_boolean_literals() {
        let input = b"%true %false";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![YsonToken::Boolean(true), YsonToken::Boolean(false),]
        );
    }

    #[test]
    fn test_quoted_strings() {
        let input = b"\"hello\" \"world\"";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![YsonToken::String(Vec::new()), YsonToken::String(Vec::new()),]
        );
    }

    #[test]
    fn test_quoted_string_with_escape() {
        let input = b"\"hello \\\"world\\\"\"";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(tokens, vec![YsonToken::String(Vec::new())]);
    }

    #[test]
    fn test_identifier_strings() {
        let input = b"hello _world test123 _123abc";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                YsonToken::String(b"hello".to_vec()),
                YsonToken::String(b"_world".to_vec()),
                YsonToken::String(b"test123".to_vec()),
                YsonToken::String(b"_123abc".to_vec()),
            ]
        );
    }

    #[test]
    fn test_signed_integers() {
        let input = b"123 0 999";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                YsonToken::SignedInteger(123),
                YsonToken::SignedInteger(0),
                YsonToken::SignedInteger(999),
            ]
        );
    }

    #[test]
    fn test_unsigned_integers() {
        let input = b"123u 0u";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                YsonToken::UnsignedInteger(123),
                YsonToken::UnsignedInteger(0),
            ]
        );
    }

    #[test]
    fn test_floats() {
        let input = b"123.45 0.0 999.999";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                YsonToken::Float(123.45),
                YsonToken::Float(0.0),
                YsonToken::Float(999.999),
            ]
        );
    }

    #[test]
    fn test_whitespace_handling() {
        let input = b"  \t\n\r123  \t\n\r  456\n";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![YsonToken::SignedInteger(123), YsonToken::SignedInteger(456),]
        );
    }

    #[test]
    fn test_mixed_tokens() {
        let input = b"[hello = 123, world = %true]";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                YsonToken::LeftBracket,
                YsonToken::String(b"hello".to_vec()),
                YsonToken::EqualSign,
                YsonToken::SignedInteger(123),
                YsonToken::String(b",".to_vec()),
                YsonToken::String(b"world".to_vec()),
                YsonToken::EqualSign,
                YsonToken::Boolean(true),
                YsonToken::RightBracket,
            ]
        );
    }

    #[test]
    fn test_binary_encoded_tokens() {
        // Test binary encoded string (0x01 followed by length and data)
        let mut input = vec![0x01, 0x05]; // String with length 5
        input.extend_from_slice(b"hello");
        let tokens = lex_bytes(&input).unwrap();
        assert_eq!(tokens, vec![YsonToken::String(b"hello".to_vec())]);
    }

    #[test]
    fn test_binary_boolean_tokens() {
        let input = &[0x04, 0x05]; // true, false
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(
            tokens,
            vec![YsonToken::Boolean(true), YsonToken::Boolean(false),]
        );
    }

    #[test]
    fn test_empty_input() {
        let input = b"";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(tokens, vec![]);
    }

    #[test]
    fn test_only_whitespace() {
        let input = b"   \t\n\r   ";
        let tokens = lex_bytes(input).unwrap();
        assert_eq!(tokens, vec![]);
    }
}
