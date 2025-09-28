use std::{error::Error, iter::Peekable};

use crate::lexer::YsonToken;

pub type YsonString = Vec<u8>;

#[derive(Debug, Clone, PartialEq)]
pub enum YsonValue {
    Array(Vec<YsonNode>),
    Map(Vec<(YsonString, YsonNode)>),
    Entity,
    Boolean(bool),
    SignedInteger(i64),
    UnsignedInteger(u64),
    Double(f64),
    String(YsonString),
}

#[derive(Debug, Clone, PartialEq)]
pub struct YsonNode {
    pub value: YsonValue,
    pub attributes: Vec<(YsonString, YsonNode)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum YsonParseError<E: Error> {
    UnexpectedToken(YsonToken),
    UnexpectedEndOfInput,
    IteratorError(E),
}

impl<E: Error> std::fmt::Display for YsonParseError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            YsonParseError::UnexpectedToken(token) => write!(f, "Unexpected token: {:?}", token),
            YsonParseError::UnexpectedEndOfInput => write!(f, "Unexpected end of input"),
            YsonParseError::IteratorError(err) => write!(f, "Iterator error: {}", err),
        }
    }
}

impl<E: Error> Error for YsonParseError<E> {}

impl<E: Error> From<E> for YsonParseError<E> {
    fn from(err: E) -> Self {
        YsonParseError::IteratorError(err)
    }
}

macro_rules! eof_guard {
    ($item: expr) => {
        $item.ok_or(YsonParseError::UnexpectedEndOfInput)?
    };
}

pub struct YsonParser<E: Error, R: Iterator<Item = Result<YsonToken, E>>> {
    lexer: Peekable<R>,
}

impl<E: Error, R: Iterator<Item = Result<YsonToken, E>>> YsonParser<E, R> {
    pub fn new(lexer: R) -> Self {
        YsonParser {
            lexer: lexer.peekable(),
        }
    }

    pub fn parse_complete(&mut self) -> Result<YsonNode, YsonParseError<E>> {
        let attributes: Option<Vec<(YsonString, YsonNode)>> = self.parse_attributes()?;
        let token = eof_guard!(self.lexer.peek());
        let token = match token {
            Ok(v) => v,
            Err(_) => &self.lexer.next().unwrap()?,
        };
        Ok(YsonNode {
            value: match token {
                YsonToken::LeftBracket => self.parse_array()?,
                YsonToken::LeftBrace => self.parse_map()?,
                YsonToken::Entity => self.parse_entity()?,
                YsonToken::Boolean(_) => self.parse_boolean()?,
                YsonToken::SignedInteger(_) => self.parse_i64()?,
                YsonToken::UnsignedInteger(_) => self.parse_u64()?,
                YsonToken::Double(_) => self.parse_f64()?,
                YsonToken::String(_) => self.parse_string()?,
                _ => {
                    return Err(YsonParseError::UnexpectedToken(eof_guard!(
                        self.lexer.next()
                    )?));
                }
            },
            attributes: attributes.unwrap_or_default(),
        })
    }

    fn parse_attributes(
        &mut self,
    ) -> Result<Option<Vec<(YsonString, YsonNode)>>, YsonParseError<E>> {
        let tok = self.lexer.peek();
        let tok = match tok {
            Some(v) => v,
            None => return Ok(None),
        };
        let tok = match tok {
            Ok(v) => v,
            Err(_) => &self.lexer.next().unwrap()?,
        };
        Ok(match tok {
            &YsonToken::LeftAngle => {
                self.lexer.next(); // Consume '<'
                let mut attributes = Vec::new();
                loop {
                    let tok = eof_guard!(self.lexer.next())?;
                    match tok {
                        YsonToken::RightAngle => {
                            break;
                        }
                        YsonToken::String(name) => {
                            let eq = eof_guard!(self.lexer.next())?;
                            if eq != YsonToken::EqualSign {
                                return Err(YsonParseError::UnexpectedToken(eq));
                            }
                            let value = self.parse_complete()?;
                            let next = eof_guard!(self.lexer.peek());
                            let _ = match next {
                                Err(_) | Ok(YsonToken::Semicolon) => {
                                    eof_guard!(self.lexer.next())?;
                                }
                                Ok(YsonToken::RightAngle) => {}
                                Ok(_) => {
                                    Err(YsonParseError::UnexpectedToken(
                                        eof_guard!(self.lexer.next()).unwrap(),
                                    ))?;
                                }
                            };
                            attributes.push((name, value));
                        }
                        tok => return Err(YsonParseError::UnexpectedToken(tok)),
                    }
                }
                Some(attributes)
            }
            _ => None,
        })
    }

    fn parse_array(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::LeftBracket => {
                let mut elements = Vec::new();
                loop {
                    let right_bracket = eof_guard!(self.lexer.peek());
                    match right_bracket {
                        Ok(YsonToken::RightBracket) => {
                            self.lexer.next();
                            break
                        },
                        Ok(_) => (),
                        Err(_) => {
                            eof_guard!(self.lexer.next())?;
                        }
                    }
                    let node = self.parse_complete()?;
                    elements.push(node);
                    let next = eof_guard!(self.lexer.peek());
                    match next {
                        Ok(YsonToken::RightBracket) => continue,
                        Ok(YsonToken::Semicolon) => { self.lexer.next(); },
                        Ok(_) => return Err(YsonParseError::UnexpectedToken(self.lexer.next().unwrap().unwrap())),
                        Err(_) => return Err(YsonParseError::IteratorError(self.lexer.next().unwrap().unwrap_err())),
                    }
                }
                Ok(YsonValue::Array(elements))
            }
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }

    fn parse_map(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::LeftBrace => {
                let mut pairs = Vec::new();
                loop {
                    let right_brace = eof_guard!(self.lexer.peek());
                    match right_brace {
                        Ok(YsonToken::RightBrace) => {
                            self.lexer.next();
                            break
                        },
                        Ok(_) => (),
                        Err(_) => {
                            eof_guard!(self.lexer.next())?;
                        }
                    }
                    let key_tok = eof_guard!(self.lexer.next())?;
                    let key = match key_tok {
                        YsonToken::String(s) => s,
                        tok => return Err(YsonParseError::UnexpectedToken(tok)),
                    };
                    let next = eof_guard!(self.lexer.next())?;
                    match next {
                        YsonToken::EqualSign => (),
                        tok => return Err(YsonParseError::UnexpectedToken(tok)),
                    }
                    let value = self.parse_complete()?;
                    pairs.push((key, value));
                    let next = eof_guard!(self.lexer.peek());
                    match next {
                        Ok(YsonToken::Semicolon) => {
                            self.lexer.next();
                        },
                        Ok(YsonToken::RightBrace) => (),
                        Ok(_) => return Err(YsonParseError::UnexpectedToken(self.lexer.next().unwrap().unwrap())),
                        Err(_) => return Err(YsonParseError::IteratorError(self.lexer.next().unwrap().unwrap_err())),
                    }
                }
                Ok(YsonValue::Map(pairs))
            }
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }

    fn parse_boolean(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::Boolean(b) => Ok(YsonValue::Boolean(b)),
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }

    fn parse_i64(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::SignedInteger(i) => Ok(YsonValue::SignedInteger(i)),
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }

    fn parse_u64(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::UnsignedInteger(u) => Ok(YsonValue::UnsignedInteger(u)),
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }

    fn parse_f64(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::Double(f) => Ok(YsonValue::Double(f)),
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }

    fn parse_string(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::String(s) => Ok(YsonValue::String(s)),
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }

    fn parse_entity(&mut self) -> Result<YsonValue, YsonParseError<E>> {
        match eof_guard!(self.lexer.next())? {
            YsonToken::Entity => Ok(YsonValue::Entity),
            tok => Err(YsonParseError::UnexpectedToken(tok)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{YsonLexError, YsonLexer};

    // Helper to make parsing from string easier in tests
    fn parse_str(
        input: &str,
    ) -> Result<YsonNode, YsonParseError<YsonLexError<std::convert::Infallible>>> {
        let lexer = YsonLexer::new(input.bytes().map(Ok));
        let mut parser = YsonParser::new(lexer);
        parser.parse_complete()
    }

    #[test]
    fn test_parse_i64() {
        let node = parse_str("123").unwrap();
        assert_eq!(node.value, YsonValue::SignedInteger(123));
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_u64() {
        let node = parse_str("123u").unwrap();
        assert_eq!(node.value, YsonValue::UnsignedInteger(123));
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_f64() {
        let node = parse_str("123.45").unwrap();
        assert_eq!(node.value, YsonValue::Double(123.45));
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_boolean() {
        let node = parse_str("%true").unwrap();
        assert_eq!(node.value, YsonValue::Boolean(true));
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_string() {
        let node = parse_str(r#""hello""#).unwrap();
        assert_eq!(node.value, YsonValue::String(b"hello".to_vec()));
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_entity() {
        let node = parse_str("#").unwrap();
        assert_eq!(node.value, YsonValue::Entity);
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_empty_array() {
        let node = parse_str("[]").unwrap();
        assert_eq!(node.value, YsonValue::Array(vec![]));
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_array() {
        let node = parse_str("[1;2;3]").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Array(vec![
                YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] },
                YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] },
                YsonNode { value: YsonValue::SignedInteger(3), attributes: vec![] },
            ])
        );
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_array_with_trailing_semicolon() {
        let node = parse_str("[1;2;3;]").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Array(vec![
                YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] },
                YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] },
                YsonNode { value: YsonValue::SignedInteger(3), attributes: vec![] },
            ])
        );
    }

    #[test]
    fn test_parse_empty_map() {
        let node = parse_str("{}").unwrap();
        assert_eq!(node.value, YsonValue::Map(vec![]));
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_map() {
        let node = parse_str("{a=1;b=2}").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Map(vec![
                (
                    b"a".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] }
                ),
                (
                    b"b".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] }
                ),
            ])
        );
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_parse_map_with_trailing_semicolon() {
        let node = parse_str("{a=1;b=2;}").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Map(vec![
                (
                    b"a".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] }
                ),
                (
                    b"b".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] }
                ),
            ])
        );
    }

    #[test]
    fn test_attributes() {
        let node = parse_str("<a=1;b=%true>2").unwrap();
        assert_eq!(node.value, YsonValue::SignedInteger(2));
        assert_eq!(node.attributes.len(), 2);
        assert_eq!(node.attributes[0].0, b"a".to_vec());
        assert_eq!(node.attributes[0].1.value, YsonValue::SignedInteger(1));
        assert_eq!(node.attributes[1].0, b"b".to_vec());
        assert_eq!(node.attributes[1].1.value, YsonValue::Boolean(true));
    }

    #[test]
    fn test_nested_with_attributes() {
        let node = parse_str("<a=1>[{x=y;z=<foo=bar>#}]").unwrap();
        assert_eq!(node.attributes[0].0, b"a".to_vec());
        assert_eq!(node.attributes[0].1.value, YsonValue::SignedInteger(1));
        if let YsonValue::Array(arr) = node.value {
            if let YsonValue::Map(map) = &arr[0].value {
                assert_eq!(map[0].0, b"x".to_vec());
                assert_eq!(map[0].1.value, YsonValue::String(b"y".to_vec()));
                assert_eq!(map[1].0, b"z".to_vec());
                assert_eq!(map[1].1.value, YsonValue::Entity);
                assert_eq!(map[1].1.attributes[0].0, b"foo".to_vec());
                assert_eq!(
                    map[1].1.attributes[0].1.value,
                    YsonValue::String(b"bar".to_vec())
                );
            } else {
                panic!("Expected a map");
            }
        } else {
            panic!("Expected an array");
        }
    }

    #[test]
    fn test_parse_list_with_whitespace() {
        let node = parse_str("[ 1 ; 2 ; 3 ]").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Array(vec![
                YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] },
                YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] },
                YsonNode { value: YsonValue::SignedInteger(3), attributes: vec![] },
            ])
        );
    }

    #[test]
    fn test_parse_map_with_whitespace() {
        let node = parse_str("{ a = 1 ; b = 2 } ").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Map(vec![
                (
                    b"a".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] }
                ),
                (
                    b"b".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] }
                ),
            ])
        );
    }

    #[test]
    fn test_string_with_escapes() {
        let node = parse_str("\"a\\\"b\\\\c\"").unwrap();
        assert_eq!(node.value, YsonValue::String(b"a\"b\\c".to_vec()));
    }

    #[test]
    fn test_nested_list() {
        let node = parse_str("[1;[2;3];4]").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Array(vec![
                YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] },
                YsonNode {
                    value: YsonValue::Array(vec![
                        YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] },
                        YsonNode { value: YsonValue::SignedInteger(3), attributes: vec![] },
                    ]),
                    attributes: vec![]
                },
                YsonNode { value: YsonValue::SignedInteger(4), attributes: vec![] },
            ])
        );
    }

    #[test]
    fn test_nested_map() {
        let node = parse_str("{a=1;b={c=2;d=3};e=4}").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Map(vec![
                (
                    b"a".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] }
                ),
                (
                    b"b".to_vec(),
                    YsonNode {
                        value: YsonValue::Map(vec![
                            (
                                b"c".to_vec(),
                                YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] }
                            ),
                            (
                                b"d".to_vec(),
                                YsonNode { value: YsonValue::SignedInteger(3), attributes: vec![] }
                            ),
                        ]),
                        attributes: vec![]
                    }
                ),
                (
                    b"e".to_vec(),
                    YsonNode { value: YsonValue::SignedInteger(4), attributes: vec![] }
                ),
            ])
        );
    }

    #[test]
    fn test_list_of_maps() {
        let node = parse_str("[{a=1};{b=2}]").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Array(vec![
                YsonNode {
                    value: YsonValue::Map(vec![(
                        b"a".to_vec(),
                        YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] }
                    )]),
                    attributes: vec![]
                },
                YsonNode {
                    value: YsonValue::Map(vec![(
                        b"b".to_vec(),
                        YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] }
                    )]),
                    attributes: vec![]
                },
            ])
        );
    }

    #[test]
    fn test_map_of_lists() {
        let node = parse_str("{a=[1;2];b=[3;4]}").unwrap();
        assert_eq!(
            node.value,
            YsonValue::Map(vec![
                (
                    b"a".to_vec(),
                    YsonNode {
                        value: YsonValue::Array(vec![
                            YsonNode { value: YsonValue::SignedInteger(1), attributes: vec![] },
                            YsonNode { value: YsonValue::SignedInteger(2), attributes: vec![] },
                        ]),
                        attributes: vec![]
                    }
                ),
                (
                    b"b".to_vec(),
                    YsonNode {
                        value: YsonValue::Array(vec![
                            YsonNode { value: YsonValue::SignedInteger(3), attributes: vec![] },
                            YsonNode { value: YsonValue::SignedInteger(4), attributes: vec![] },
                        ]),
                        attributes: vec![]
                    }
                ),
            ])
        );
    }

    #[test]
    fn test_error_unexpected_token_mismatched_brackets() {
        let result = parse_str("[1; 2}");
        assert!(result.is_err());
        match result.unwrap_err() {
            YsonParseError::UnexpectedToken(token) => {
                // Should get RightBrace when expecting RightBracket
                assert!(matches!(token, YsonToken::RightBrace));
            }
            _ => panic!("Expected UnexpectedToken error"),
        }
    }

    #[test]
    fn test_error_unexpected_end_of_input() {
        let result = parse_str("[1; 2");
        assert!(result.is_err());
        match result.unwrap_err() {
            YsonParseError::UnexpectedEndOfInput => {
                // This is the expected error
            }
            _ => panic!("Expected UnexpectedEndOfInput error"),
        }
    }

    #[test]
    fn test_error_unexpected_token_invalid_map_syntax() {
        let result = parse_str("{key=}");
        assert!(result.is_err());
        match result.unwrap_err() {
            YsonParseError::UnexpectedToken(token) => {
                // Should get RightBrace when expecting a value
                assert!(matches!(token, YsonToken::RightBrace));
            }
            _ => panic!("Expected UnexpectedToken error"),
        }
    }
}
