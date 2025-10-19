pub mod block;
pub mod event_type;
pub mod project;

use crate::block::{Block, Variable};
use crate::event_type::EventType;
use crate::project::Project;

#[derive(Debug)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Parse error at line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

pub struct Parser<'src> {
    source: &'src str,
    pos: usize,
}

impl<'src> Parser<'src> {
    pub fn new(input: &'src str) -> Self {
        Parser {
            source: input,
            pos: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Project<'src>, ParseError> {
        let mut blocks = Vec::new();

        while self.pos < self.source.len() {
            let Some(block) = self.parse_block()? else {
                break;
            };
            blocks.push(block);
        }

        Ok(Project {
            blocks: blocks.try_into().expect("Too many blocks"),
        })
    }

    fn skip_whitespace_and_comments(&mut self) {
        let mut in_comment = false;
        let content_bytes = self.source.as_bytes();
        while self.pos < self.source.len() {
            let c = content_bytes[self.pos];
            if c == b';' {
                in_comment = true;
            } else if c == b'\n' {
                in_comment = false;
            } else if c == b' ' || c == b'\t' || in_comment {
            } else {
                break;
            }
            self.pos += 1;
        }
    }

    fn parse_identifier(&mut self) -> Result<&'src str, ParseError> {
        let identifier_start = self.pos;
        let content_bytes = self.source.as_bytes();
        while self.pos < self.source.len() {
            let c = content_bytes[self.pos];
            if !(c.is_ascii_alphanumeric() || c == b'_') {
                break;
            }
            self.pos += 1;
        }
        if self.pos == identifier_start {
            return Err(ParseError {
                line: self.pos,
                message: "Expected identifier, found none".to_string(),
            });
        }
        let identifier_end = self.pos - 1;
        Ok(&self.source[identifier_start..=identifier_end])
    }

    fn consume_string(&mut self, seq: &str) -> Result<(), ParseError> {
        let seq_bytes = seq.as_bytes();
        let content_bytes = self.source.as_bytes();
        for (i, c) in seq_bytes.iter().enumerate() {
            let cs = content_bytes[self.pos + i];
            if cs != *c {
                return Err(ParseError {
                    line: self.pos,
                    message: format!("Expected {}, got {}", seq, cs as char),
                });
            }
        }
        self.pos += seq_bytes.len();
        Ok(())
    }

    fn parse_block(&mut self) -> Result<Option<Block<'src>>, ParseError> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.source.len() {
            return Ok(None);
        }

        match self.parse_event_block()? {
            Some(block) => return Ok(Some(block)),
            None => {}
        }

        Ok(None) // TODO: Implement other block types
    }

    fn parse_event_block(&mut self) -> Result<Option<Block<'src>>, ParseError> {
        self.skip_whitespace_and_comments();
        self.consume_string("ON:")?;
        let block_type = self.parse_identifier()?;
        let ty = match block_type.trim() {
            "BUILD" => EventType::Build,
            _ => return Ok(None),
        };
        self.skip_whitespace_and_comments();
        self.consume_string("DO")?;
        let body = self.parse_block_body()?;
        Ok(Some(Block::Event {
            ty,
            content: body
                .into_iter()
                .map(|(name, value)| Variable { name, value })
                .collect::<Vec<_>>()
                .try_into()
                .expect("Too many variables"),
        }))
    }

    fn parse_block_body(&mut self) -> Result<Vec<(&'src str, &'src str)>, ParseError> {
        let mut var_list = Vec::new();
        let content_bytes = self.source.as_bytes();
        self.skip_whitespace_and_comments();
        loop {
            let name = self.parse_identifier()?;
            self.skip_whitespace_and_comments();
            if name == "STRING" {
                let name = self.parse_identifier()?;
                self.skip_whitespace_and_comments();
                let content_start = self.pos;
                let mut content_end = self.pos;
                loop {
                    let old_pod = self.pos;
                    match self.consume_string("EOF") {
                        Ok(_) => break,
                        _ => {}
                    };
                    self.pos = old_pod;

                    loop {
                        let c = content_bytes[self.pos];
                        if c == b'\n' {
                            content_end = self.pos;
                            self.pos += 1;
                            break;
                        }
                        self.pos += 1;
                    }
                }
                var_list.push((name, &self.source[content_start..=content_end]));
            } else if name == "END" {
                break;
            } else {
                let content_start = self.pos;
                let content_end;
                loop {
                    let c = content_bytes[self.pos];
                    if c == b'\n' {
                        content_end = self.pos - 1;
                        self.pos += 1;
                        break;
                    }
                    self.pos += 1;
                }
                var_list.push((name, &self.source[content_start..=content_end]));
            }
            self.skip_whitespace_and_comments();
        }

        Ok(var_list)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_parser() {
        let input = "ON:BUILD DO\nSTRING message Hello World!\nEOF\nEND";
        let mut parser = Parser::new(input);
        let project = parser.parse().unwrap();

        assert_eq!(project.blocks.len(), 1);

        match project.blocks[0] {
            Block::Event { ty, content } => {
                assert_eq!(ty, EventType::Build);
                assert_eq!(content.len(), 1);
                assert_eq!(content[0].name, "message");
                assert_eq!(content[0].value, "Hello World!\n");
            }
            _ => panic!("Expected Event block"),
        }
    }

    #[test]
    fn test_parser_simple() {
        let input = "ON:BUILD DO\nMESSAGE hello\nEND";
        let mut parser = Parser::new(input);
        let project = parser.parse().unwrap();

        assert_eq!(project.blocks.len(), 1);

        match project.blocks[0] {
            Block::Event { ty, content } => {
                assert_eq!(ty, EventType::Build);
                assert_eq!(content.len(), 1);
                assert_eq!(content[0].name, "MESSAGE");
                assert_eq!(content[0].value, "hello");
            }
            _ => panic!("Expected Event block"),
        }
    }

    #[test]
    fn test_parse_empty() {
        let input = "";
        let mut parser = Parser::new(input);
        let project = parser.parse().unwrap();
        assert_eq!(project.blocks.len(), 0);
    }

    #[test]
    fn test_parse_whitespace_and_comments() {
        let input =
            "  ; Comment\n  ON:BUILD DO\n  ; Another comment\n  STRING message Hi\nEOF\nEND  ";
        let mut parser = Parser::new(input);
        let project = parser.parse().unwrap();

        assert_eq!(project.blocks.len(), 1);
        match project.blocks[0] {
            Block::Event { ty, content } => {
                assert_eq!(ty, EventType::Build);
                assert_eq!(content.len(), 1);
                assert_eq!(content[0].name, "message");
                assert_eq!(content[0].value, "Hi\n");
            }
            _ => panic!("Expected Event block"),
        }
    }

    #[test]
    fn test_invalid_event() {
        let input = "ON:INVALID DO\nEND";
        let mut parser = Parser::new(input);
        let project = parser.parse().unwrap();
        assert_eq!(project.blocks.len(), 0);
    }
}
