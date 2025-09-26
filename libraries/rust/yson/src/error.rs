use crate::protoshim::ProtoshimError;
use ::thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum YsonParseErrorVariant {
    #[error("Reached EOF before value was completed")]
    IncompleteValue,
    #[error("Parser was not implemented")]
    NotImplementedError,
    #[error("{0}")]
    ProtoshimError(ProtoshimError),
    // Add string parsing errors
    #[error("Invalid escape sequence: \\{0}")]
    InvalidEscape(char),
    #[error("Invalid hexadecimal digit in escape sequence")]
    InvalidHexDigit,
    #[error("Invalid octal digit in escape sequence")]
    InvalidOctalDigit,
    #[error("Unterminated string literal")]
    UnterminatedString,
    #[error("Invalid UTF-8 in string")]
    InvalidUtf8,
}


#[derive(Error, Debug)]
pub struct YsonParseError {
    pub variant: YsonParseErrorVariant,
    /// how many bytes since the start of the input
    pub at: usize,
}

impl From<ProtoshimError> for YsonParseError {
    fn from(value: ProtoshimError) -> Self {
        let at = value.at();
        Self {
            variant: YsonParseErrorVariant::ProtoshimError(value),
            at,
        }
    }
}

impl std::fmt::Display for YsonParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reached error \"{}\" {} bytes into the sequence",
            self.variant, self.at
        )
    }
}

impl YsonParseError {
    pub fn incomplete(at: usize) -> Self {
        return Self {
            variant: YsonParseErrorVariant::IncompleteValue,
            at,
        };
    }

    pub fn todo(at: usize) -> Self {
        return Self {
            variant: YsonParseErrorVariant::NotImplementedError,
            at,
        };
    }
}
