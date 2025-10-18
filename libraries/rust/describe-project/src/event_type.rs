use std::str::FromStr;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    Build,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EventType::Build => write!(f, "BUILD"),
        }
    }
}

impl FromStr for EventType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "BUILD" => Ok(EventType::Build),
            _ => Err(format!("Unknown event type: {}", s)),
        }
    }
}
