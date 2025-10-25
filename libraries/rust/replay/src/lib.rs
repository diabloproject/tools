use yson::lexer::{YsonLexError, YsonLexer};
use yson::parser::{YsonParseError, YsonParser};
use yson::types::YsonValue;

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayRow {
    pub timestamp: u64,
    pub log: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Replay {
    pub rows: Vec<ReplayRow>,
    pub client: String,
}

impl Replay {
    pub fn new() -> Self {
        Replay {
            rows: Vec::new(),
            client: String::new(),
        }
    }

    pub fn to_yson(&self) -> String {
        let mut rows = String::new();
        for row in &self.rows {
            rows.push_str(&self.write_yson_replay_row(row));
            rows.push(';');
        }

        format!("{{rows=[{}];client={:?};}}", rows, self.client)
    }

    fn write_yson_replay_row(&self, v: &ReplayRow) -> String {
        format!("{{timestamp={:?}u;log={:?};}}", v.timestamp, v.log)
    }

    pub fn from_yson(s: &str) -> Result<Self, &'static str> {
        let lexer = YsonLexer::new(
            s.bytes()
                .map(|x| Ok(x) as Result<u8, YsonParseError<YsonLexError<std::io::Error>>>),
        );
        let mut parser = YsonParser::new(lexer);
        let root = parser.parse_complete().map_err(|_| "Invalid YSON")?;
        match root.value {
            YsonValue::Map(map) => {
                let mut replay = Replay::new();
                let mut client_found = false;
                let mut rows_found = false;

                for (k, v) in map {
                    if k == b"rows" {
                        let rows = v.value;
                        match rows {
                            YsonValue::Array(array) => {
                                for row in array {
                                    replay.rows.push(Self::read_yson_replay_row(&row.value)?);
                                }
                            }
                            _ => return Err("Invalid rows"),
                        }
                        rows_found = true;
                    } else if k == b"client" {
                        replay.client = match v.value {
                            YsonValue::String(ref s) => {
                                std::str::from_utf8(s.as_slice()).unwrap().to_owned()
                            }
                            _ => return Err("Invalid client"),
                        };
                        client_found = true;
                    } else {
                        return Err("Invalid key");
                    }
                }
                if !rows_found {
                    return Err("Invalid rows");
                }
                if !client_found {
                    return Err("Invalid client");
                }
                Ok(replay)
            }
            _ => Err("Invalid YSON"),
        }
    }
    fn read_yson_replay_row(v: &YsonValue) -> Result<ReplayRow, &'static str> {
        match v {
            YsonValue::Map(map) => {
                let mut row = ReplayRow {
                    timestamp: 0,
                    log: String::new(),
                };
                let mut timestamp_found = false;
                let mut log_found = false;

                for (k, v) in map {
                    if k == b"timestamp" {
                        row.timestamp = match v.value {
                            YsonValue::UnsignedInteger(i) => i,
                            _ => return Err("Invalid timestamp"),
                        };
                        timestamp_found = true;
                    } else if k == b"log" {
                        row.log = match v.value {
                            YsonValue::String(ref s) => {
                                std::str::from_utf8(s.as_slice()).unwrap().to_owned()
                            }
                            _ => return Err("Invalid log"),
                        };
                        log_found = true;
                    } else {
                        return Err("Invalid key");
                    }
                }
                if !timestamp_found {
                    return Err("Invalid timestamp");
                }
                if !log_found {
                    return Err("Invalid log");
                }
                Ok(row)
            }
            _ => Err(""),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use yson::types::YsonNode;

    #[test]
    fn test_replay_new() {
        let replay = Replay::new();
        assert!(replay.rows.is_empty());
        assert_eq!(replay.client, "");
    }

    #[test]
    fn test_to_yson() {
        let mut replay = Replay::new();
        replay.client = "test_client".to_string();
        replay.rows.push(ReplayRow {
            timestamp: 123456789,
            log: "Test log".to_string(),
        });

        let yson_str = replay.to_yson();
        assert_eq!(
            yson_str,
            "{rows=[{timestamp=123456789u;log=\"Test log\";};];client=\"test_client\";}"
        );
    }

    #[test]
    fn test_from_yson() {
        let yson_str = "{rows=[{timestamp=123456789u;log=\"Test log\";};];client=\"test_client\";}";
        let replay_result = Replay::from_yson(yson_str);

        assert!(replay_result.is_ok());
        let replay = replay_result.unwrap();
        assert_eq!(replay.client, "test_client");
        assert_eq!(replay.rows.len(), 1);
        let row = &replay.rows[0];
        assert_eq!(row.timestamp, 123456789);
        assert_eq!(row.log, "Test log");
    }

    #[test]
    fn test_invalid_yson_format() {
        let invalid_yson_str = "{invalid=yson}";
        let replay_result = Replay::from_yson(invalid_yson_str);
        assert!(replay_result.is_err());
    }

    #[test]
    fn test_read_yson_replay_row() {
        let map = [
            (
                b"timestamp".to_vec(),
                YsonNode {
                    value: YsonValue::UnsignedInteger(123),
                    attributes: vec![],
                },
            ),
            (
                b"log".to_vec(),
                YsonNode {
                    value: YsonValue::String(b"Log entry".to_vec()),
                    attributes: vec![],
                },
            ),
        ]
        .iter()
        .cloned()
        .collect();

        let yson_value = YsonValue::Map(map);
        let row_result = Replay::read_yson_replay_row(&yson_value);

        assert!(row_result.is_ok());
        let row = row_result.unwrap();
        assert_eq!(row.timestamp, 123);
        assert_eq!(row.log, "Log entry");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original_replay = Replay {
            rows: vec![
                ReplayRow {
                    timestamp: 123456789,
                    log: "First log".to_string(),
                },
                ReplayRow {
                    timestamp: 987654321,
                    log: "Second log".to_string(),
                },
            ],
            client: "test_client".to_string(),
        };

        let yson_str = original_replay.to_yson();
        let deserialized_replay = Replay::from_yson(&yson_str).expect("Deserialization failed");

        assert_eq!(original_replay, deserialized_replay);
    }
}
