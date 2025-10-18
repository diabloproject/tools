mod impls;

use std::str::FromStr;
use crate::impls::LogWriter;
use crate::impls::console::ConsoleLogWriter;
use stdd::time::DateTime;
use std::sync::LazyLock;
use std::time::SystemTime;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRC"),
            LogLevel::Debug => write!(f, "DBG"),
            LogLevel::Info => write!(f, "INF"),
            LogLevel::Warn => write!(f, "WAR"),
            LogLevel::Error => write!(f, "ERR"),
            LogLevel::Fatal => write!(f, "FAT"),
        }
    }
}

#[derive(Debug)]
pub struct InvalidLogLevel;

impl std::fmt::Display for InvalidLogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid log level")
    }
}

impl std::error::Error for InvalidLogLevel {}

impl FromStr for LogLevel {
    type Err = InvalidLogLevel;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "trace" => Ok(LogLevel::Trace),
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warn" => Ok(LogLevel::Warn),
            "error" => Ok(LogLevel::Error),
            "fatal" => Ok(LogLevel::Fatal),
            _ => Err(InvalidLogLevel),
        }
    }
}

pub(crate) struct LogRecord {
    level: LogLevel,
    message: String,
    timestamp: DateTime,
}

enum LogEvent {
    Log(LogRecord),
    CreateProgressBar {
        id: u64,
        description: String,
        total: u64,
    },
    PushProgress {
        id: u64,
        value: u64,
    },
    FinishProgressBar {
        id: u64,
    },
}

fn log_thread(rx: std::sync::mpsc::Receiver<LogEvent>) {
    let minimum_log_level = std::env::var("DLOG_LEVEL").unwrap_or_else(|_| "trace".to_string());
    let minimum_log_level = LogLevel::from_str(&minimum_log_level).unwrap();
    let mut console = ConsoleLogWriter::new();
    let mut progress_bars = std::collections::HashMap::new();
    loop {
        let next = match rx.recv() {
            Ok(next) => next,
            Err(_channel_closed) => break,
        };
        match next {
            LogEvent::Log(record) => {
                if record.level >= minimum_log_level {
                    console.push_record(&record);
                }
            }
            LogEvent::CreateProgressBar { id, description, total } => {
                console.start_progress(id);
                progress_bars.insert(id, (description.clone(), total));
                console.update_progress(id, 0, total, &description);
            }
            LogEvent::PushProgress { id, value } => {
                let bar = progress_bars.get(&id).expect("Invalid progress id");
                console.update_progress(id, value, bar.1, &bar.0);
            }
            LogEvent::FinishProgressBar { id } => {
                progress_bars.remove(&id);
                console.finish_progress(id);
            }
        }
    }
}

static LOG_SOCKET: LazyLock<std::sync::mpsc::Sender<LogEvent>> = LazyLock::new(|| {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || log_thread(rx));
    tx
});

pub fn log(level: LogLevel, message: impl Into<String>) {
    LOG_SOCKET
        .send(LogEvent::Log(LogRecord {
            level,
            message: message.into(),
            timestamp: DateTime::now(),
        }))
        .expect("Failed to send log event. Thread died?");
}

pub struct ProgressBar(u64);

impl ProgressBar {
    pub fn new(description: impl Into<String>, total: u64) -> ProgressBar {
        // Choose id based on current time
        let id = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        LOG_SOCKET
            .send(LogEvent::CreateProgressBar {
                id,
                description: description.into(),
                total,
            })
            .expect("Failed to send log event. Thread died?");
        Self(id)
    }

    pub fn push(&self, value: u64) {
        LOG_SOCKET
            .send(LogEvent::PushProgress { id: self.0, value })
            .expect("Failed to send log event. Thread died?");
    }
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        LOG_SOCKET
            .send(LogEvent::FinishProgressBar { id: self.0 })
            .expect("Failed to send log event. Thread died?")
    }
}
