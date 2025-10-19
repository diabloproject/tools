mod impls;
mod pipe;

use crate::impls::LogWriter;
use crate::impls::console::ConsoleLogWriter;
use crate::impls::simple::SimpleLogWriter;
use std::net::TcpStream;
use std::str::FromStr;
use std::sync::LazyLock;
use std::time::SystemTime;
use stdd::time::DateTime;

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
    pub(crate) level: LogLevel,
    pub(crate) message: String,
    pub(crate) timestamp: DateTime,
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
    Drop {
        tx: std::sync::mpsc::Sender<()>,
    },
}

enum LogSink {
    Local(std::sync::mpsc::Sender<LogEvent>),
    Remote(std::sync::Mutex<TcpStream>),
}

impl LogSink {
    fn send(&self, event: LogEvent) -> Result<(), String> {
        match self {
            LogSink::Local(tx) => tx.send(event).map_err(|_| "Channel closed".to_string()),
            LogSink::Remote(stream) => {
                if let LogEvent::Drop { tx } = event {
                    tx.send(()).expect("Failed to send drop event");
                    return Ok(());
                }
                let mut stream = stream.lock().unwrap();
                pipe::send_log_event(&mut stream, &event)
                    .map_err(|e| format!("Failed to send to pipe: {}", e))
            }
        }
    }
}

fn log_thread(rx: std::sync::mpsc::Receiver<LogEvent>) {
    let minimum_log_level = std::env::var("DLOG_LEVEL").unwrap_or_else(|_| "trace".to_string());
    let minimum_log_level = LogLevel::from_str(&minimum_log_level).unwrap();

    let (forward_tx, forward_rx) = std::sync::mpsc::channel();
    let pipe_addr = pipe::start_pipe_listener(forward_tx);
    if let Ok(addr) = &pipe_addr {
        std::env::set_var("DLOG_PIPE", addr);
    }

    let use_progress = std::env::var("DLOG_NO_PROGRESS").unwrap_or_else(|_| "FALSE".to_string());
    let use_progress = !if use_progress.to_ascii_lowercase() == "true" {
        true
    } else {
        false
    };

    let use_simple = std::env::var("DLOG_SIMPLE").unwrap_or_else(|_| "FALSE".to_string());
    let use_simple = use_simple.to_ascii_lowercase() == "true";

    let mut writer: Box<dyn LogWriter> = if use_simple {
        Box::new(SimpleLogWriter::new())
    } else {
        Box::new(ConsoleLogWriter::new())
    };

    let mut progress_bars = std::collections::HashMap::new();
    loop {
        let next = match rx.try_recv() {
            Ok(next) => next,
            Err(std::sync::mpsc::TryRecvError::Empty) => match forward_rx.try_recv() {
                Ok(next) => next,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            },
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        };
        match next {
            LogEvent::Log(record) => {
                if record.level >= minimum_log_level {
                    writer.push_record(&record);
                }
            }
            LogEvent::CreateProgressBar {
                id,
                description,
                total,
            } => {
                if !use_progress {
                    continue;
                }
                writer.start_progress(id);
                progress_bars.insert(id, (description.clone(), total));
                writer.update_progress(id, 0, total, &description);
            }
            LogEvent::PushProgress { id, value } => {
                if !use_progress {
                    continue;
                }
                let bar = progress_bars.get(&id).expect("Invalid progress id");
                writer.update_progress(id, value, bar.1, &bar.0);
            }
            LogEvent::FinishProgressBar { id } => {
                if !use_progress {
                    continue;
                }
                progress_bars.remove(&id);
                writer.finish_progress(id);
            }
            LogEvent::Drop { tx } => {
                writer.complete();
                tx.send(()).expect("Failed to send drop event");
            }
        }
    }
}

static LOG_SOCKET: LazyLock<LogSink> = LazyLock::new(|| {
    if let Ok(pipe_addr) = std::env::var("DLOG_PIPE") {
        if let Ok(stream) = TcpStream::connect(&pipe_addr) {
            return LogSink::Remote(std::sync::Mutex::new(stream));
        }
    }

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || log_thread(rx));
    LogSink::Local(tx)
});

pub fn log(level: LogLevel, message: impl Into<String>) {
    LOG_SOCKET
        .send(LogEvent::Log(LogRecord {
            level,
            message: message.into(),
            timestamp: DateTime::now(),
        }))
        .expect("Failed to send log event");
}

pub fn finish() {
    let (tx, rx) = std::sync::mpsc::channel();
    LOG_SOCKET
        .send(LogEvent::Drop { tx })
        .expect("Failed to send log event");
    rx.recv().expect("Failed to receive drop event");
}

pub struct ProgressBar(u64);

impl ProgressBar {
    pub fn new(description: impl Into<String>, total: u64) -> ProgressBar {
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
            .expect("Failed to send log event");
        Self(id)
    }

    pub fn push(&self, value: u64) {
        LOG_SOCKET
            .send(LogEvent::PushProgress { id: self.0, value })
            .expect("Failed to send log event");
    }
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        LOG_SOCKET
            .send(LogEvent::FinishProgressBar { id: self.0 })
            .expect("Failed to send log event")
    }
}
