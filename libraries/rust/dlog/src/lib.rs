mod impls;

use crate::impls::LogWriter;
use crate::impls::console::ConsoleLogWriter;
use diabloproject::std::time::DateTime;
use std::sync::LazyLock;
use std::time::SystemTime;

pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
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

struct LogRecord {
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
    let mut console = ConsoleLogWriter::new();
    loop {
        let next = match rx.recv() {
            Ok(next) => next,
            Err(_channel_closed) => break,
        };
        match next {
            LogEvent::Log(record) => {
                console.push_record(&record);
            }
            LogEvent::CreateProgressBar { .. } => {}
            LogEvent::PushProgress { .. } => {}
            LogEvent::FinishProgressBar { .. } => {}
        }
    }
}

static LOG_SOCKET: LazyLock<std::sync::mpsc::Sender<LogEvent>> = LazyLock::new(|| {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || log_thread(rx));
    tx
});

pub fn log(level: LogLevel, message: String) {
    LOG_SOCKET
        .send(LogEvent::Log(LogRecord {
            level,
            message,
            timestamp: DateTime::now(),
        }))
        .expect("Failed to send log event. Thread died?");
}

pub struct ProgressBar(u64);

impl ProgressBar {
    pub fn new(description: String, total: u64) -> ProgressBar {
        // Choose id based on current time
        let id = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        LOG_SOCKET
            .send(LogEvent::CreateProgressBar {
                id,
                description,
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
