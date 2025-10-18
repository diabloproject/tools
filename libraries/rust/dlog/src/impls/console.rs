use crate::impls::LogWriter;
use crate::{LogLevel, LogRecord};
use std::time::SystemTime;

pub struct ConsoleLogWriter {}

impl ConsoleLogWriter {
    pub fn new() -> Self {
        Self {}
    }
}

impl LogWriter for ConsoleLogWriter {
    fn push_record(&mut self, record: &LogRecord) {
        println!(
            "\x1b[{}m[{} {}]\x1b[0m {}",
            match record.level {
                LogLevel::Trace => "37",
                LogLevel::Debug => "36",
                LogLevel::Info => "32",
                LogLevel::Warn => "33",
                LogLevel::Error => "31",
                LogLevel::Fatal => "35",
            },
            record.level,
            record.timestamp,
            &record.message
        );
    }

    fn start_progress(&mut self, id: u64) {
        todo!()
    }

    fn update_progress(&mut self, id: u64, total: u64, description: &str) {
        todo!()
    }

    fn finish_progress(&mut self, id: u64) {
        todo!()
    }

    fn force_flush(&mut self, now: SystemTime) {
        todo!()
    }
}
