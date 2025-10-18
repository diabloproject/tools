use crate::impls::LogWriter;
use crate::{LogLevel, LogRecord};
use std::io::Write;

pub(crate) struct SimpleLogWriter {
    progresses: std::collections::HashMap<u64, (String, u64)>,
}

impl SimpleLogWriter {
    pub fn new() -> Self {
        Self {
            progresses: std::collections::HashMap::new(),
        }
    }
}

impl SimpleLogWriter {
    fn display_log_record(&self, record: &LogRecord) {
        self.set_color(match record.level {
            LogLevel::Trace => "37",
            LogLevel::Debug => "36",
            LogLevel::Info => "32",
            LogLevel::Warn => "33",
            LogLevel::Error => "31",
            LogLevel::Fatal => "35",
        });
        println!(
            "[{} {}]\x1b[0m {}",
            record.level, record.timestamp, &record.message
        );
        std::io::stdout().flush().ok();
    }

    fn set_color(&self, color: &str) {
        print!("\x1b[{}m", color);
    }
}

impl LogWriter for SimpleLogWriter {
    fn push_record(&mut self, record: &LogRecord) {
        self.display_log_record(record);
    }

    fn start_progress(&mut self, id: u64) {
        self.progresses.insert(id, (String::new(), 0));
    }

    fn update_progress(&mut self, id: u64, _value: u64, total: u64, description: &str) {
        let progress = self.progresses.get_mut(&id).unwrap();
        if progress.0 != description {
            progress.0 = description.to_string();
        }
        progress.1 = total;
    }

    fn finish_progress(&mut self, id: u64) {
        if let Some((description, total)) = self.progresses.remove(&id) {
            println!("Progress completed: {} ({})", description, total);
            std::io::stdout().flush().ok();
        }
    }

    fn complete(&mut self) {}
}
