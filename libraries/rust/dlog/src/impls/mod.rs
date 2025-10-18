use crate::LogRecord;
use std::time::SystemTime;

pub mod console;

pub trait LogWriter {
    fn push_record(&mut self, record: &LogRecord);
    fn start_progress(&mut self, id: u64);
    fn update_progress(&mut self, id: u64, total: u64, description: &str);
    fn finish_progress(&mut self, id: u64);
    fn force_flush(&mut self, now: SystemTime);
}
