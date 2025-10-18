use crate::LogRecord;

pub mod console;

pub(crate) trait LogWriter {
    fn push_record(&mut self, record: &LogRecord);
    fn start_progress(&mut self, id: u64);
    fn update_progress(&mut self, id: u64, value: u64, total: u64, description: &str);
    fn finish_progress(&mut self, id: u64);
}
