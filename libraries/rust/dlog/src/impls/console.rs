use crate::impls::LogWriter;
use crate::{LogLevel, LogRecord};
use std::io::Write;

pub(crate) struct ConsoleLogWriter {
    progresses: std::collections::HashMap<u64, (String, u64, u64)>,
    churn: u64,
    lines_written: u64,
    current_cursor_x: u64,
}

impl ConsoleLogWriter {
    pub fn new() -> Self {
        Self {
            progresses: std::collections::HashMap::new(),
            // This demands a small explanation. The problem here is that when we press enter,
            // The cursor is moved to the next line. But the system is built around the idea that
            // at the start of the last written line.
            // As a side effect, if we said that we have one "churny" line
            // that we have already written to the screen, the system will try to
            // reuse the churny line.
            // Therefore, we set the churn and lines_written to 1.
            churn: 1,
            lines_written: 1,
            current_cursor_x: 0,
        }
    }
}

impl ConsoleLogWriter {
    fn display_progress_bar(&self, description: &str, total: u64, value: u64) {
        let mut width: usize = 80;
        if let Some(size) = terminal_size::terminal_size() {
            width = size.0.0 as usize;
        }
        let bar_width = width.saturating_sub(description.len().min(40) + 12);
        let progress = value as f64 / (total.max(value)) as f64;
        let filled_width = (bar_width as f64 * progress) as usize;
        let empty_width = bar_width.saturating_sub(filled_width);

        let filled = "#".repeat(filled_width);
        let empty = "-".repeat(empty_width);

        let percent = (progress * 100.0) as usize;

        print!(
            "[ {} |{}{}| {:3}% ]",
            &description[..description.len().min(40)],
            filled,
            empty,
            percent,
        );
        std::io::stdout().flush().ok();
    }

    fn display_progress_bars(&mut self) {
        if self.progresses.is_empty() {
            return;
        }
        self.move_cursor_before_first_progress_bar();
        self.move_cursor_down(1);
        let mut ids: Vec<_> = self.progresses.keys().cloned().collect();
        ids.sort_by(|a, b| b.cmp(a));
        for id in ids {
            let (description, total, value) = &self.progresses[&id];
            self.move_cursor_to_line_start();
            self.clear_line();
            self.display_progress_bar(description, *total, *value);
            self.move_cursor_down(1);
        }
        self.move_cursor_to_default_position();
    }

    fn display_log_record(&self, record: &LogRecord) {
        self.set_color(match record.level {
            LogLevel::Trace => "37",
            LogLevel::Debug => "36",
            LogLevel::Info => "32",
            LogLevel::Warn => "33",
            LogLevel::Error => "31",
            LogLevel::Fatal => "35",
        });
        print!(
            "[{} {}]\x1b[0m {}",
            record.level, record.timestamp, &record.message
        );
        std::io::stdout().flush().ok();
    }

    fn set_color(&self, color: &str) {
        print!("\x1b[{}m", color);
    }

    fn move_cursor_before_first_progress_bar(&mut self) {
        self.move_cursor_to_default_position();
        self.move_cursor_up(self.progresses.len() as u64);
    }

    fn move_cursor_up(&mut self, mut n: u64) {
        if self.current_cursor_x + n >= self.lines_written {
            n -= (self.lines_written - self.current_cursor_x) - 1
        }
        self.current_cursor_x += n;
        if n > 0 {
            print!("\x1b[{}A", n);
            std::io::stdout().flush().ok();
        }
    }

    fn move_cursor_down(&mut self, mut n: u64) {
        if self.current_cursor_x < n {
            n = self.current_cursor_x;
        }
        if n > 0 {
            print!("\x1b[{}B", n);
            std::io::stdout().flush().ok();
        }
        self.current_cursor_x -= n;
    }

    fn clear_line(&self) {
        print!("\x1b[2K");
        std::io::stdout().flush().ok();
    }

    fn move_cursor_to_line_start(&self) {
        print!("\x1b[G");
        std::io::stdout().flush().ok();
    }

    fn append_line(&mut self) {
        self.move_cursor_to_default_position();
        self.move_cursor_to_line_start();
        println!();
        std::io::stdout().flush().ok();
        self.lines_written += 1;
    }

    fn move_cursor_to_default_position(&mut self) {
        self.move_cursor_down(self.current_cursor_x);
        self.move_cursor_to_line_start();
    }
}

impl LogWriter for ConsoleLogWriter {
    fn push_record(&mut self, record: &LogRecord) {
        self.move_cursor_to_default_position();
        if self.churn == 0 {
            self.move_cursor_before_first_progress_bar();
            self.move_cursor_down(1);
            self.move_cursor_to_line_start();
            if !self.progresses.is_empty() {
                self.clear_line();
            }
            self.move_cursor_to_default_position();
            self.append_line();
            self.churn += 1;
        }
        self.churn -= 1;
        self.move_cursor_before_first_progress_bar();
        self.move_cursor_up(self.churn);
        self.move_cursor_to_line_start();
        self.clear_line();
        self.display_log_record(record);
        self.move_cursor_down(self.churn);
        self.display_progress_bars();
    }
    fn start_progress(&mut self, id: u64) {
        self.progresses.insert(id, (String::new(), 0, 0));
        self.append_line();
    }

    fn update_progress(&mut self, id: u64, value: u64, total: u64, description: &str) {
        let progress = self.progresses.get_mut(&id).unwrap();
        if progress.0 != description {
            progress.0 = description.to_string();
        }
        progress.1 = total;
        progress.2 = value;

        self.display_progress_bars();
    }

    fn finish_progress(&mut self, id: u64) {
        self.move_cursor_before_first_progress_bar();
        self.move_cursor_down(1);
        self.clear_line();
        let (description, total, value) = self.progresses.get(&id).unwrap();
        self.display_progress_bar(description, *total, *value);
        self.progresses.remove(&id);
        self.churn += 1;
        self.display_progress_bars();
    }

    fn complete(&mut self) {
        println!();
    }
}
