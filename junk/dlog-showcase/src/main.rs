use dlog::{LogLevel, ProgressBar, log};
use std::thread;
use std::time::Duration;

fn main() {
    log(LogLevel::Info, "Starting dlog showcase application");
    thread::sleep(Duration::from_millis(500));
    log(
        LogLevel::Debug,
        "Debug message - application initialization",
    );
    thread::sleep(Duration::from_millis(500));

    // Simulate some work with a single progress bar
    log(LogLevel::Info, "Processing first task with progress bar");
    let pb1 = ProgressBar::new("Downloading files", 100);
    for i in 0..=100 {
        thread::sleep(Duration::from_millis(30));
        pb1.push(i);
        if i % 25 == 0 && i > 0 {
            log(LogLevel::Info, format!("Checkpoint reached: {}%", i));
        }
    }
    drop(pb1);
    log(LogLevel::Info, "First task completed");

    // Multiple concurrent progress bars
    log(LogLevel::Info, "Starting multiple concurrent tasks");
    let pb2 = ProgressBar::new("Processing data", 50);
    let pb3 = ProgressBar::new("Analyzing results", 30);

    for i in 0..=50 {
        thread::sleep(Duration::from_millis(40));
        pb2.push(i);

        if i <= 30 {
            pb3.push(i);
        }

        if i == 10 {
            log(LogLevel::Warn, "Warning: High memory usage detected");
        }
        if i == 25 {
            log(LogLevel::Debug, "Debug: Intermediate state saved");
        }
    }
    drop(pb2);
    drop(pb3);

    log(LogLevel::Info, "All tasks completed successfully");

    // Demonstrate different log levels
    log(LogLevel::Trace, "Trace: Very detailed information");
    log(LogLevel::Debug, "Debug: Diagnostic information");
    log(LogLevel::Info, "Info: General information");
    log(LogLevel::Warn, "Warn: Warning message");
    log(LogLevel::Error, "Error: Error occurred but recoverable");
    log(LogLevel::Fatal, "Fatal: Critical error (simulated)");

    log(LogLevel::Info, "Showcase completed");

    // Give the log thread time to flush
    thread::sleep(Duration::from_millis(100));
}
