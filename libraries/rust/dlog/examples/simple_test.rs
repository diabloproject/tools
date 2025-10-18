use dlog::{LogLevel, ProgressBar, log};

fn main() {
    log(LogLevel::Info, "Starting application");
    log(LogLevel::Debug, "This is a debug message");

    let pb = ProgressBar::new("Processing items", 100);
    for i in 0..=100 {
        pb.push(i);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    drop(pb);

    log(LogLevel::Info, "Processing complete");
    log(LogLevel::Warn, "This is a warning");
    log(LogLevel::Error, "This is an error");

    let pb2 = ProgressBar::new("Second task", 50);
    for i in 0..=50 {
        pb2.push(i);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    drop(pb2);

    log(LogLevel::Info, "Application finished");
    std::thread::sleep(std::time::Duration::from_millis(100));
}
