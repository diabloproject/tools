use dlog::{LogLevel, ProgressBar, finish, log};
use std::process::Command;

fn main() {
    // Check if we're the child process
    if std::env::args().any(|arg| arg == "--child") {
        run_child();
        return;
    }

    // Parent process
    log(LogLevel::Info, "Parent: Starting up");

    // Give log thread time to initialize
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Create a progress bar in parent
    let pb = ProgressBar::new("Parent task", 100);

    // Get the pipe address
    let pipe_addr = std::env::var("DLOG_PIPE").expect("DLOG_PIPE not set");

    // Spawn child process
    let child = Command::new(std::env::current_exe().unwrap())
        .arg("--child")
        .env("DLOG_PIPE", &pipe_addr)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn child");

    // Update parent progress while child runs
    for i in 0..=100 {
        pb.push(i);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    drop(pb);
    log(LogLevel::Info, "Parent: Progress complete");

    // Wait for child
    let status = child.wait_with_output().expect("Failed to wait for child");
    std::thread::sleep(std::time::Duration::from_millis(100));
    log(
        LogLevel::Info,
        format!("Parent: Child exited with status: {}", status.status),
    );

    log(LogLevel::Info, "Parent: Complete");
    finish()
}

fn run_child() {
    log(LogLevel::Info, "Child: Starting");

    let pb = ProgressBar::new("Child task", 50);
    for i in 0..=50 {
        pb.push(i);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    drop(pb);

    log(LogLevel::Info, "Child: Complete");
    std::thread::sleep(std::time::Duration::from_millis(50));
    finish()
}
