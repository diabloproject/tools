use dlog::{LogLevel, log};
use std::process::Command;

fn main() {
    // Parent process - starts logging and spawns child
    println!("=== Parent process starting ===");

    log(LogLevel::Info, "Parent: Starting up");
    log(LogLevel::Debug, "Parent: About to spawn child process");

    // Get the current executable path
    let current_exe = std::env::current_exe().expect("Failed to get current exe");

    // Check if we're the child process
    if std::env::args().any(|arg| arg == "--child") {
        println!("=== Child process starting ===");
        println!("DLOG_PIPE env var: {:?}", std::env::var("DLOG_PIPE"));
        run_child();
        return;
    }

    // Give log thread time to start and set up the listener
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Get the pipe address
    let pipe_addr = std::env::var("DLOG_PIPE").expect("DLOG_PIPE not set");
    println!("Parent pipe address: {}", pipe_addr);

    // Spawn child process with DLOG_PIPE env var
    let output = Command::new(&current_exe)
        .arg("--child")
        .env("DLOG_PIPE", &pipe_addr)
        .output()
        .expect("Failed to spawn child");
    log(
        LogLevel::Trace,
        format!(
            "Here are all the output produced by the child process: {:?}",
            String::from_utf8(output.stdout)
        ),
    );
    log(
        LogLevel::Info,
        format!("Parent: Child exited with status: {}", output.status),
    );
    log(LogLevel::Debug, "Parent: All done");

    std::thread::sleep(std::time::Duration::from_millis(200));
}

fn run_child() {
    // Child process - should send logs to parent via pipe
    log(LogLevel::Info, "Child: Hello from child process!");
    log(LogLevel::Warn, "Child: This is a warning from child");
    log(LogLevel::Error, "Child: This is an error from child");

    // Give logs time to flush
    std::thread::sleep(std::time::Duration::from_millis(50));
}
