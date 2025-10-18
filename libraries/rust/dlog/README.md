# dlog - Distributed Logging for Rust

A logging library with support for inter-process communication (IPC), allowing child processes to send logs to a parent process.

## Features

- **Simple logging API** with multiple log levels (Trace, Debug, Info, Warn, Error, Fatal)
- **Progress bars** with automatic console rendering
- **Inter-process logging** via TCP sockets
- **Zero external dependencies** (except workspace dependencies)

## Basic Usage

```rust
use dlog::{log, LogLevel};

log(LogLevel::Info, "Hello, world!");
log(LogLevel::Error, "Something went wrong");
```

## Progress Bars

```rust
use dlog::ProgressBar;

let pb = ProgressBar::new("Processing", 100);
for i in 0..=100 {
    pb.push(i);
    // do work
}
// Progress bar automatically finishes when dropped
```

## Inter-Process Logging

Parent process:
```rust
use dlog::{log, LogLevel};
use std::process::Command;

// Start logging
log(LogLevel::Info, "Parent starting");

// Wait for log thread to initialize
std::thread::sleep(std::time::Duration::from_millis(100));

// Get the pipe address (set automatically by dlog)
let pipe_addr = std::env::var("DLOG_PIPE").unwrap();

// Spawn child with DLOG_PIPE environment variable
Command::new("./my_child")
    .env("DLOG_PIPE", &pipe_addr)
    .spawn()
    .unwrap();
```

Child process:
```rust
use dlog::{log, LogLevel};

// These logs will automatically be sent to the parent
// if DLOG_PIPE is set
log(LogLevel::Info, "Child process logging");
```

## How IPC Works

1. When the first log is created in a process, a log thread starts
2. The log thread creates a TCP listener on `127.0.0.1:<random_port>`
3. The address is stored in the `DLOG_PIPE` environment variable
4. Child processes check for `DLOG_PIPE` and connect to the parent's listener
5. All log events from child processes are forwarded to the parent for output

## Environment Variables

- `DLOG_LEVEL`: Minimum log level to display (default: "trace")
- `DLOG_PIPE`: TCP address for IPC logging (set automatically by parent process)
- `DLOG_NO_PROGRESS`: Disable progress bars (default: "false")
- `DLOG_SIMPLE`: Use simple log writer that only outputs start/end of progress bars and all logs, without interactive progress updates (default: "false")

## Examples

Run the examples to see IPC logging in action:

```bash
cargo run --example ipc_test
cargo run --example progress_ipc
```

Run with simple log writer (no interactive progress bars):

```bash
DLOG_SIMPLE=true cargo run --example simple_test
```
