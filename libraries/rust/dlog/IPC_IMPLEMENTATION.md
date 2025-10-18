# dlog IPC Implementation

## Overview
The dlog crate now supports inter-process communication (IPC) for logging, allowing child processes to send their logs to a parent process for centralized output.

## How It Works

### Parent Process
1. When the first log is created, the log thread starts
2. The log thread creates a TCP listener on `127.0.0.1:<random_port>`
3. The address is stored in the `DLOG_PIPE` environment variable
4. The log thread listens for both:
   - Local log events from the same process (via mpsc channel)
   - Remote log events from child processes (via TCP socket)
5. All logs are written to the console by the parent's log thread

### Child Process
1. When a child process starts logging:
   - It checks for the `DLOG_PIPE` environment variable
   - If present, it connects to the parent's TCP listener
   - All log events are serialized and sent via the socket
2. If `DLOG_PIPE` is not set:
   - The process behaves as normal (local logging)

## Usage

```rust
use dlog::{log, LogLevel};
use std::process::Command;

fn parent() {
    // Start logging
    log(LogLevel::Info, "Parent starting");
    
    // Wait for log thread to initialize
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    // Get the pipe address
    let pipe_addr = std::env::var("DLOG_PIPE").unwrap();
    
    // Spawn child with DLOG_PIPE env var
    Command::new("./child")
        .env("DLOG_PIPE", &pipe_addr)
        .spawn()
        .unwrap();
}

fn child() {
    // Child logs will automatically go to parent if DLOG_PIPE is set
    log(LogLevel::Info, "Child logging");
}
```

## Implementation Details

### Serialization Format
Log events are serialized to a custom binary format without external dependencies:

**Format: `[length: u32][payload]`**

Where payload depends on event type:
- **Log**: `[0][level: u8][msg_len: u32][message][ts_len: u32][timestamp_str]`
- **CreateProgressBar**: `[1][id: u64][desc_len: u32][description][total: u64]`
- **PushProgress**: `[2][id: u64][value: u64]`
- **FinishProgressBar**: `[3][id: u64]`

### Key Components

1. **pipe.rs**: TCP listener and serialization logic
2. **lib.rs**: 
   - `LogSink` enum: Either local channel or remote socket
   - `log_thread()`: Multiplexes local and remote log events
   - `LOG_SOCKET` static: Automatically chooses local or remote mode

### Thread Safety
- The TCP stream is wrapped in a `Mutex` for safe concurrent access
- Each child connection gets its own thread
- The parent's log thread uses `try_recv()` to poll both channels

## Testing
See `examples/ipc_test.rs` for a complete working example that demonstrates parent-child IPC logging.
