mod dfs;

use std::env;
use std::process::Stdio;
use anyhow::{anyhow, Context, Result};
use tokio::fs;
use tokio::io::{stdin, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use num_cpus;
use dfs::{DfsClient, InMemoryDfsClient};

#[tokio::main]
async fn main() -> Result<()> {
    // Collect command-line arguments to run.
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: tb <command> [args...]");
        std::process::exit(1);
    }

    // Initialize a DFS client (in-memory stub by default).
    let client = InMemoryDfsClient::new();

    // Pre-process @ (input) and & (output) file arguments.
    for arg in args.iter_mut() {
        if let Some(path) = arg.strip_prefix('@') {
            // Read local file and upload to DFS.
            let content = fs::read(path)
                .await
                .with_context(|| format!("reading file '{}'", path))?;
            let id = client.create().await?;
            client.store(id, content).await?;
            *arg = id.to_string();
        } else if arg.starts_with('&') {
            // Reserve an empty object for output.
            let id = client.create().await?;
            *arg = id.to_string();
        }
    }

    // Read tdh handles (one per line) from stdin.
    let stdin = BufReader::new(stdin());
    let mut lines = stdin.lines();
    let mut handles: Vec<String> = Vec::new();
    while let Some(line) = lines.next_line().await? {
        if !line.is_empty() {
            handles.push(line);
        }
    }
    if handles.is_empty() {
        return Ok(());
    }

    // Determine number of worker processes to spawn.
    let n_workers = num_cpus::get().max(1);
    let total = handles.len();
    let chunk_size = (total + n_workers - 1) / n_workers;

    // Spawn child processes, feeding each its chunk of handles.
    let mut tasks = Vec::new();
    for chunk in handles.chunks(chunk_size) {
        // Prepare command.
        let mut cmd = Command::new(&args[0]);
        if args.len() > 1 {
            cmd.args(&args[1..]);
        }
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        // Spawn.
        let mut child = cmd.spawn()
            .with_context(|| format!("failed to spawn process '{:?}'", &args))?;

        // Pipe our chunk into the child's stdin.
        if let Some(mut stdin_child) = child.stdin.take() {
            let input = chunk.join("\n") + "\n";
            tokio::spawn(async move {
                let _ = stdin_child.write_all(input.as_bytes()).await;
                let _ = stdin_child.shutdown().await;
            });
        }

        // Collect output from child.
        let task = tokio::spawn(async move {
            let output = child.wait_with_output()
                .await
                .with_context(|| "waiting for child process")?;
            if !output.status.success() {
                return Err(anyhow!("child exited with {}", output.status));
            }
            let s = String::from_utf8(output.stdout)
                .map_err(|e| anyhow!("invalid utf-8 from child: {}", e))?;
            Ok::<String, anyhow::Error>(s)
        });
        tasks.push(task);
    }

    // Print each worker's output in turn.
    for task in tasks {
        // task.await? unwraps the JoinHandle, then ? unwraps the child result
        let out = task.await??;
        print!("{}", out);
    }

    Ok(())
}
