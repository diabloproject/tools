#![feature(trim_prefix_suffix)]

use replay::{Replay, ReplayRow};
use reqwest::blocking::Client;
use std::io::Read;
use std::sync::Arc;
use stdd::sync::worker::Worker;

fn main() {
    let server_uri = std::env::var("SPECTRE_API_URI")
        .unwrap_or_else(|_| "https://diabloproject.space/spectre/api".to_string());
    let client = Client::new();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let executable = args.first().unwrap().clone();
    let response = client
        .post(format!("{}/push", server_uri))
        .body(
            Replay {
                rows: vec![],
                client: "spectre-cli".to_string(),
            }
            .to_yson(),
        )
        .send()
        .expect("Failed to push replay");
    let id = response.text().expect("Failed to get replay id");
    println!("https://diabloproject.space/spectre/replay/{}", id);
    let id_ref = Arc::new(id);
    let worker = Worker::new(move |value| logging_worker(id_ref.as_ref().clone(), value));
    let mut child = std::process::Command::new(executable)
        .args(&args[1..])
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn child process");
    let mut leftovers: Vec<u8> = vec![];
    let mut buf: [u8; 1] = [0; 1];
    let mut stdout = child.stdout.take().expect("Failed to get stdout");
    while stdout.read_exact(&mut buf).is_ok() {
        leftovers.extend_from_slice(&buf);
        while let Some(idx) = leftovers.iter().position(|&b| b == b'\n') {
            let temp = leftovers.split_off(idx);
            let line = leftovers;
            leftovers = temp[1..].to_vec();
            let row = ReplayRow {
                timestamp: stdd::time::DateTime::now().to_millis() as u64,
                log: String::from_utf8_lossy(&tui_elements::strip_ansi_escape::strip(&line))
                    .to_string(),
            };
            worker.send(row).expect("Failed to send replay row");
        }
    }
    child.wait().expect("Failed to wait for child process");
    drop(worker);
}

fn logging_worker(id: String, row: ReplayRow) {
    let server_uri =
        std::env::var("SPECTRE_API_URI").unwrap_or_else(|_| "http://localhost:3000/".to_string());
    let client = Client::new();
    let response = client
        .post(format!("{}/append/{}", server_uri.trim_suffix('/'), &id))
        .body(
            Replay {
                rows: vec![row],
                client: "spectre-cli".to_string(),
            }
            .to_yson(),
        )
        .send()
        .expect("Failed to push replay");
}
