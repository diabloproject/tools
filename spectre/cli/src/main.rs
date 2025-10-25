#![feature(trim_prefix_suffix)]

use replay::{Replay, ReplayRow};
use reqwest::blocking::Client;
use std::io::Read;
use std::sync::Arc;
use stdd::sync::worker::Worker;

#[derive(Default)]
struct Screen {
    buffer: Vec<String>, // each line
    cursor_row: usize,
    cursor_col: usize,
}


impl Screen {
    fn new() -> Self {
        Self {
            buffer: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
        }
    }

    fn ensure_row(&mut self, row: usize) {
        while self.buffer.len() <= row {
            self.buffer.push(String::new());
        }
    }

    fn print_char(&mut self, c: char) {
        self.ensure_row(self.cursor_row);
        let line = &mut self.buffer[self.cursor_row];
        if self.cursor_col >= line.len() {
            line.extend(std::iter::repeat(' ').take(self.cursor_col - line.len()));
            line.push(c);
        } else {
            line.remove(self.cursor_col);
            line.insert(self.cursor_col, c);
        }
        self.cursor_col += 1;
    }

    fn newline(&mut self) {
        self.cursor_row += 1;
        self.cursor_col = 0;
        self.ensure_row(self.cursor_row);
    }

    fn carriage_return(&mut self) {
        self.cursor_col = 0;
    }

    fn clear_line(&mut self) {
        self.ensure_row(self.cursor_row);
        self.buffer[self.cursor_row].clear();
        self.cursor_col = 0;
    }

    fn move_cursor_up(&mut self, n: usize) {
        if n > self.cursor_row {
            self.cursor_row = 0;
        } else {
            self.cursor_row -= n;
        }
        self.ensure_row(self.cursor_row);
    }

    fn move_cursor_down(&mut self, n: usize) {
        self.cursor_row += n;
        self.ensure_row(self.cursor_row);
    }

    fn move_cursor_right(&mut self, n: usize) {
        self.cursor_col += n;
    }

    fn move_cursor_left(&mut self, n: usize) {
        self.cursor_col = self.cursor_col.saturating_sub(n);
    }

    fn final_output(&self) -> String {
        self.buffer.join("\n")
    }
}

impl vte::Perform for Screen {
    fn print(&mut self, c: char) {
        self.print_char(c);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            b'\r' => self.carriage_return(),
            b'\n' => self.newline(),
            _ => {}
        }
    }

    // ignore control sequences
    fn hook(&mut self, _: &vte::Params, _: &[u8], _: bool, _: char) {}
    fn put(&mut self, _: u8) {}
    fn unhook(&mut self) {}
    fn osc_dispatch(&mut self, _: &[&[u8]], _: bool) {}
    fn csi_dispatch(
        &mut self,
        params: &vte::Params,
        _intermediates: &[u8],
        _ignore: bool,
        action: char,
    ) {
        let params = params.iter().map(|p| p).collect::<Vec<_>>();
        let n = params.iter()
            .next()
            .and_then(|param_group| param_group.first())
            .copied()
            .unwrap_or(1);
        match action {
            'A' => self.move_cursor_up(n as usize),
            'B' => self.move_cursor_down(n as usize),
            'C' => self.move_cursor_right(n as usize),
            'D' => self.move_cursor_left(n as usize),
            'K' => self.clear_line(),
            _ => {}
        }
    }
}

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
                log: String::from_utf8_lossy(&tui_elements::strip_ansi_escape::strip(&line)).to_string(),
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
