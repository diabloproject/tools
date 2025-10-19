use diabloproject::log::{LogLevel, ProgressBar, finish, log};
use git2::build::RepoBuilder;
use git2::{FetchOptions, Progress, RemoteCallbacks};
use std::path::Path;
use std::process::Command;

const REPO_URL: &str = "https://github.com/diabloproject/tools";

use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn simple_rand(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (*seed / 65536) % 32768
}

fn random_alphanumeric_string(len: usize) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    (0..len)
        .map(|_| {
            let idx = simple_rand(&mut seed) as usize % CHARSET.len();
            CHARSET[idx] as char
        })
        .collect()
}

fn main() {
    let cache_dir: &Path = Path::new("/tmp/d-temp");
    let target = random_alphanumeric_string(16);
    let target = cache_dir.join(target);

    log(
        LogLevel::Info,
        format!("Cloning {} to {:?}", REPO_URL, target),
    );
    {
        let mut progress_bar: Option<ProgressBar> = None;
        let pbref = &mut progress_bar;
        let mut cb = RemoteCallbacks::new();

        cb.transfer_progress(move |stats: Progress| {
            let total = stats.total_objects();
            let received = stats.received_objects();
            let indexed = stats.indexed_objects();
            let bytes = stats.received_bytes();

            log(
                LogLevel::Trace,
                format!(
                    "Total objects: {}, Received: {}, Indexed: {}, Bytes: {}",
                    total, received, indexed, bytes
                ),
            );

            if total > 0 {
                if pbref.is_none() {
                    *pbref = Some(ProgressBar::new("Receiving objects", total as u64));
                    log(
                        LogLevel::Trace,
                        "Created progress bar for git clone operation",
                    );
                }

                if let Some(bar) = pbref {
                    bar.push(received as u64);
                }
            }

            true
        });

        let mut fo = FetchOptions::new();
        fo.remote_callbacks(cb);

        let mut builder = RepoBuilder::new();
        builder.fetch_options(fo);

        match builder.clone(REPO_URL, &target) {
            Ok(_) => log(
                LogLevel::Info,
                format!("Successfully cloned to: {:?}", target),
            ),
            Err(e) => log(
                LogLevel::Error,
                format!("Failed to clone repository: {}", e),
            ),
        }
    }
    std::thread::sleep(Duration::from_millis(100));
    // Run cargo run in `/tmp/d-temp/<repo>/d`
    let cargo_run = Command::new("cargo")
        .arg("run")
        .current_dir(target.join("d"))
        .output()
        .expect("Failed to run cargo run");
    log(
        LogLevel::Info,
        format!(
            "Cargo run output: {:?}",
            String::from_utf8_lossy(&cargo_run.stdout)
        ),
    );
    log(
        LogLevel::Info,
        format!(
            "Cargo run error: {:?}",
            String::from_utf8_lossy(&cargo_run.stderr)
        ),
    );
    log(LogLevel::Info, "Finished".to_string());
    finish()
}
