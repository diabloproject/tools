use diabloproject::log::{LogLevel, ProgressBar, log};
use git2::build::RepoBuilder;
use git2::{FetchOptions, Progress, RemoteCallbacks};
use std::path::Path;
use std::sync::{Arc, Mutex};

const REPO_URL: &str = "https://github.com/diabloproject/tools";

use std::time::{SystemTime, UNIX_EPOCH};

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

    let progress_bar: Arc<Mutex<Option<ProgressBar>>> = Arc::new(Mutex::new(None));
    let progress_bar_clone = progress_bar.clone();
    let mut cb = RemoteCallbacks::new();

    cb.transfer_progress(move |stats: Progress| {
        let total = stats.total_objects();
        let received = stats.received_objects();
        let indexed = stats.indexed_objects();
        let bytes = stats.received_bytes();

        log(LogLevel::Trace, format!("Total objects: {}, Received: {}, Indexed: {}, Bytes: {}", total, received, indexed, bytes));

        if total > 0 {
            let mut pb = progress_bar_clone.lock().unwrap();
            if pb.is_none() {
                *pb = Some(ProgressBar::new("Receiving objects", total as u64));
                log(LogLevel::Trace, "Created progress bar for git clone operation");
            }

            if let Some(ref bar) = *pb {
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
        Ok(_) => log(LogLevel::Info, format!("Successfully cloned to: {:?}", target)),
        Err(e) => log(LogLevel::Error, format!("Failed to clone repository: {}", e)),
    }
}