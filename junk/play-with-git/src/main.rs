use diabloproject::log::{LogLevel, log};
use git2::build::RepoBuilder;
use git2::{Cred, FetchOptions, Progress, RemoteCallbacks, Repository};
use std::path::Path;

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

    // Repository::clone(REPO_URL, &target).unwrap();

    let mut cb = RemoteCallbacks::new();

    cb.transfer_progress(|stats: Progress| {
        let received = stats.received_objects();
        let total = stats.total_objects();
        let indexed = stats.indexed_objects();

        log(
            LogLevel::Info,
            format!(
                "Received {}/{} objects ({} indexed)",
                received, total, indexed
            ),
        );

        if stats.received_objects() == stats.total_objects() {
            log(
                LogLevel::Info,
                format!(
                    "Resolving deltas {}/{}",
                    stats.indexed_deltas(),
                    stats.total_deltas()
                ),
            );
        }

        true // return false to abort
    });

    let mut fo = FetchOptions::new();
    fo.remote_callbacks(cb);

    let mut builder = RepoBuilder::new();
    builder.fetch_options(fo);

    builder.clone(REPO_URL, &target).unwrap();

    log(LogLevel::Info, format!("Cloned to: {:?}", target));
}
