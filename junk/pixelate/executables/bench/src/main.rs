use engine_proto::engine::data_service_client::DataServiceClient;
use engine_proto::engine::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tonic::transport::Channel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pixelate Data Node Benchmark ===\n");

    // Connect to server
    println!("Connecting to server at http://127.0.0.1:50051...");
    let client = DataServiceClient::connect("http://127.0.0.1:50051").await?;
    println!("✓ Connected!\n");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let test_type = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let duration_secs = args
        .get(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10);
    let concurrency = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    println!("Test configuration:");
    println!("  Test type: {}", test_type);
    println!("  Duration: {}s", duration_secs);
    println!("  Concurrency: {}", concurrency);
    println!();

    match test_type {
        "push" => bench_push_pixel(client, duration_secs, concurrency).await?,
        "read" => bench_pixel_info(client, duration_secs, concurrency).await?,
        "snapshot" => bench_snapshot(client, duration_secs).await?,
        "all" => {
            bench_push_pixel(client.clone(), duration_secs, concurrency).await?;
            println!();
            bench_pixel_info(client.clone(), duration_secs, concurrency).await?;
            println!();
            bench_snapshot(client, duration_secs).await?;
        }
        _ => {
            println!("Unknown test type: {}", test_type);
            println!("Usage: pixelate-bench [push|read|snapshot|all] [duration_secs] [concurrency]");
            println!("Example: pixelate-bench push 10 50");
        }
    }

    Ok(())
}

async fn bench_push_pixel(
    _client: DataServiceClient<Channel>,
    duration_secs: u64,
    concurrency: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Push Pixel Benchmark ---");
    println!("Testing write performance with {} concurrent workers", concurrency);
    
    // Create a connection pool - reasonable number of connections
    let pool_size = concurrency.min(20);
    println!("Using connection pool with {} connections", pool_size);
    
    let mut clients = Vec::new();
    for i in 0..pool_size {
        match DataServiceClient::connect("http://127.0.0.1:50051").await {
            Ok(c) => clients.push(Arc::new(tokio::sync::Mutex::new(c))),
            Err(e) => {
                eprintln!("Failed to create connection {}: {}", i, e);
                return Err(Box::new(e));
            }
        }
    }
    let clients = Arc::new(clients);

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let duration = Duration::from_secs(duration_secs);
    let start = Instant::now();

    let mut total_requests = 0u64;
    let mut tasks = Vec::new();

    // Spawn workers - they share the connection pool
    for worker_id in 0..concurrency {
        let clients = clients.clone();
        let semaphore = semaphore.clone();
        let worker_duration = duration;
        let worker_start = start;

        let task = tokio::spawn(async move {
            let mut worker_requests = 0u64;
            let mut rng = StdRng::seed_from_u64(worker_id as u64);
            
            // Pick a client from the pool (round-robin)
            let client_idx = worker_id % clients.len();
            let client = clients[client_idx].clone();

            while worker_start.elapsed() < worker_duration {
                let _permit = semaphore.acquire().await.unwrap();

                let x = rng.gen_range(0..512);
                let y = rng.gen_range(0..512);
                let color = rng.gen_range(0..256);

                let request = PushPixelRequest {
                    x,
                    y,
                    color,
                    user_id: worker_id as u64,
                };

                let mut c = client.lock().await;
                if c.push_pixel(request).await.is_ok() {
                    worker_requests += 1;
                }
            }

            worker_requests
        });

        tasks.push(task);
    }

    // Collect results
    for task in tasks {
        total_requests += task.await?;
    }

    let elapsed = start.elapsed();
    let rps = total_requests as f64 / elapsed.as_secs_f64();

    println!("\nResults:");
    println!("  Total requests: {}", total_requests);
    println!("  Duration: {:.2}s", elapsed.as_secs_f64());
    println!("  RPS: {:.2}", rps);
    println!("  Avg latency: {:.2}ms", (elapsed.as_secs_f64() * 1000.0) / total_requests as f64);

    Ok(())
}

async fn bench_pixel_info(
    _client: DataServiceClient<Channel>,
    duration_secs: u64,
    concurrency: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Pixel Info Benchmark ---");
    println!("Testing read performance with {} concurrent workers", concurrency);
    
    // Create a connection pool - reasonable number of connections
    let pool_size = concurrency.min(20);
    println!("Using connection pool with {} connections", pool_size);
    
    let mut clients = Vec::new();
    for i in 0..pool_size {
        match DataServiceClient::connect("http://127.0.0.1:50051").await {
            Ok(c) => clients.push(Arc::new(tokio::sync::Mutex::new(c))),
            Err(e) => {
                eprintln!("Failed to create connection {}: {}", i, e);
                return Err(Box::new(e));
            }
        }
    }
    let clients = Arc::new(clients);

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let duration = Duration::from_secs(duration_secs);
    let start = Instant::now();

    let mut total_requests = 0u64;
    let mut tasks = Vec::new();

    // Spawn workers - they share the connection pool
    for worker_id in 0..concurrency {
        let clients = clients.clone();
        let semaphore = semaphore.clone();
        let worker_duration = duration;
        let worker_start = start;

        let task = tokio::spawn(async move {
            let mut worker_requests = 0u64;
            let mut rng = StdRng::seed_from_u64(worker_id as u64);
            
            // Pick a client from the pool (round-robin)
            let client_idx = worker_id % clients.len();
            let client = clients[client_idx].clone();

            while worker_start.elapsed() < worker_duration {
                let _permit = semaphore.acquire().await.unwrap();

                let x = rng.gen_range(0..512);
                let y = rng.gen_range(0..512);

                let request = PixelInfoRequest { x, y };

                let mut c = client.lock().await;
                if c.pixel_info(request).await.is_ok() {
                    worker_requests += 1;
                }
            }

            worker_requests
        });

        tasks.push(task);
    }

    // Collect results
    for task in tasks {
        total_requests += task.await?;
    }

    let elapsed = start.elapsed();
    let rps = total_requests as f64 / elapsed.as_secs_f64();

    println!("\nResults:");
    println!("  Total requests: {}", total_requests);
    println!("  Duration: {:.2}s", elapsed.as_secs_f64());
    println!("  RPS: {:.2}", rps);
    println!("  Avg latency: {:.2}ms", (elapsed.as_secs_f64() * 1000.0) / total_requests as f64);

    Ok(())
}

async fn bench_snapshot(
    client: DataServiceClient<Channel>,
    duration_secs: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Snapshot Benchmark ---");
    println!("Testing snapshot performance (sequential)");

    let mut client = client;
    let duration = Duration::from_secs(duration_secs);
    let start = Instant::now();

    let mut total_requests = 0u64;
    let mut total_pixels = 0u64;

    while start.elapsed() < duration {
        let request = SnapshotRequest { timestamp: 0 };

        if let Ok(response) = client.snapshot(request).await {
            let resp = response.into_inner();
            total_pixels = resp.pixel_values.len() as u64;
            total_requests += 1;
        }
    }

    let elapsed = start.elapsed();
    let rps = total_requests as f64 / elapsed.as_secs_f64();

    println!("\nResults:");
    println!("  Total requests: {}", total_requests);
    println!("  Pixels per snapshot: {}", total_pixels);
    println!("  Duration: {:.2}s", elapsed.as_secs_f64());
    println!("  RPS: {:.2}", rps);
    println!("  Avg latency: {:.2}ms", (elapsed.as_secs_f64() * 1000.0) / total_requests as f64);
    println!("  Throughput: {:.2} Mpixels/s", (total_requests * total_pixels) as f64 / elapsed.as_secs_f64() / 1_000_000.0);

    Ok(())
}
