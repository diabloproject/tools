mod backing_store;
mod config;
mod controller;
mod types;

use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, transport::Server};

use engine_proto::engine::data_service_server::{DataService, DataServiceServer};
use engine_proto::engine::*;

use crate::backing_store::Frame;
use crate::backing_store::ram::RamBackingStore;
use crate::backing_store::disk::DiskBackingStore;
use crate::config::get_config;
use crate::controller::primitive::PrimitiveStoreController;
use crate::controller::optimized::OptimizedStoreController;
use crate::types::PixelInfo;
use tracing::{Level, debug, error, info, warn, instrument, event};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PixelLoc {
    x: i64,
    y: i64,
}

impl std::fmt::Display for PixelLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<X: Into<i64>, Y: Into<i64>> From<(X, Y)> for PixelLoc {
    fn from(loc: (X, Y)) -> Self {
        Self {
            x: loc.0.into(),
            y: loc.1.into(),
        }
    }
}

struct DataNode {
    controller: Arc<dyn controller::StoreController>,
    backing: Arc<dyn backing_store::BackingStore>,
}

impl DataNode {
    fn new() -> Self {
        let config = get_config();
        let frame = Frame {
            width: config.tiling.size.0,
            height: config.tiling.size.1,
            center_x: -(config.tiling.size.0 as i64 * config.tiling.pos.0),
            center_y: -(config.tiling.size.1 as i64 * config.tiling.pos.1),
        };
        
        info!("Initializing data node");
        info!("Canvas size: {}x{}, position: ({}, {})", 
              config.tiling.size.0, config.tiling.size.1,
              config.tiling.pos.0, config.tiling.pos.1);
        info!("Frame offset: x={}, y={}", frame.center_x, frame.center_y);
        info!("Backing store: {}", config.storage.backing_store);
        info!("Controller: {}", config.storage.controller);
        
        let start = Instant::now();
        
        let backing: Arc<dyn backing_store::BackingStore> = match config.storage.backing_store.as_str() {
            "disk" => {
                info!("Initializing disk backing store at: {}", config.storage.data_dir);
                let store = Arc::new(
                    DiskBackingStore::new(
                        frame,
                        &config.storage.data_dir,
                        |x, y| PixelInfo {
                            x,
                            y,
                            color: 255,
                            timestamp: 0,
                            user_id: 0,
                            generation: 0,
                        },
                    )
                    .expect("Failed to initialize disk backing store"),
                );
                info!("Disk backing store initialized in {:?}", start.elapsed());
                store
            }
            "ram" => {
                info!("Initializing RAM backing store");
                let store = Arc::new(RamBackingStore::new(
                    frame,
                    |x, y| PixelInfo {
                        x,
                        y,
                        color: 255,
                        timestamp: 0,
                        user_id: 0,
                        generation: 0,
                    },
                ));
                info!("RAM backing store initialized in {:?}", start.elapsed());
                store
            }
            _ => panic!("Unknown backing store type: '{}'. Valid options: 'ram', 'disk'", config.storage.backing_store),
        };
        
        let controller_start = Instant::now();
        let controller: Arc<dyn controller::StoreController> = match config.storage.controller.as_str() {
            "primitive" => {
                info!("Initializing primitive controller");
                let ctrl = Arc::new(PrimitiveStoreController::new(backing.clone()));
                info!("Primitive controller initialized in {:?}", controller_start.elapsed());
                ctrl
            }
            "optimized" => {
                info!("Initializing optimized controller with snapshot caching");
                let ctrl = Arc::new(OptimizedStoreController::new(backing.clone()));
                info!("Optimized controller initialized in {:?}", controller_start.elapsed());
                ctrl
            }
            _ => panic!("Unknown controller type: '{}'. Valid options: 'primitive', 'optimized'", config.storage.controller),
        };
        
        info!("Data node fully initialized in {:?}", start.elapsed());
        
        Self {
            controller,
            backing,
        }
    }
}

#[tonic::async_trait]
impl DataService for DataNode {
    #[instrument(skip(self, request), fields(x, y, color, user_id))]
    async fn push_pixel(
        &self,
        request: Request<PushPixelRequest>,
    ) -> Result<Response<PushPixelResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        
        tracing::Span::current().record("x", req.x);
        tracing::Span::current().record("y", req.y);
        tracing::Span::current().record("color", req.color);
        tracing::Span::current().record("user_id", req.user_id);
        
        let time = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        
        self.controller.push_pixel(
            req.x,
            req.y,
            req.color as u8,
            req.user_id,
            time.as_millis() as u64,
        );
        
        let elapsed = start.elapsed();
        event!(Level::DEBUG, "push_pixel completed in {:?}", elapsed);
        
        if elapsed > Duration::from_millis(10) {
            warn!("Slow push_pixel operation: {:?} for pixel ({}, {})", elapsed, req.x, req.y);
        }
        
        Ok(Response::new(PushPixelResponse {}))
    }

    #[instrument(skip(self, request), fields(x, y))]
    async fn pixel_info(
        &self,
        request: Request<PixelInfoRequest>,
    ) -> Result<Response<PixelInfoResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        
        tracing::Span::current().record("x", req.x);
        tracing::Span::current().record("y", req.y);
        
        let Some(pi) = self.controller.pixel_info_at(req.x, req.y, 0) else {
            return Err(Status::new(
                tonic::Code::NotFound,
                format!("Pixel ({}, {}) not found", req.x, req.y),
            ));
        };
        
        let elapsed = start.elapsed();
        event!(Level::DEBUG, "pixel_info completed in {:?}", elapsed);
        
        if elapsed > Duration::from_millis(5) {
            warn!("Slow pixel_info operation: {:?} for pixel ({}, {})", elapsed, req.x, req.y);
        }
        
        Ok(Response::new(PixelInfoResponse {
            x: pi.x,
            y: pi.y,
            color: pi.color as u32,
            user_id: pi.user_id,
            generation: pi.generation,
        }))
    }

    #[instrument(skip(self, request), fields(timestamp))]
    async fn snapshot(
        &self,
        request: Request<SnapshotRequest>,
    ) -> Result<Response<SnapshotResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        
        tracing::Span::current().record("timestamp", req.timestamp);
        
        let infos = self.controller.snapshot(req.timestamp);
        let frame = self.backing.describe().frame;
        let colors = infos.iter().map(|pi| pi.color as u32).collect::<Vec<_>>();
        
        let elapsed = start.elapsed();
        info!("snapshot completed in {:?} ({} pixels)", elapsed, infos.len());
        
        if elapsed > Duration::from_millis(100) {
            warn!("Slow snapshot operation: {:?} for {} pixels at timestamp {}", 
                  elapsed, infos.len(), req.timestamp);
        }
        
        Ok(Response::new(SnapshotResponse {
            height: frame.height,
            width: frame.width,
            center_x: frame.center_x,
            center_y: frame.center_y,
            pixel_values: colors,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse()?;
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pixel_server=debug,tower_http=debug,tonic=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let server = DataNode::new();
    info!("Starting server on {}", addr);

    Server::builder()
        .add_service(DataServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
