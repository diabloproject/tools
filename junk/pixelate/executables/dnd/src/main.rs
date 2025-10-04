mod backing_store;
mod config;
mod controller;
mod types;

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, transport::Server};

use engine_proto::engine::data_service_server::{DataService, DataServiceServer};
use engine_proto::engine::*;

use crate::backing_store::Frame;
use crate::backing_store::ram::RamBackingStore;
use crate::config::get_config;
use crate::controller::primitive::PrimitiveStoreController;
use crate::types::PixelInfo;
use tracing::{Level, debug, error, info, warn};
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
        let backing = Arc::new(RamBackingStore::new(
            Frame {
                width: config.tiling.size.0,
                height: config.tiling.size.1,
                center_x: -(config.tiling.size.0 as i64 * config.tiling.pos.0),
                center_y: -(config.tiling.size.1 as i64 * config.tiling.pos.1),
            },
            |x, y| PixelInfo {
                x,
                y,
                color: 255,
                timestamp: 0,
                user_id: 0,
                generation: 0,
            },
        ));
        Self {
            controller: Arc::new(PrimitiveStoreController::new(backing.clone())),
            backing,
        }
    }
}

#[tonic::async_trait]
impl DataService for DataNode {
    async fn push_pixel(
        &self,
        request: Request<PushPixelRequest>,
    ) -> Result<Response<PushPixelResponse>, Status> {
        let req = request.into_inner();
        let time = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        self.controller.push_pixel(
            req.x,
            req.y,
            req.color as u8,
            req.user_id,
            time.as_millis() as u64, // There is no way it will overflow
        );
        Ok(Response::new(PushPixelResponse {}))
    }

    async fn pixel_info(
        &self,
        request: Request<PixelInfoRequest>,
    ) -> Result<Response<PixelInfoResponse>, Status> {
        let req = request.into_inner();
        let Some(pi) = self.controller.pixel_info_at(req.x, req.y, 0) else {
            return Err(Status::new(
                tonic::Code::NotFound,
                format!("Pixel ({}, {}) not found", req.x, req.y),
            ));
        };
        Ok(Response::new(PixelInfoResponse {
            x: pi.x,
            y: pi.y,
            color: pi.color as u32,
            user_id: pi.user_id,
            generation: pi.generation,
        }))
    }

    async fn snapshot(
        &self,
        request: Request<SnapshotRequest>,
    ) -> Result<Response<SnapshotResponse>, Status> {
        let req = request.into_inner();
        let infos = self.controller.snapshot(req.timestamp);
        let frame = self.backing.describe().frame;
        let colors = infos.iter().map(|pi| pi.color as u32).collect::<Vec<_>>();
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
