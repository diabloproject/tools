use engine_proto::engine::data_service_client::DataServiceClient;
use engine_proto::engine::logic_service_server::{LogicService, LogicServiceServer};
use engine_proto::engine::{EventType, PushPixelRequest};
use engine_proto::engine::{TriggerEventRequest, TriggerEventResponse};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};
use tracing::{Level, debug, error, event, info, instrument, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

struct LogicNode {
    // data_client: Arc<Mutex<DataServiceClient<Channel>>>,
}

impl LogicNode {
    async fn new() -> Self {
        Self {
            // data_client: Arc::new(Mutex::new(
            //     DataServiceClient::connect("http://127.0.0.1:50051".to_string())
            //         .await
            //         .unwrap(),
            // )),
        }
    }
}

#[tonic::async_trait]
impl LogicService for LogicNode {
    async fn trigger_event(
        &self,
        request: Request<TriggerEventRequest>,
    ) -> Result<Response<TriggerEventResponse>, Status> {
        info!("Request: {:?}", request);
        let request = request.into_inner();
        let mut data_client = DataServiceClient::connect("http://localhost:50051")
            .await
            .expect("Failed to connect");
        match request.ty {
            1 => {
                let uid = request.user_id;
                let TriggerEventRequest { x, y, color, .. } = request;
                data_client
                    .push_pixel(PushPixelRequest { x, y, color, user_id: uid })
                    .await?;
                info!("Pixel data pushed successfully");
                return Ok(Response::new(TriggerEventResponse {}));
            }
            _ => warn!("Trigger event not found"),
        }
        Err(Status::unimplemented("Not yet implemented"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50052".parse()?;
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pixel_server=debug,tower_http=debug,tonic=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    info!("Trying to connect to data node");
    let _ = DataServiceClient::connect("http://127.0.0.1:50051".to_string()).await.expect("Failed to connect to data node");
    info!("Connected to data node");
    let server = LogicNode::new().await;
    info!("Starting server on {}", addr);

    Server::builder()
        .add_service(LogicServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
