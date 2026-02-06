mod config;
mod connection;
mod plugin;
mod plugins;

use anyhow::Result;
use config::Config;
use connection::ProxyConnection;
use plugin::PluginManager;
use plugins::{ChatFilter, PacketLogger};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.yaml".to_string());

    let config = match Config::load(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            info!("Could not load config from {}: {}. Using defaults.", config_path, e);
            Config::default()
        }
    };

    let mut plugin_manager = PluginManager::new();
    
    // Add example plugins - all work with Box<dyn Trait> for future dylib support
    plugin_manager.add_filter(Box::new(PacketLogger::new(true, true)));
    plugin_manager.add_filter(Box::new(ChatFilter::new(vec!["badword".to_string()])));

    let plugin_manager = Arc::new(plugin_manager);

    let bind_addr = format!("{}:{}", config.proxy.bind_address, config.proxy.bind_port);
    let upstream_addr = format!("{}:{}", config.upstream.address, config.upstream.port);

    info!("Starting Minecraft proxy server");
    info!("Listening on: {}", bind_addr);
    info!("Upstream server: {}", upstream_addr);

    let listener = TcpListener::bind(&bind_addr).await?;

    loop {
        match listener.accept().await {
            Ok((client_socket, client_addr)) => {
                info!("New client connection from: {}", client_addr);
                
                let upstream_addr = upstream_addr.clone();
                let plugin_manager = Arc::clone(&plugin_manager);

                tokio::spawn(async move {
                    match ProxyConnection::new(client_socket, &upstream_addr, plugin_manager).await {
                        Ok(proxy) => {
                            if let Err(e) = proxy.handle().await {
                                error!("Proxy connection error: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Failed to establish upstream connection: {}", e);
                        }
                    }
                    info!("Client {} disconnected", client_addr);
                });
            }
            Err(e) => {
                error!("Failed to accept connection: {}", e);
            }
        }
    }
}
