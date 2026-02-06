use std::io::Cursor;
use anyhow::Result;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::plugin::PluginManager;

pub struct ProxyConnection {
    client: TcpStream,
    upstream: TcpStream,
    plugin_manager: Arc<PluginManager>,
}

impl ProxyConnection {
    pub async fn new(
        client: TcpStream,
        upstream_addr: &str,
        plugin_manager: Arc<PluginManager>,
    ) -> Result<Self> {
        let upstream = TcpStream::connect(upstream_addr).await?;
        
        Ok(Self {
            client,
            upstream,
            plugin_manager,
        })
    }

    pub async fn handle(self) -> Result<()> {
        let (mut client_read, mut client_write) = self.client.into_split();
        let (mut upstream_read, mut upstream_write) = self.upstream.into_split();

        let _plugin_manager_fwd = Arc::clone(&self.plugin_manager);
        let _plugin_manager_bwd = Arc::clone(&self.plugin_manager);
        let mut buf = Cursor::new(Vec::new());
        let mut encryption = None;
        let mut decryption = None;
        let forward = tokio::spawn(async move {
            loop {
                let res = azalea_protocol::read::read_packet(&mut client_read, &mut buf, None, &mut encryption).await;
                match res {
                    Ok(packet) => {
                        tracing::trace!("Packet received: {}", &packet);
                        let res = azalea_protocol::write::write_packet(
                            &packet,
                            &mut client_write,
                            None,
                            &mut decryption
                        ).await;
                        match res {
                            Ok(_) => {}
                            Err(error) => {
                                tracing::error!("{}", error);
                            }
                        }
                    }
                    Err (error) => {
                        tracing::error!("{}", error);
                    }
                }
            }


            let mut buffer = vec![0u8; 8192];
            loop {
                match client_read.read(&mut buffer).await {
                    Ok(0) => break,
                    Ok(n) => {
                        tracing::trace!("Read {} bytes from client", n);

                        // TODO: Parse packets here and process through plugin manager
                        // For now, pass through directly
                        if let Err(e) = upstream_write.write_all(&buffer[..n]).await {
                            tracing::error!("Failed to write to upstream: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read from client: {}", e);
                        break;
                    }
                }
            }
            tracing::info!("Forward connection closed");
        });

        let backward = tokio::spawn(async move {
            let mut buffer = vec![0u8; 8192];
            loop {
                match upstream_read.read(&mut buffer).await {
                    Ok(0) => break,
                    Ok(n) => {
                        tracing::trace!("Read {} bytes from upstream", n);
                        
                        // TODO: Parse packets here and process through plugin manager
                        // For now, pass through directly
                        if let Err(e) = client_write.write_all(&buffer[..n]).await {
                            tracing::error!("Failed to write to client: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read from upstream: {}", e);
                        break;
                    }
                }
            }
            tracing::info!("Backward connection closed");
        });

        tokio::select! {
            _ = forward => tracing::info!("Forward task completed"),
            _ = backward => tracing::info!("Backward task completed"),
        }

        Ok(())
    }
}
