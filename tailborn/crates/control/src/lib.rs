mod entity;
mod proto;
use std::net::{IpAddr, SocketAddr};

use anyhow::Context;
use entity::Entity;
use proto::{IdentificationDigit, Request, Response};

use log::{error, info};
use tokio::io::AsyncWriteExt;


pub async fn run_control_server() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .expect("Failed to bind to address");
    info!("Control server started at '127.0.0.1:8080'");

    loop {
        match listener.accept().await {
            Ok((stream, s_addr)) => {
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, s_addr).await {
                        error!("Error handling connection: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Error accepting connection: {}", e);
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Invalid identification digit")]
pub struct InvalidDigit;

impl TryFrom<u8> for IdentificationDigit {
    type Error = InvalidDigit;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(IdentificationDigit::ControlServer),
            2 => Ok(IdentificationDigit::ComputeNode),
            4 => Ok(IdentificationDigit::DfsNode),
            8 => Ok(IdentificationDigit::Client),
            16 => Ok(IdentificationDigit::Relay),
            _ => Err(InvalidDigit),
        }
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    s_addr: SocketAddr,
) -> anyhow::Result<()> {
    loop {
        let msg = match crate::proto::Request::read_from_stream_async(&mut stream).await {
            Ok(msg) => msg,
            Err(err) => {
                use tokio::io::ErrorKind as EK;
                return match err.kind() {
                    EK::UnexpectedEof => Ok(()),
                    EK::ConnectionReset => Ok(()),
                    _ => Err(err).context("Failed to read message"),
                };
            }
        };
        handle_message(&mut stream, msg, s_addr.ip())
            .await
            .context("Failed to handle received message")?;
    }
}

async fn handle_message(
    _stream: &mut tokio::net::TcpStream,
    msg: crate::proto::Request,
    addr: IpAddr,
) -> anyhow::Result<()> {
    match msg {
        proto::Request::Authorize { identity, key } => {
            println!("{:?}", Entity::new(identity, addr, vec![]));
            todo!()
        }
        proto::Request::RetransmissionData(vec) => todo!(),
    }
}

pub struct Credentials {
    identity: IdentificationDigit,
    key: [u8; 32],
}

impl Credentials {
    pub fn root() -> Self {
        return Self {
            identity: IdentificationDigit::Client,
            key: [0; 32],
        };
    }
}

pub async fn connect(
    address: &str,
    credentials: Credentials,
) -> anyhow::Result<tokio::net::TcpStream> {
    // Connect to the server
    let mut stream = tokio::net::TcpStream::connect(&address)
        .await
        .context(format!("Failed to connect to {}", address))?;
    let auth_message = crate::proto::Request::Authorize {
        identity: credentials.identity,
        key: credentials.key,
    };
    stream.write_all(&auth_message.bytes()).await?;
    Ok(stream)
}
