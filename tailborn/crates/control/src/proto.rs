use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IdentificationDigit {
    ControlServer = 1,
    ComputeNode = 2,
    DfsNode = 4,
    Client = 8,
    Relay = 16,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Request {
    Authorize {
        /// What the thing connecting to us is
        identity: IdentificationDigit,
        /// 256-bit key, that we can use to verify if the thing connecting to us is authorized to connect
        key: [u8; 32],
    } = 0,
    RetransmissionData(Vec<u8>) = 1,
}

impl Request {
    pub async fn read_from_stream_async(
        mut async_stream: impl AsyncReadExt + Unpin,
    ) -> tokio::io::Result<Request> {
        let message_type = async_stream.read_u8().await?;

        Ok(match message_type {
            1 => {
                let identity_byte = async_stream.read_u8().await?;
                let identity = match identity_byte {
                    1 => IdentificationDigit::ControlServer,
                    2 => IdentificationDigit::ComputeNode,
                    4 => IdentificationDigit::DfsNode,
                    8 => IdentificationDigit::Client,
                    16 => IdentificationDigit::Relay,
                    _ => panic!("Invalid identification digit: {}", identity_byte),
                };

                // Read the 32-byte key
                let mut key = [0u8; 32];
                async_stream.read_exact(&mut key).await?;

                Request::Authorize { identity, key }
            }
            2 => {
                let mut length_bytes = [0u8; 4];
                async_stream.read_exact(&mut length_bytes).await?;
                let length = u32::from_be_bytes(length_bytes) as usize;

                let mut data = vec![0u8; length];
                async_stream.read_exact(&mut data).await?;

                Request::RetransmissionData(data)
            }
            _ => panic!("Unknown message type: {}", message_type),
        })
    }

    pub fn bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        match self {
            Request::Authorize { identity, key } => {
                output.push(1); // message type for Authorize
                output.push(*identity as u8);
                output.extend_from_slice(key);
            }
            Request::RetransmissionData(data) => {
                output.push(2); // message type for RetransmissionData
                output.extend_from_slice(&(data.len() as u32).to_be_bytes());
                output.extend_from_slice(data);
            }
        }
        output
    }
}


pub enum Response {

}
