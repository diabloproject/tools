//! This proto file defines common routing capabilities and mappings for every network entity, regardless of the purpose.

use crate::address::TailbornAddress;

#[derive(Debug, thiserror::Error)]
pub enum PacketDecodeError {
    #[error("Unexpected end of stream")]
    UnexpectedEndOfPacket,
    #[error("Malformed packet")]
    MalformedPacket,
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

pub trait PayloadConvertable
where
    Self: Sized,
{
    fn from_payload(payload: Vec<u8>) -> Result<Self, PacketDecodeError>;
    fn to_payload(&self) -> Vec<u8>;
}

pub trait FromPayloadStream
where
    Self: Sized,
{
    fn from_payload_stream<R: std::io::Read>(reader: &mut R) -> Result<Self, PacketDecodeError>;
}

pub struct L1Packet<T: PayloadConvertable + Sized> {
    pub src: TailbornAddress,
    pub dst: TailbornAddress,
    pub payload: T,
}

impl<T: PayloadConvertable + Sized> PayloadConvertable for L1Packet<T> {
    fn from_payload(payload: Vec<u8>) -> Result<Self, PacketDecodeError> {
        if payload.len() < 20 {
            return Err(PacketDecodeError::UnexpectedEndOfPacket);
        }
        let src: [u8; 8] = payload[0..8].try_into().unwrap();
        let dst: [u8; 8] = payload[8..16].try_into().unwrap();
        let payload_size = u32::from_be_bytes(payload[16..20].try_into().unwrap());
        if payload_size > (payload.len() as u32 + 20) {
            return Err(PacketDecodeError::UnexpectedEndOfPacket);
        }
        let payload = T::from_payload(payload[20..payload_size as usize + 20].to_vec())?;
        Ok(L1Packet {
            src: src.into(),
            dst: dst.into(),
            payload,
        })
    }

    fn to_payload(&self) -> Vec<u8> {
        let src: [u8; 8] = self.src.into();
        let dst: [u8; 8] = self.dst.into();
        let payload_bytes = self.payload.to_payload();
        let mut payload = Vec::with_capacity(16 + self.payload.to_payload().len());

        payload.extend_from_slice(&src);
        payload.extend_from_slice(&dst);
        payload.extend_from_slice(&(payload_bytes.len() as u32).to_be_bytes());
        payload.extend_from_slice(&payload_bytes);
        payload
    }
}

impl<T: PayloadConvertable + Sized> FromPayloadStream for L1Packet<T> {
    fn from_payload_stream<R: std::io::Read>(reader: &mut R) -> Result<Self, PacketDecodeError> {
        let mut src = [0u8; 8];
        let mut dst = [0u8; 8];
        let mut payload_size = [0u8; 4];

        reader.read_exact(&mut src)?;
        reader.read_exact(&mut dst)?;
        reader.read_exact(&mut payload_size)?;

        let payload_size = u32::from_be_bytes(payload_size);
        let mut payload = vec![0u8; payload_size as usize];

        reader.read_exact(&mut payload)?;

        let payload = T::from_payload(payload)?;
        Ok(L1Packet {
            src: src.into(),
            dst: dst.into(),
            payload,
        })
    }
}
