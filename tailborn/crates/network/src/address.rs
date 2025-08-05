use serde::Deserialize;
use serde::Serialize;
use serde::Serializer;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TailbornAddress {
    // Namespace
    ns: u16,
    // Operation
    op: u16,
    // Operation instance
    opi: u32,
}

impl Default for TailbornAddress {
    fn default() -> Self {
        Self {
            ns: 0,
            op: 0,
            opi: 0,
        }
    }
}

impl std::fmt::Debug for TailbornAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The format is FF:FF:FFFF for ns:op:opi
        write!(f, "{:04X}:{:04X}:{:08X}", self.ns, self.op, self.opi)
    }
}

impl std::fmt::Display for TailbornAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The format is FF:FF:FFFF for ns:op:opi
        write!(f, "{:04X}:{:04X}:{:08X}", self.ns, self.op, self.opi)
    }
}

impl From<u64> for TailbornAddress {
    fn from(value: u64) -> Self {
        Self {
            ns: (value >> 48) as u16,
            op: ((value >> 32) & 0xFFFF) as u16,
            opi: (value & 0xFFFFFFFF) as u32,
        }
    }
}

impl From<TailbornAddress> for u64 {
    fn from(value: TailbornAddress) -> Self {
        ((value.ns as u64) << 48) | ((value.op as u64) << 32) | (value.opi as u64)
    }
}

impl Serialize for TailbornAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(u64::from(*self))
    }
}

impl<'a> Deserialize<'a> for TailbornAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(Self::from(value))
    }
}

impl From<[u8; 8]> for TailbornAddress {
    fn from(value: [u8; 8]) -> Self {
        Self {
            ns: u16::from_be_bytes([value[0], value[1]]),
            op: u16::from_be_bytes([value[2], value[3]]),
            opi: u32::from_be_bytes([value[4], value[5], value[6], value[7]]),
        }
    }
}

impl From<TailbornAddress> for [u8; 8] {
    fn from(value: TailbornAddress) -> Self {
        [
            value.ns.to_be_bytes()[0],
            value.ns.to_be_bytes()[1],
            value.op.to_be_bytes()[0],
            value.op.to_be_bytes()[1],
            value.opi.to_be_bytes()[0],
            value.opi.to_be_bytes()[1],
            value.opi.to_be_bytes()[2],
            value.opi.to_be_bytes()[3],
        ]
    }
}
