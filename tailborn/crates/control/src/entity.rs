use std::net::IpAddr;

use crate::proto::IdentificationDigit;

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum EntityCapability {
    Basic = 1,
}

struct CapabilityIterator(u8, EntityCapability);

impl Iterator for CapabilityIterator {
    type Item = EntityCapability;

    fn next(&mut self) -> Option<Self::Item> {
        fn next_cap(cap: EntityCapability) -> Option<EntityCapability> {
            match cap {
                EntityCapability::Basic => None,
            }
        }

        loop {
            let current_cap = self.1;
            let has_capability = self.0 & (current_cap as u8) != 0;

            match next_cap(current_cap) {
                Some(next) => {
                    self.1 = next;
                    if has_capability {
                        return Some(current_cap);
                    }
                }
                None => {
                    return if has_capability {
                        Some(current_cap)
                    } else {
                        None
                    };
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Capabilities(u8);

impl Capabilities {
    fn new() -> Self {
        Self(0)
    }

    fn contains(&self, cap: EntityCapability) -> bool {
        self.0 & cap as u8 != 0
    }

    fn iter_set(&self) -> CapabilityIterator {
        return CapabilityIterator(self.0, EntityCapability::Basic);
    }
}

impl std::fmt::Display for Capabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for cap in self.iter_set() {
            if !first {
                f.write_str("+")?;
            }
            f.write_str(match cap {
                EntityCapability::Basic => "bs",
            })?;
            first = false;
        }
        if first {
            write!(f, "none")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Entity {
    entity_type: IdentificationDigit,
    addr: IpAddr,
    caps: Capabilities,
}

impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<{}@{}: {}>",
            match self.entity_type {
                IdentificationDigit::ControlServer => "control",
                IdentificationDigit::ComputeNode => "compute",
                IdentificationDigit::DfsNode => "dfs",
                IdentificationDigit::Client => "client",
                IdentificationDigit::Relay => "relay",
            },
            self.addr,
            self.caps
        )
    }
}

// impl std::fmt::Display for Entity {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         std::fmt::Debug::fmt(self, f)
//     }
// }

impl Entity {
    pub fn new(
        entity_type: IdentificationDigit,
        addr: IpAddr,
        caps: Vec<EntityCapability>,
    ) -> Self {
        let mut caps_bitset = Capabilities::new();
        for &cap in &caps {
            caps_bitset.0 |= cap as u8;
        }
        Self {
            entity_type,
            addr,
            caps: caps_bitset,
        }
    }
}
