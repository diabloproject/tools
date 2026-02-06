use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub proxy: ProxyConfig,
    pub upstream: UpstreamConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    pub bind_address: String,
    pub bind_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    pub address: String,
    pub port: u16,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            proxy: ProxyConfig {
                bind_address: "127.0.0.1".to_string(),
                bind_port: 25565,
            },
            upstream: UpstreamConfig {
                address: "localhost".to_string(),
                port: 25566,
            },
        }
    }
}
