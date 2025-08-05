use std::collections::HashMap;
use std::net::Ipv4Addr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserConfiguration {
    pub hosts: HashMap<String, Host>,
}

impl Default for UserConfiguration {
    fn default() -> Self {
        let mut conf = Self {
            hosts: HashMap::new(),
        };
        conf.hosts.insert(
            "myself".to_string(),
            Host::Local {
                roles: serde_symbol_default_roles(),
            },
        );
        conf
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Compute,
    Dfs,
}

fn serde_symbol_default_port() -> u16 {
    6298
}

fn serde_symbol_default_roles() -> Vec<Role> {
    Vec::from([Role::Compute, Role::Dfs])
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Host {
    Local {
        #[serde(default = "serde_symbol_default_roles")]
        roles: Vec<Role>,
    },
    Remote {
        ip: Ipv4Addr,
        #[serde(default = "serde_symbol_default_port")]
        port: u16,
        #[serde(default = "serde_symbol_default_roles")]
        roles: Vec<Role>,
        #[serde(flatten)]
        access_scheme: AccessScheme,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "access_scheme", rename_all = "lowercase")]
pub enum AccessScheme {
    /// Scheme running assumes that the server is alredy available
    /// at the address specified in configuration. Fail if not.
    #[default]
    #[serde(rename = "running")]
    Running,
    /// Scheme ssh
    #[serde(rename = "ssh")]
    Ssh {
        username: String,
        #[serde(flatten)]
        auth_method: AuthMethod,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum AuthMethod {
    Password {
        password: String,
    },
    IdentityFile {
        identity_file: String,
    },
    Key {
        key: String,
    },
    #[default]
    Auto,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigurationFileError {
    #[error(transparent)]
    Toml(#[from] toml::de::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
