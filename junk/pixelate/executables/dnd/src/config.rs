use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TilingConfig {
    pub size: (u64, u64),
    pub pos: (i64, i64),
}

#[derive(Debug, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_backing_store")]
    pub backing_store: String,
    #[serde(default = "default_controller")]
    pub controller: String,
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
}

fn default_backing_store() -> String {
    "ram".to_string()
}

fn default_controller() -> String {
    "optimized".to_string()
}

fn default_data_dir() -> String {
    "./data".to_string()
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backing_store: default_backing_store(),
            controller: default_controller(),
            data_dir: default_data_dir(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub tiling: TilingConfig,
    #[serde(default)]
    pub storage: StorageConfig,
}

pub fn get_config() -> Config {
    let settings = ::config::Config::builder()
        .add_source(::config::File::with_name("config.yaml"))
        .add_source(::config::Environment::with_prefix("PIXELATE_ENGINE_PARTITION"))
        .build()
        .unwrap();

    settings.try_deserialize().expect("Invalid configuration")
}