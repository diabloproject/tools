use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TilingConfig {
    pub size: (u64, u64),
    pub pos: (i64, i64),
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub tiling: TilingConfig
}

pub fn get_config() -> Config {
    let settings = ::config::Config::builder()
        .add_source(::config::File::with_name("config.yaml"))
        .add_source(::config::Environment::with_prefix("PIXELATE_ENGINE_PARTITION"))
        .build()
        .unwrap();

    settings.try_deserialize().expect("Invalid configuration")
}