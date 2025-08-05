pub mod logging;
pub mod user;

use std::path::PathBuf;

use crate::user::ConfigurationFileError;
use crate::user::UserConfiguration;

pub fn user_config() -> Result<UserConfiguration, ConfigurationFileError> {
    let config_dir = directories::ProjectDirs::from("dev", "diabloproject", "tailborn")
        .unwrap()
        .config_dir()
        .to_path_buf();
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).unwrap();
    }
    let config_file = config_dir.join("config.toml");
    if !config_file.exists() {
        let default_config = UserConfiguration::default();
        let content =
            toml::to_string(&default_config).expect("Failed to serialize default configuration");
        std::fs::write(
            &config_file,
            format!(
                "# `myself` host, configured by default, will execute everythig on your local machine. Add other hosts to use tailborn at scale!\n{}",
                content
            ),
        )?;
    }
    let content = ::std::fs::read_to_string(&config_file)?;
    let conf: UserConfiguration = ::toml::from_str(&content)?;
    Ok(conf)
}

pub fn log4rs_config_path() -> Result<PathBuf, ConfigurationFileError> {
    let project_dirs = directories::ProjectDirs::from("dev", "diabloproject", "tailborn").unwrap();
    let config_dir = project_dirs.config_dir().to_path_buf();
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).unwrap();
    }
    let config_file = config_dir.join("log4rs.yml");
    if !config_file.exists() {
        let content = include_str!("log4rs.yml");
        std::fs::write(&config_file, content)?;
    }
    Ok(config_file)
}
