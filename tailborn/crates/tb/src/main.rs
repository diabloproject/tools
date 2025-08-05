use std::error::Error;

use control::Credentials;
// Possible arguments:
// tb -control — run in control plane mode
// tb -dfs — run as a storage node
// tb -compute — run as a compute node
// tb -map "<command>" — run map operation on compute cluster

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Client,
    Control,
    Dfs,
    Compute,
    Shell,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", network::address::TailbornAddress::default());
    let conf = libconf::user_config()?;
    let log4rs_conf_path = libconf::log4rs_config_path()?;
    log4rs::init_file(log4rs_conf_path, Default::default()).expect("Invalid configuration.");
    let args: Vec<String> = std::env::args().collect();
    let mut mode = None;
    for arg in &args {
        if let Some(new_mode) = match arg.as_str() {
            "-control" => Some(Mode::Control),
            "-dfs" => Some(Mode::Dfs),
            "-compute" => Some(Mode::Compute),
            "-map" | "-reduce" => Some(Mode::Client),
            _ => None,
        } {
            if mode.is_some() {
                return Err("Multiple modes specified".into());
            }
            mode = Some(new_mode);
        }
    }

    let mode = mode.unwrap_or(Mode::Shell);

    match mode {
        Mode::Compute => {
            todo!()
        }
        Mode::Dfs => {
            todo!()
        }
        Mode::Client => {
            todo!()
        }
        Mode::Control => {
            control::run_control_server().await;
        }
        Mode::Shell => {
            control::connect("127.0.0.1:8080", Credentials::root()).await?;
        }
    }

    eprintln!("{conf:?}");
    println!("Hello, world!");
    Ok(())
}
