use pixelate::{write_log_entry, LogEntry};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;

type DynError = Box<dyn std::error::Error>;

fn append_log_bytes(log: &mut dyn Write, value: impl AsRef<[u8]>) -> usize {
    log.write_all(value.as_ref())
        .expect("failed to write bytes");
    value.as_ref().len()
}

fn append_log(log: &mut dyn Write, entry: &LogEntry) -> usize {
    append_log_bytes(log, b"log{")
        + write_log_entry(log, entry)
        + append_log_bytes(log, b"};\r\n")
}

fn create_log<P: AsRef<Path>>(path: P) -> Box<dyn Write> {
    Box::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .expect("failed to open file"),
    )
}

fn recv_u32(io: &mut dyn Read) -> Result<u32, DynError> {
    let mut buf = [0u8; 4];
    io.read_exact(&mut buf)
        .map_err(|_| "client failed to provide necessary data")?;
    Ok(u32::from_be_bytes(buf))
}

fn handle_connection(log: &mut dyn Write, io: &mut TcpStream) -> Result<(), DynError> {
    let mut entry = LogEntry! { 0, 0, 0, 0 };
    entry.x = recv_u32(io)?;
    entry.y = recv_u32(io)?;
    entry.color = recv_u32(io)?;
    entry.user = recv_u32(io)?;
    append_log(log, &entry);
    Ok(())
}

fn main() {
    let listener = TcpListener::bind("0.0.0.0:8844").unwrap();
    let mut log = create_log("/tmp/fsmap");
    for connection in listener.incoming() {
        let mut con = connection.expect("Invalid connection");
        let res = handle_connection(&mut log, &mut con);
        if let Err(err) = res {
            println!("{}", err)
        }
    }
}
