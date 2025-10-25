use crate::LogEvent;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

pub(crate) fn start_pipe_listener(
    log_tx: std::sync::mpsc::Sender<LogEvent>,
) -> std::io::Result<String> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let pipe_addr = format!("127.0.0.1:{}", addr.port());

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let tx = log_tx.clone();
            std::thread::spawn(move || handle_client(stream, tx));
        }
    });

    Ok(pipe_addr)
}

fn handle_client(mut stream: TcpStream, tx: std::sync::mpsc::Sender<LogEvent>) {
    loop {
        match read_log_event(&mut stream) {
            Ok(Some(event)) => {
                if tx.send(event).is_err() {
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
}

fn read_log_event(stream: &mut TcpStream) -> std::io::Result<Option<LogEvent>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 10_000_000 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Message too large",
        ));
    }

    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;

    match deserialize_log_event(&buf) {
        Some(event) => Ok(Some(event)),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to deserialize",
        )),
    }
}

pub(crate) fn send_log_event(stream: &mut TcpStream, event: &LogEvent) -> std::io::Result<()> {
    let data = serialize_log_event(event);
    let len = data.len() as u32;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(&data)?;
    stream.flush()?;
    Ok(())
}

fn serialize_log_event(event: &LogEvent) -> Vec<u8> {
    let mut buf = Vec::new();
    match event {
        LogEvent::Log(record) => {
            buf.push(0);
            buf.push(record.level as u8);
            let msg_bytes = record.message.as_bytes();
            buf.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
            buf.extend_from_slice(msg_bytes);
            // Serialize timestamp as string to avoid needing internal access
            let ts_str = record.timestamp.to_string();
            let ts_bytes = ts_str.as_bytes();
            buf.extend_from_slice(&(ts_bytes.len() as u32).to_be_bytes());
            buf.extend_from_slice(ts_bytes);
        }
        LogEvent::CreateProgressBar {
            id,
            description,
            total,
        } => {
            buf.push(1);
            buf.extend_from_slice(&id.to_be_bytes());
            let desc_bytes = description.as_bytes();
            buf.extend_from_slice(&(desc_bytes.len() as u32).to_be_bytes());
            buf.extend_from_slice(desc_bytes);
            buf.extend_from_slice(&total.to_be_bytes());
        }
        LogEvent::PushProgress { id, value } => {
            buf.push(2);
            buf.extend_from_slice(&id.to_be_bytes());
            buf.extend_from_slice(&value.to_be_bytes());
        }
        LogEvent::FinishProgressBar { id } => {
            buf.push(3);
            buf.extend_from_slice(&id.to_be_bytes());
        }
        LogEvent::Drop { .. } => {
            unreachable!()
        }
    }
    buf
}

fn deserialize_log_event(buf: &[u8]) -> Option<LogEvent> {
    if buf.is_empty() {
        return None;
    }

    match buf[0] {
        0 => {
            if buf.len() < 6 {
                return None;
            }
            let level = match buf[1] {
                0 => crate::LogLevel::Trace,
                1 => crate::LogLevel::Debug,
                2 => crate::LogLevel::Info,
                3 => crate::LogLevel::Warn,
                4 => crate::LogLevel::Error,
                5 => crate::LogLevel::Fatal,
                _ => return None,
            };

            let msg_len = u32::from_be_bytes([buf[2], buf[3], buf[4], buf[5]]) as usize;
            if buf.len() < 6 + msg_len + 4 {
                return None;
            }

            let message = String::from_utf8(buf[6..6 + msg_len].to_vec()).ok()?;

            let ts_len_offset = 6 + msg_len;
            let ts_len = u32::from_be_bytes([
                buf[ts_len_offset],
                buf[ts_len_offset + 1],
                buf[ts_len_offset + 2],
                buf[ts_len_offset + 3],
            ]) as usize;

            if buf.len() < ts_len_offset + 4 + ts_len {
                return None;
            }

            let ts_str =
                String::from_utf8(buf[ts_len_offset + 4..ts_len_offset + 4 + ts_len].to_vec())
                    .ok()?;
            let timestamp = ts_str.parse::<stdd::time::DateTime>().ok()?;

            Some(LogEvent::Log(crate::LogRecord {
                level,
                message,
                timestamp,
            }))
        }
        1 => {
            if buf.len() < 13 {
                return None;
            }
            let id = u64::from_be_bytes([
                buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8],
            ]);
            let desc_len = u32::from_be_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;
            if buf.len() < 13 + desc_len + 8 {
                return None;
            }
            let description = String::from_utf8(buf[13..13 + desc_len].to_vec()).ok()?;
            let total = u64::from_be_bytes([
                buf[13 + desc_len],
                buf[14 + desc_len],
                buf[15 + desc_len],
                buf[16 + desc_len],
                buf[17 + desc_len],
                buf[18 + desc_len],
                buf[19 + desc_len],
                buf[20 + desc_len],
            ]);
            Some(LogEvent::CreateProgressBar {
                id,
                description,
                total,
            })
        }
        2 => {
            if buf.len() < 17 {
                return None;
            }
            let id = u64::from_be_bytes([
                buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8],
            ]);
            let value = u64::from_be_bytes([
                buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15], buf[16],
            ]);
            Some(LogEvent::PushProgress { id, value })
        }
        3 => {
            if buf.len() < 9 {
                return None;
            }
            let id = u64::from_be_bytes([
                buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8],
            ]);
            Some(LogEvent::FinishProgressBar { id })
        }
        _ => None,
    }
}
