pub struct LogEntry {
    pub x: u32,
    pub y: u32,
    pub color: u32,
    pub user: u32,
}

#[macro_export]
macro_rules! LogEntry {
    {$x: expr, $y: expr, $color: expr, $user: expr} => {
        LogEntry {
            x: ($x),
            y: ($y),
            color: ($color),
            user: ($user)
        }
    };
}

fn write_bytes(log: &mut dyn std::io::Write, value: impl AsRef<[u8]>) -> usize {
    log.write_all(value.as_ref())
        .expect("failed to write bytes");
    value.as_ref().len()
}

pub fn write_log_entry(sink: &mut dyn std::io::Write, entry: &LogEntry) -> usize {
    write_bytes(sink, entry.x.to_be_bytes())
        + write_bytes(sink, entry.y.to_be_bytes())
        + write_bytes(sink, entry.color.to_be_bytes())
        + write_bytes(sink, entry.user.to_be_bytes())
}
