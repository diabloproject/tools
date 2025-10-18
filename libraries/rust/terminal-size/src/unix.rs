use super::{Height, Width};
use std::os::unix::io::AsFd;

pub fn terminal_size() -> Option<(Width, Height)> {
    if let Some(size) = terminal_size_of(std::io::stdout()) {
        Some(size)
    } else if let Some(size) = terminal_size_of(std::io::stderr()) {
        Some(size)
    } else if let Some(size) = terminal_size_of(std::io::stdin()) {
        Some(size)
    } else {
        None
    }
}

pub fn terminal_size_of<Fd: AsFd>(fd: Fd) -> Option<(Width, Height)> {
    use rustix::termios::{isatty, tcgetwinsize};

    if !isatty(&fd) {
        return None;
    }

    let winsize = tcgetwinsize(&fd).ok()?;

    let rows = winsize.ws_row;
    let cols = winsize.ws_col;

    if rows > 0 && cols > 0 {
        Some((Width(cols), Height(rows)))
    } else {
        None
    }
}
