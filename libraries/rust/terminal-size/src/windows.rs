use super::{Height, Width};
use std::os::windows::io::{AsHandle, AsRawHandle, BorrowedHandle, RawHandle};

pub fn terminal_size() -> Option<(Width, Height)> {
    use windows_sys::Win32::System::Console::{
        GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };

    if let Some(size) = terminal_size_of(unsafe {
        BorrowedHandle::borrow_raw(GetStdHandle(STD_OUTPUT_HANDLE) as RawHandle)
    }) {
        Some(size)
    } else if let Some(size) = terminal_size_of(unsafe {
        BorrowedHandle::borrow_raw(GetStdHandle(STD_ERROR_HANDLE) as RawHandle)
    }) {
        Some(size)
    } else if let Some(size) = terminal_size_of(unsafe {
        BorrowedHandle::borrow_raw(GetStdHandle(STD_INPUT_HANDLE) as RawHandle)
    }) {
        Some(size)
    } else {
        None
    }
}

pub fn terminal_size_of<Handle: AsHandle>(handle: Handle) -> Option<(Width, Height)> {
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Console::{
        CONSOLE_SCREEN_BUFFER_INFO, COORD, GetConsoleScreenBufferInfo, SMALL_RECT,
    };

    // convert between windows_sys::Win32::Foundation::HANDLE and std::os::windows::raw::HANDLE
    let hand = handle.as_handle().as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;

    if hand == INVALID_HANDLE_VALUE {
        return None;
    }

    let zc = COORD { X: 0, Y: 0 };
    let mut csbi = CONSOLE_SCREEN_BUFFER_INFO {
        dwSize: zc,
        dwCursorPosition: zc,
        wAttributes: 0,
        srWindow: SMALL_RECT {
            Left: 0,
            Top: 0,
            Right: 0,
            Bottom: 0,
        },
        dwMaximumWindowSize: zc,
    };
    if unsafe { GetConsoleScreenBufferInfo(hand, &mut csbi) } == 0 {
        return None;
    }

    let w: Width = Width((csbi.srWindow.Right - csbi.srWindow.Left + 1) as u16);
    let h: Height = Height((csbi.srWindow.Bottom - csbi.srWindow.Top + 1) as u16);
    Some((w, h))
}
