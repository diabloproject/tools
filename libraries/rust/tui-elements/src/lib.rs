pub mod sequences {
    pub const CSI: u8 = b'[';
    pub const DELIMITER: u8 = b';';
    pub const BEL: u8 = 0x07;
    pub const BS: u8 = 0x08;
    pub const HT: u8 = 0x09;
    pub const LF: u8 = 0x0a;
    pub const VT: u8 = 0x0b;
    pub const FF: u8 = 0x0c;
    pub const CR: u8 = 0x0d;
    pub const ESC: u8 = 0x1b;
    pub const DEL: u8 = 0x7f;

    pub mod cursor {
        use super::*;
        use std::io::{Read, Write, stdin, stdout};
        const MAX_U64: usize = 20;

        pub fn move_home() {
            stdout()
                .write_all(&[ESC, CSI, b'H'])
                .expect("failed to write to stdout");
        }

        pub fn move_up(n: u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "{}A", n).expect("failed to write to stdout");
        }

        pub fn move_down(n: u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "{}B", n).expect("failed to write to stdout");
        }

        pub fn move_right(n: u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "{}C", n).expect("failed to write to stdout");
        }

        pub fn move_left(n: u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "{}D", n).expect("failed to write to stdout");
        }

        pub fn move_after(n: u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "{}E", n).expect("failed to write to stdout");
        }

        pub fn move_before(n: u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "{}F", n).expect("failed to write to stdout");
        }

        pub fn move_to_column(n: u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "{}G", n).expect("failed to write to stdout");
        }

        pub fn get_position() -> (u64, u64) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "?6n").expect("failed to write to stdout");
            stdout().flush().expect("failed to flush stdout");
            let mut buf = [0u8; MAX_U64 * 2 + 4];
            let len = stdin().read(&mut buf).expect("failed to read from stdin");
            assert!(len > 0);
            assert!(buf[0] == ESC);
            assert!(buf[1] == CSI);
            let mut del_pos = 0;
            while del_pos < buf.len() {
                if buf[del_pos] == DELIMITER {
                    break;
                }
                del_pos += 1;
            }
            assert!(del_pos < buf.len());
            let linen_view = &buf[2..del_pos];
            let mut finish_pos = del_pos;
            while finish_pos < buf.len() {
                if buf[finish_pos] == b'R' {
                    break;
                }
                finish_pos += 1;
            }
            assert!(finish_pos < buf.len());
            let col_view = &buf[del_pos + 1..finish_pos];
            let line: u64 = unsafe { std::str::from_utf8_unchecked(linen_view) }
                .parse()
                .expect("Terminal reported invalid line number");
            let col: u64 = unsafe { std::str::from_utf8_unchecked(col_view) }
                .parse()
                .expect("Terminal reported invalid column number");
            (line, col)
        }

        pub fn move_up_by_one_and_scroll() {
            stdout()
                .write_all(&[ESC, b'M'])
                .expect("failed to write to stdout");
        }

        pub fn save_cursor_position() {
            stdout()
                .write_all(&[ESC, b'7'])
                .expect("failed to write to stdout");
        }

        pub fn restore_cursor_position() {
            stdout()
                .write_all(&[ESC, b'8'])
                .expect("failed to write to stdout");
        }
    }

    pub mod erase {
        use super::*;
        use std::io::{Write, stdout};

        pub fn erase_in_display() {
            stdout()
                .write_all(&[ESC, CSI, b'J'])
                .expect("failed to write to stdout");
        }

        pub fn erase_screen_after_cursor() {
            stdout()
                .write_all(&[ESC, CSI, b'0', b'J'])
                .expect("failed to write to stdout");
        }

        pub fn erase_screen_before_cursor() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'J'])
                .expect("failed to write to stdout");
        }

        pub fn erase_screen() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'J'])
                .expect("failed to write to stdout");
        }

        pub fn erase_saved_lines() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'J'])
                .expect("failed to write to stdout");
        }

        pub fn erase_in_line() {
            stdout()
                .write_all(&[ESC, CSI, b'K'])
                .expect("failed to write to stdout");
        }

        pub fn erase_from_cursor_to_end_of_line() {
            stdout()
                .write_all(&[ESC, CSI, b'0', b'K'])
                .expect("failed to write to stdout");
        }

        pub fn erase_from_cursor_to_start_of_line() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'K'])
                .expect("failed to write to stdout");
        }

        pub fn erase_entire_line() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'K'])
                .expect("failed to write to stdout");
        }
    }

    pub mod graphics {
        use super::*;
        use std::io::{Write, stdout};

        // Reset all modes
        pub fn reset() {
            stdout()
                .write_all(&[ESC, CSI, b'0', b'm'])
                .expect("failed to write to stdout");
        }

        // Text styling
        pub fn set_bold() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_dim() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn reset_bold_dim() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'2', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_italic() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn reset_italic() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'3', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_underline() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn reset_underline() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'4', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_blinking() {
            stdout()
                .write_all(&[ESC, CSI, b'5', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn reset_blinking() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'5', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_inverse() {
            stdout()
                .write_all(&[ESC, CSI, b'7', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn reset_inverse() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'7', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_hidden() {
            stdout()
                .write_all(&[ESC, CSI, b'8', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn reset_hidden() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'8', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_strikethrough() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn reset_strikethrough() {
            stdout()
                .write_all(&[ESC, CSI, b'2', b'9', b'm'])
                .expect("failed to write to stdout");
        }

        // 8/16 basic colors - Foreground
        pub fn set_fg_black() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'0', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_red() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'1', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_green() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'2', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_yellow() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'3', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_blue() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'4', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_magenta() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'5', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_cyan() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'6', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_white() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'7', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_default() {
            stdout()
                .write_all(&[ESC, CSI, b'3', b'9', b'm'])
                .expect("failed to write to stdout");
        }

        // 8/16 basic colors - Background
        pub fn set_bg_black() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'0', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_red() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'1', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_green() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'2', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_yellow() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'3', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_blue() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'4', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_magenta() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'5', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_cyan() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'6', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_white() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'7', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_default() {
            stdout()
                .write_all(&[ESC, CSI, b'4', b'9', b'm'])
                .expect("failed to write to stdout");
        }

        // Bright colors (aixterm) - Foreground
        pub fn set_fg_bright_black() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'0', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_bright_red() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'1', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_bright_green() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'2', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_bright_yellow() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'3', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_bright_blue() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'4', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_bright_magenta() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'5', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_bright_cyan() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'6', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_fg_bright_white() {
            stdout()
                .write_all(&[ESC, CSI, b'9', b'7', b'm'])
                .expect("failed to write to stdout");
        }

        // Bright colors (aixterm) - Background
        pub fn set_bg_bright_black() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'0', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_bright_red() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'1', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_bright_green() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'2', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_bright_yellow() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'3', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_bright_blue() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'4', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_bright_magenta() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'5', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_bright_cyan() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'6', b'm'])
                .expect("failed to write to stdout");
        }

        pub fn set_bg_bright_white() {
            stdout()
                .write_all(&[ESC, CSI, b'1', b'0', b'7', b'm'])
                .expect("failed to write to stdout");
        }

        // 256 colors
        pub fn set_fg_color_256(id: u8) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "38;5;{}m", id).expect("failed to write to stdout");
        }

        pub fn set_bg_color_256(id: u8) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "48;5;{}m", id).expect("failed to write to stdout");
        }

        // RGB/Truecolor
        pub fn set_fg_color_rgb(r: u8, g: u8, b: u8) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "38;2;{};{};{}m", r, g, b).expect("failed to write to stdout");
        }

        pub fn set_bg_color_rgb(r: u8, g: u8, b: u8) {
            stdout()
                .write_all(&[ESC, CSI])
                .expect("failed to write to stdout");
            write!(stdout(), "48;2;{};{};{}m", r, g, b).expect("failed to write to stdout");
        }
    }

    pub mod screen {
        use super::*;
        use std::io::{Write, stdout};

        pub fn set_mode_40x25_mono() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'0', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_40x25_color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_80x25_mono() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'2', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_80x25_color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'3', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_320x200_4color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'4', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_320x200_mono() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'5', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_640x200_mono() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'6', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn enable_line_wrapping() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'7', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn disable_line_wrapping() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'7', b'l'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_320x200_color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'3', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_640x200_16color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'4', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_640x350_mono() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'5', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_640x350_16color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'6', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_640x480_mono() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'7', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_640x480_16color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'8', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn set_mode_320x200_256color() {
            stdout()
                .write_all(&[ESC, CSI, b'=', b'1', b'9', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn reset_mode(value: u8) {
            stdout()
                .write_all(&[ESC, CSI, b'='])
                .expect("failed to write to stdout");
            write!(stdout(), "{}l", value).expect("failed to write to stdout");
        }

        pub fn hide_cursor() {
            stdout()
                .write_all(&[ESC, CSI, b'?', b'2', b'5', b'l'])
                .expect("failed to write to stdout");
        }

        pub fn show_cursor() {
            stdout()
                .write_all(&[ESC, CSI, b'?', b'2', b'5', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn restore_screen() {
            stdout()
                .write_all(&[ESC, CSI, b'?', b'4', b'7', b'l'])
                .expect("failed to write to stdout");
        }

        pub fn save_screen() {
            stdout()
                .write_all(&[ESC, CSI, b'?', b'4', b'7', b'h'])
                .expect("failed to write to stdout");
        }

        pub fn enable_alternative_buffer() {
            stdout()
                .write_all(&[
                    ESC, CSI, b'?', b'1', b'0', b'4', b'9', b'h',
                ])
                .expect("failed to write to stdout");
        }

        pub fn disable_alternative_buffer() {
            stdout()
                .write_all(&[
                    ESC, CSI, b'?', b'1', b'0', b'4', b'9', b'l',
                ])
                .expect("failed to write to stdout");
        }
    }
}
