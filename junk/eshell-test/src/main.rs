use std::io::Write;
use std::time::Duration;

use terminal_size::terminal_size;
use tui_elements as te;

fn main() {
    println!("{:?}", terminal_size());
    te::sequences::cursor::move_home();
    for i in 0..80 {
        std::thread::sleep(Duration::from_millis(100));
        print!("\r|{:2}| {}", i, "#".repeat(i));
        std::io::stdout().flush().unwrap();
    }
}
