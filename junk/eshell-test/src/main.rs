use std::time::Duration;
use std::io::Write;

use tui_elements as te;
use terminal_size::terminal_size;

fn main() {
    println!("{:?}", terminal_size());
    te::sequences::cursor::move_home();
    for i in 0..80 {
        std::thread::sleep(Duration::from_millis(100));
        print!("\r|{:2}| {}", i, "#".repeat(i));
        std::io::stdout().flush().unwrap();
    }
}
