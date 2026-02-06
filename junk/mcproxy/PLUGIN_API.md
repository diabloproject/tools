# Plugin API for Dynamic Loading

This document describes how to create dynamically loadable plugins for the Minecraft proxy.

## Current Status

The proxy is designed to support dynamic plugin loading via shared libraries (dylib), but currently only supports compiled-in plugins. The trait-based architecture is ready for dynamic loading.

## Plugin Types

All plugins must implement one of these traits:
- `PacketFilter` - Filter packets (allow/deny)
- `PacketMapper` - Transform packets
- `PacketInjector` - Inject new packets

## Creating a Dylib Plugin

### Step 1: Create a new library crate

```bash
cargo new --lib my_minecraft_plugin
cd my_minecraft_plugin
```

### Step 2: Configure Cargo.toml

```toml
[package]
name = "my_minecraft_plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
mcproxy = { path = "../mcproxy" }  # Or from crates.io once published
```

### Step 3: Implement your plugin

```rust
use mcproxy::plugin::{PacketDirection, PacketFilter, RawPacket};

#[derive(Debug)]
pub struct MyPlugin;

impl PacketFilter for MyPlugin {
    fn name(&self) -> &str {
        "MyDynamicPlugin"
    }

    fn filter(&self, packet: &RawPacket, direction: PacketDirection) -> bool {
        // Your logic here
        println!("Packet size: {} bytes, direction: {:?}", packet.data.len(), direction);
        true
    }
}

// Export the plugin creation function
#[no_mangle]
pub extern "C" fn create_filter() -> *mut dyn PacketFilter {
    Box::into_raw(Box::new(MyPlugin))
}

#[no_mangle]
pub extern "C" fn destroy_filter(ptr: *mut dyn PacketFilter) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}
```

### Step 4: Build the plugin

```bash
cargo build --release
```

This creates:
- Linux: `target/release/libmy_minecraft_plugin.so`
- macOS: `target/release/libmy_minecraft_plugin.dylib`
- Windows: `target/release/my_minecraft_plugin.dll`

## Loading Plugins (Future Implementation)

Here's how plugin loading will work once implemented in the proxy:

```rust
use libloading::{Library, Symbol};

pub struct PluginLoader;

impl PluginLoader {
    pub unsafe fn load_filter(path: &str) -> anyhow::Result<Box<dyn PacketFilter>> {
        let lib = Library::new(path)?;
        
        let create: Symbol<unsafe extern "C" fn() -> *mut dyn PacketFilter> = 
            lib.get(b"create_filter")?;
        
        let raw = create();
        Ok(Box::from_raw(raw))
    }
}

// Usage in main.rs
let plugin = unsafe { PluginLoader::load_filter("./plugins/my_plugin.so")? };
plugin_manager.add_filter(plugin);
```

## Safety Considerations

Dynamic plugin loading involves `unsafe` code and has several considerations:

1. **ABI Compatibility**: The plugin must be compiled with the same Rust version and settings as the proxy
2. **Memory Management**: Plugins must properly manage memory to avoid leaks
3. **Panic Safety**: Plugins should not panic across FFI boundaries
4. **Version Compatibility**: Plugin API versioning should be enforced

## Advanced: Plugin Configuration

Plugins can be configured via YAML:

```yaml
proxy:
  bind_address: "127.0.0.1"
  bind_port: 25565

upstream:
  address: "localhost"
  port: 25566

plugins:
  - type: filter
    path: "./plugins/my_filter.so"
    config:
      enabled: true
      max_packet_size: 1024
      
  - type: mapper
    path: "./plugins/my_mapper.so"
    config:
      transform_chat: true
```

## Example Plugins

### 1. Packet Size Monitor

```rust
#[derive(Debug)]
pub struct PacketSizeMonitor {
    threshold: usize,
}

impl PacketFilter for PacketSizeMonitor {
    fn name(&self) -> &str {
        "PacketSizeMonitor"
    }

    fn filter(&self, packet: &RawPacket, direction: PacketDirection) -> bool {
        if packet.data.len() > self.threshold {
            eprintln!(
                "WARNING: Large packet detected: {} bytes ({:?})",
                packet.data.len(),
                direction
            );
        }
        true
    }
}
```

### 2. Rate Limiter

```rust
use std::time::{Duration, Instant};
use std::sync::Mutex;

#[derive(Debug)]
pub struct RateLimiter {
    last_packet: Mutex<Instant>,
    min_interval: Duration,
}

impl PacketFilter for RateLimiter {
    fn name(&self) -> &str {
        "RateLimiter"
    }

    fn filter(&self, _packet: &RawPacket, direction: PacketDirection) -> bool {
        if direction == PacketDirection::Forward {
            let mut last = self.last_packet.lock().unwrap();
            let now = Instant::now();
            
            if now.duration_since(*last) < self.min_interval {
                return false; // Drop packet
            }
            
            *last = now;
        }
        true
    }
}
```

### 3. Packet Logger to File

```rust
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Mutex;

#[derive(Debug)]
pub struct FileLogger {
    file: Mutex<std::fs::File>,
}

impl PacketFilter for FileLogger {
    fn name(&self) -> &str {
        "FileLogger"
    }

    fn filter(&self, packet: &RawPacket, direction: PacketDirection) -> bool {
        if let Ok(mut file) = self.file.lock() {
            writeln!(
                file,
                "{:?} | ID: {:?} | Size: {}",
                direction,
                packet.packet_id,
                packet.data.len()
            ).ok();
        }
        true
    }
}
```

## Testing Your Plugin

Create a test harness:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn test_my_plugin() {
        let plugin = MyPlugin;
        let packet = RawPacket {
            data: Bytes::from_static(b"test data"),
            packet_id: Some(0x00),
        };
        
        assert!(plugin.filter(&packet, PacketDirection::Forward));
    }
}
```

## Future Enhancements

1. **Hot Reloading**: Reload plugins without restarting the proxy
2. **Plugin Dependencies**: Allow plugins to depend on other plugins
3. **Event System**: Plugins can subscribe to proxy events
4. **Metrics API**: Built-in metrics collection for plugins
5. **Inter-Plugin Communication**: Message passing between plugins

## Contributing

To add dynamic loading support to the proxy:

1. Add `libloading` dependency
2. Implement `PluginLoader` in `src/plugin_loader.rs`
3. Add plugin path configuration to `config.yaml`
4. Load plugins before starting the proxy server
5. Add proper error handling and logging
