# Minecraft Proxy Server

A Minecraft proxy server written in Rust that sits between the client and upstream server, allowing packet inspection, filtering, mapping, and injection.

## Architecture

### Core Components

- **Client**: The Minecraft game client that connects to the proxy
- **Upstream**: The actual Minecraft server that processes game events
- **Proxy**: This server that forwards packets bidirectionally

### Plugin System

The proxy supports three types of packet processors:

#### 1. Packet Filters
Filters decide whether a packet should be forwarded or dropped.

```rust
pub trait PacketFilter: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn filter(&self, packet: &RawPacket, direction: PacketDirection) -> bool;
}
```

#### 2. Packet Mappers
Mappers transform packets before forwarding or can drop them.

```rust
pub trait PacketMapper: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn map(&self, packet: RawPacket, direction: PacketDirection) -> PacketAction;
}
```

#### 3. Packet Injectors
Injectors can create and inject new packets into the stream based on conditions.

```rust
pub trait PacketInjector: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn should_inject(&self, direction: PacketDirection) -> bool;
    fn inject(&self, direction: PacketDirection) -> Vec<RawPacket>;
}
```

### Packet Direction

- **Forward**: Packets going from client → upstream
- **Backward**: Packets going from upstream → client

## Configuration

Create a `config.yaml` file:

```yaml
proxy:
  bind_address: "127.0.0.1"
  bind_port: 25565

upstream:
  address: "localhost"
  port: 25566
```

## Usage

### Running the Proxy

```bash
# Use default config.yaml
cargo run

# Use custom config
cargo run -- /path/to/config.yaml

# With debug logging
RUST_LOG=debug cargo run
```

### Creating Custom Plugins

Plugins work with `Box<dyn Plugin>` for future dynamic library support.

#### Example Filter Plugin

```rust
use crate::plugin::{PacketDirection, PacketFilter, RawPacket};

#[derive(Debug)]
pub struct MyFilter;

impl PacketFilter for MyFilter {
    fn name(&self) -> &str {
        "MyFilter"
    }

    fn filter(&self, packet: &RawPacket, direction: PacketDirection) -> bool {
        // Return false to drop the packet
        true
    }
}

// Register in main.rs
plugin_manager.add_filter(Box::new(MyFilter));
```

#### Example Mapper Plugin

```rust
use crate::plugin::{PacketDirection, PacketMapper, PacketAction, RawPacket};

#[derive(Debug)]
pub struct MyMapper;

impl PacketMapper for MyMapper {
    fn name(&self) -> &str {
        "MyMapper"
    }

    fn map(&self, packet: RawPacket, direction: PacketDirection) -> PacketAction {
        // Transform packet data here
        PacketAction::Forward(packet)
        
        // Or drop it
        // PacketAction::Drop
    }
}

// Register in main.rs
plugin_manager.add_mapper(Box::new(MyMapper));
```

#### Example Injector Plugin

```rust
use crate::plugin::{PacketDirection, PacketInjector, RawPacket};
use bytes::Bytes;

#[derive(Debug)]
pub struct MyInjector {
    should_inject: bool,
}

impl PacketInjector for MyInjector {
    fn name(&self) -> &str {
        "MyInjector"
    }

    fn should_inject(&self, direction: PacketDirection) -> bool {
        self.should_inject && direction == PacketDirection::Forward
    }

    fn inject(&self, _direction: PacketDirection) -> Vec<RawPacket> {
        vec![RawPacket {
            data: Bytes::from_static(b"custom packet data"),
            packet_id: Some(0x00),
        }]
    }
}

// Register in main.rs
plugin_manager.add_injector(Box::new(MyInjector { should_inject: true }));
```

## Built-in Plugins

### PacketLogger
Logs packet information for debugging.

```rust
// Log both directions
plugin_manager.add_filter(Box::new(PacketLogger::new(true, true)));

// Log only client → upstream
plugin_manager.add_filter(Box::new(PacketLogger::new(true, false)));
```

### ChatFilter
Example filter for blocking chat messages containing specific words (placeholder implementation).

```rust
plugin_manager.add_filter(Box::new(ChatFilter::new(vec![
    "badword".to_string(),
])));
```

## Future Enhancements

### Dynamic Plugin Loading (dylib)
The architecture supports loading plugins as dynamic libraries. To implement:

1. Define a plugin loading API
2. Use `libloading` crate to load `.so`/`.dylib`/`.dll` files
3. Plugins export a function that returns `Box<dyn PacketFilter>` etc.

```rust
// Example future API
let plugin: Box<dyn PacketFilter> = load_plugin("./plugins/my_filter.so")?;
plugin_manager.add_filter(plugin);
```

### Full Minecraft Protocol Support
Currently, the proxy forwards raw bytes. For full functionality:

1. Implement VarInt/VarLong encoding/decoding
2. Parse packet headers (length, ID)
3. Handle compression (threshold negotiation)
4. Handle encryption (AES/CFB8 after handshake)
5. Parse specific packet types (chat, position, etc.)

## Supported Minecraft Versions

Designed for Minecraft 1.20.1, 1.20.x, and 1.21.x. The raw packet approach means the proxy is protocol-agnostic, but plugins that inspect packet contents need version-specific knowledge.

## Development

```bash
# Build
cargo build

# Build with optimizations
cargo build --release

# Run tests
cargo test

# Check code
cargo clippy
```

## License

This project is provided as-is for educational and development purposes.
