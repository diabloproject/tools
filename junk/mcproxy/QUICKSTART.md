# Quick Start Guide

Get your Minecraft proxy up and running in 5 minutes!

## Prerequisites

- Rust 1.70+ installed ([rustup.rs](https://rustup.rs))
- A Minecraft server to proxy to
- Basic understanding of Minecraft server setup

## Installation

### 1. Clone or Download

```bash
cd mcproxy
```

### 2. Build the Project

```bash
cargo build --release
```

The binary will be at `target/release/mcproxy`

## Configuration

### 3. Create Configuration File

Copy the example config:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
proxy:
  bind_address: "127.0.0.1"
  bind_port: 25565        # Port clients connect to

upstream:
  address: "localhost"
  port: 25566             # Your actual Minecraft server port
```

**Important**: Make sure your upstream Minecraft server is running on port 25566 (or whatever port you configure).

## Running

### 4. Start Your Minecraft Server

First, start your actual Minecraft server on port 25566:

```bash
# Example with vanilla server
cd minecraft_server
java -Xmx1024M -Xms1024M -jar server.jar nogui
```

In `server.properties`, ensure:
```properties
server-port=25566
```

### 5. Start the Proxy

In a new terminal:

```bash
# Development mode with debug logging
RUST_LOG=debug cargo run

# Or production mode
./target/release/mcproxy
```

You should see:
```
INFO Starting Minecraft proxy server
INFO Listening on: 127.0.0.1:25565
INFO Upstream server: localhost:25566
INFO Registered filter: PacketLogger
INFO Registered filter: ChatFilter
```

### 6. Connect Your Minecraft Client

1. Open Minecraft (Java Edition)
2. Go to Multiplayer
3. Add Server:
   - **Server Address**: `127.0.0.1:25565`
   - **Server Name**: Whatever you like
4. Connect!

## Verification

You should see log output like:

```
INFO New client connection from: 127.0.0.1:54321
TRACE Read 234 bytes from client
TRACE Read 512 bytes from upstream
INFO Forward connection closed
INFO Backward connection closed
INFO Client 127.0.0.1:54321 disconnected
```

## Troubleshooting

### Connection Refused

**Problem**: Client can't connect to proxy
```
io error: Connection refused (os error 61)
```

**Solution**: 
- Ensure proxy is running
- Check firewall settings
- Verify port 25565 is not already in use

### Upstream Connection Failed

**Problem**: Proxy can't connect to upstream
```
Failed to establish upstream connection
```

**Solution**:
- Ensure Minecraft server is running
- Verify server is on the correct port (25566)
- Check `config.yaml` has correct upstream address

### Port Already in Use

**Problem**: 
```
Address already in use (os error 48)
```

**Solution**:
- Another process is using port 25565
- Find and stop it: `lsof -i :25565`
- Or change the proxy port in `config.yaml`

## Testing Without a Real Server

For testing the proxy without a real Minecraft server, you can use netcat:

```bash
# Terminal 1: Mock upstream server
nc -l 25566

# Terminal 2: Start proxy
cargo run

# Terminal 3: Test connection
nc localhost 25565
```

Type in terminal 3, and you should see it in terminal 1 through the proxy!

## Next Steps

### Customize Plugins

Edit `src/main.rs` to enable/disable plugins:

```rust
let mut plugin_manager = PluginManager::new();

// Enable packet logging (both directions)
plugin_manager.add_filter(Box::new(PacketLogger::new(true, true)));

// Enable chat filter
plugin_manager.add_filter(Box::new(ChatFilter::new(vec![
    "badword".to_string(),
    "spam".to_string(),
])));

// Add bandwidth limiter
plugin_manager.add_mapper(Box::new(BandwidthLimiter::new(32768)));
```

### Create Your Own Plugin

See [README.md](README.md) and [PLUGIN_API.md](PLUGIN_API.md) for detailed instructions.

### Production Deployment

For production use:

1. **Build optimized binary**:
   ```bash
   cargo build --release
   strip target/release/mcproxy
   ```

2. **Create systemd service** (Linux):
   ```bash
   sudo cp target/release/mcproxy /usr/local/bin/
   sudo cp config.yaml /etc/mcproxy/
   ```

3. **Set appropriate logging**:
   ```bash
   RUST_LOG=info ./mcproxy
   ```

4. **Monitor logs**:
   ```bash
   tail -f /var/log/mcproxy.log
   ```

## Performance Tips

- **Production builds**: Always use `--release` for 10-100x speedup
- **Logging**: Set `RUST_LOG=info` or `warn` in production (not `debug`/`trace`)
- **Buffer size**: Adjust in `connection.rs` for your network (default 8KB)
- **Connection limits**: Add a plugin to limit concurrent connections

## Common Use Cases

### 1. Development/Testing
- Test server changes without affecting players
- Inspect packets for debugging
- Simulate network conditions

### 2. Security
- Filter malicious packets
- Rate limit clients
- Log suspicious activity

### 3. Custom Features
- Custom commands via packet injection
- Protocol translation
- Multi-server routing

### 4. Analytics
- Track player actions
- Monitor server load
- Collect metrics

## Support

- **Documentation**: See [README.md](README.md) for full details
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for internals
- **Plugin API**: See [PLUGIN_API.md](PLUGIN_API.md) for plugin development

## Examples

### Example 1: Log All Packets

```bash
# Set environment variable for detailed logging
RUST_LOG=trace cargo run
```

### Example 2: Multiple Servers

Run multiple proxy instances for load balancing:

```bash
# Proxy 1
./mcproxy config1.yaml &

# Proxy 2  
./mcproxy config2.yaml &

# Proxy 3
./mcproxy config3.yaml &
```

### Example 3: Remote Server

Connect to a remote Minecraft server:

```yaml
proxy:
  bind_address: "0.0.0.0"  # Listen on all interfaces
  bind_port: 25565

upstream:
  address: "mc.example.com"  # Remote server
  port: 25565
```

## What's Next?

Now that your proxy is running, explore:

1. **Plugin Development**: Create custom packet processors
2. **Protocol Implementation**: Add full Minecraft protocol parsing
3. **Dynamic Loading**: Load plugins as shared libraries
4. **Web Dashboard**: Monitor connections in real-time

Happy proxying! 🎮
