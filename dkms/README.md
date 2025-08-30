# DKMS - Distributed Key Management Service

A secure, lightweight REST API for managing encrypted keys and secrets using FastAPI.

## Features

- 🔐 **Secure Encryption**: Uses Fernet symmetric encryption with PBKDF2 key derivation
- 🗄️ **SQLite Storage**: Local database storage with async operations
- 🔑 **API Key Authentication**: Bearer token authentication for all endpoints
- 🚀 **FastAPI**: Modern, fast web framework with automatic API documentation
- 📝 **Type Safety**: Full type hints and Pydantic models
- 🧪 **Testing**: Included test suite for API verification
- 🖥️ **CLI Client**: Typer-based command line interface for easy interaction

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

The installation includes both the server components and the CLI client.

### 2. Set Environment Variables (Optional)

```bash
# Set custom API key (default: "default-dev-key")
export API_KEY="your-secure-api-key-here"

# Set custom encryption key (default: "default-crypto-key") 
export CRYPTO_KEY="your-secure-encryption-key-here"

# Set custom host/port (defaults: 127.0.0.1:8000)
export HOST="0.0.0.0"
export PORT="8080"
```

### 3. Start the Server

```bash
uv run python start_server.py
```

Or use FastAPI directly:

```bash
uv run fastapi dev main.py
```

The server will be available at `http://127.0.0.1:8000` with automatic API documentation at `/docs`.

## API Endpoints

### Authentication

All endpoints require a Bearer token with your API key:

```bash
Authorization: Bearer your-api-key-here
```

### Store a Key

**POST** `/key/{name}`

```bash
curl -X POST "http://127.0.0.1:8000/key/my-secret" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"value": "my-secret-password"}'
```

Response:
```json
{
  "name": "my-secret",
  "value": "my-secret-password"
}
```

### Retrieve a Key

**GET** `/key/{name}`

```bash
curl -X GET "http://127.0.0.1:8000/key/my-secret" \
  -H "Authorization: Bearer your-api-key"
```

Response:
```json
{
  "name": "my-secret", 
  "value": "my-secret-password"
}
```

### Delete a Key

**DELETE** `/key/{name}`

```bash
curl -X DELETE "http://127.0.0.1:8000/key/my-secret" \
  -H "Authorization: Bearer your-api-key"
```

Response:
```json
{
  "message": "Key 'my-secret' deleted successfully"
}
```

### List All Keys

**GET** `/keys`

```bash
curl -X GET "http://127.0.0.1:8000/keys" \
  -H "Authorization: Bearer your-api-key"
```

Response:
```json
{
  "keys": ["my-secret", "api-token", "database-password"]
}
```

## Testing

The DKMS implementation includes comprehensive tests to verify functionality:

### Component Tests (No Server Required)

Test the core crypto and database functionality:

```bash
uv run python simple_test.py
```

### API Tests (No Server Required)

Test the FastAPI endpoints using TestClient:

```bash
uv run python test_server.py
```

### Live Server Tests (Server Required)

Test against a running server:

```bash
# Start the server in one terminal
uv run python start_server.py

# Run tests in another terminal
uv run python test_api.py
```

### Example Usage

See comprehensive usage examples:

```bash
# Start the server first
uv run python start_server.py

# Run examples in another terminal
uv run python example_usage.py
```

This will demonstrate:
- Basic key storage and retrieval
- Application configuration management
- Key rotation workflows
- Multi-environment deployments

## CLI Client Usage

DKMS includes a powerful command-line interface built with Typer for easy interaction with the key management service.

### CLI Quick Start

```bash
# Show CLI configuration
uv run python cli.py config

# Store a key
uv run python cli.py set my-secret --value "secret-password"

# Store a key with secure prompt (recommended)
uv run python cli.py set my-secret --prompt

# Retrieve a key
uv run python cli.py get my-secret

# List all keys
uv run python cli.py list

# Delete a key
uv run python cli.py delete my-secret
```

### CLI Commands

#### `config` - Show Configuration
```bash
uv run python cli.py config [--url URL] [--api-key KEY]
```

#### `set` - Store a Key
```bash
# With inline value
uv run python cli.py set KEY_NAME --value "secret"

# With secure prompt (recommended for sensitive data)
uv run python cli.py set KEY_NAME --prompt

# From stdin (for scripting)
echo "secret-value" | uv run python cli.py set KEY_NAME
```

#### `get` - Retrieve a Key
```bash
# Show key and value
uv run python cli.py get KEY_NAME

# Check existence without showing value
uv run python cli.py get KEY_NAME --hide-value
```

#### `list` - List All Keys
```bash
# Table format (default)
uv run python cli.py list

# Simple format (one per line)
uv run python cli.py list --format simple

# JSON format (for scripting)
uv run python cli.py list --format json
```

#### `delete` - Delete a Key
```bash
# With confirmation prompt
uv run python cli.py delete KEY_NAME

# Skip confirmation
uv run python cli.py delete KEY_NAME --yes
```

### CLI Configuration

The CLI can be configured through:

1. **Command line options** (highest priority):
   ```bash
   uv run python cli.py --url http://localhost:8000 --api-key your-key-here list
   ```

2. **Environment variables**:
   ```bash
   export DKMS_URL="http://localhost:8000"
   export API_KEY="your-secure-api-key"
   uv run python cli.py list
   ```

3. **Defaults** (lowest priority):
   - URL: `http://127.0.0.1:8000`
   - API Key: `default-dev-key`

### CLI Examples

#### Basic Operations
```bash
# Store database credentials
uv run python cli.py set db-password --prompt
uv run python cli.py set db-host --value "postgres.example.com"
uv run python cli.py set db-user --value "myapp"

# List all stored keys
uv run python cli.py list

# Retrieve credentials (for use in scripts)
DB_HOST=$(uv run python cli.py get db-host | grep "Value:" | cut -d: -f2- | xargs)
```

#### Scripting with CLI
```bash
# Backup all keys to a file
for key in $(uv run python cli.py list --format simple); do
    value=$(uv run python cli.py get "$key" | grep "Value:" | cut -d: -f2- | xargs)
    echo "$key=$value" >> backup.txt
done

# Batch import keys
cat secrets.txt | while read line; do
    name=$(echo $line | cut -d= -f1)
    value=$(echo $line | cut -d= -f2-)
    echo "$value" | uv run python cli.py set "$name"
done
```

#### Production Usage
```bash
# Use with custom server and API key
export DKMS_URL="https://keys.company.com"
export API_KEY="prod-api-key-here"

# Store application secrets
uv run python cli.py set stripe-key --prompt
uv run python cli.py set jwt-secret --prompt

# Verify storage
uv run python cli.py list
```

### CLI Demo

Run the comprehensive CLI examples:

```bash
# Make sure the server is running first
uv run python start_server.py

# In another terminal, run the demo
uv run python cli_examples.py
```

This will demonstrate all CLI features with real-world usage patterns.

## Security Considerations

### Production Deployment

1. **Use Strong Keys**: Generate cryptographically secure API and encryption keys:
   ```bash
   # Generate a secure API key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   
   # Generate a secure encryption key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Environment Variables**: Never hardcode keys in source code. Use environment variables or secure secret management.

3. **HTTPS**: Always use HTTPS in production to protect API keys in transit.

4. **Database Security**: Consider encrypting the SQLite database file or using a more robust database with encryption at rest.

5. **Access Control**: Implement additional authentication/authorization as needed (e.g., user-based access, key-specific permissions).

### Key Rotation

To rotate the encryption key:

1. Decrypt all existing keys with the old key
2. Re-encrypt with the new key
3. Update the `CRYPTO_KEY` environment variable

Note: There's no built-in key rotation - implement this based on your requirements.

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Crypto     │    │   Database      │
│   (main.py)     │◄──►│   Manager    │    │   (SQLite)      │
│                 │    │   (crypto.py)│    │   (database.py) │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                       │                    │
         │                       │                    │
         ▼                       ▼                    ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   HTTP Bearer   │    │   Fernet     │    │   Async SQLite  │
│   Auth          │    │   Encryption │    │   Operations    │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## File Structure

```
dkms/
├── src/
│   ├── main.py           # FastAPI application and routes
│   ├── database.py       # Database operations (SQLite)
│   └── crypto.py         # Encryption/decryption utilities
├── cli.py                # Typer-based CLI client
├── cli_examples.py       # CLI usage examples and demos
├── run_cli.py           # Simple CLI runner script
├── start_server.py      # Server startup script
├── test_api.py          # API test suite
├── keys.db              # SQLite database (created at runtime)
├── pyproject.toml       # Project dependencies
└── README.md            # This file
```

## Dependencies

- **FastAPI**: Modern web framework
- **Cryptography**: Fernet encryption implementation
- **aiosqlite**: Async SQLite database operations
- **Pydantic**: Data validation and serialization
- **httpx**: HTTP client for testing and CLI communication
- **Typer**: Modern CLI framework for the command-line client
- **Rich**: Rich text and beautiful formatting for CLI output

## License

This project is provided as-is for educational and development purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues, feature requests, or questions:
- Check the FastAPI documentation: https://fastapi.tiangolo.com/
- Review the API documentation at `/docs` when the server is running
- Create an issue in the project repository