#!/usr/bin/env python3
"""
DKMS (Distributed Key Management Service) Startup Script

This script starts the FastAPI server for the key management service.
"""

import os
import sys
import uvicorn

def main():
    """Start the DKMS server."""

    # Get environment variables with defaults
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "16724"))
    api_key = os.getenv("API_KEY", "default-dev-key")
    crypto_key = os.getenv("CRYPTO_KEY", "default-crypto-key")

    # Print startup information
    print("🔐 Starting DKMS (Distributed Key Management Service)")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"API Key: {'***' + api_key[-4:] if len(api_key) > 4 else '***'}")
    print(f"Crypto Key: {'***' + crypto_key[-4:] if len(crypto_key) > 4 else '***'}")
    print("=" * 60)

    # Warn about default keys in production
    if api_key == "default-dev-key":
        print("⚠️  WARNING: Using default API key! Set API_KEY environment variable.")
    if crypto_key == "default-crypto-key":
        print("⚠️  WARNING: Using default crypto key! Set CRYPTO_KEY environment variable.")

    if api_key == "default-dev-key" or crypto_key == "default-crypto-key":
        print("   These defaults should only be used for development!")
        print()

    print(f"🚀 Server will be available at: http://{host}:{port}")
    print("📖 API documentation at: http://{host}:{port}/docs")
    print("🔄 Alternative docs at: http://{host}:{port}/redoc")
    print()
    print("Press CTRL+C to stop the server")
    print()

    try:
        # Start the server
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
