#!/usr/bin/env python3
"""
DKMS Example Usage Script

This script demonstrates how to use the DKMS (Distributed Key Management Service)
API to store, retrieve, and manage encrypted keys and secrets.

Make sure to start the DKMS server before running this script:
    uv run python start_server.py

Then run this example in another terminal:
    uv run python example_usage.py
"""

import asyncio
import httpx
import os
from typing import Any

# Configuration
BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("API_KEY", "default-dev-key")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

class DKMSClient:
    """A simple client for the DKMS API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def store_key(self, name: str, value: str) -> dict[str, Any]:
        """Store a key in DKMS."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/key/{name}",
                json={"value": value},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

    async def get_key(self, name: str) -> str:
        """Retrieve a key from DKMS."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/key/{name}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["value"]

    async def list_keys(self) -> list[str]:
        """List all keys in DKMS."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/keys",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["keys"]

    async def delete_key(self, name: str) -> bool:
        """Delete a key from DKMS."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/key/{name}",
                headers=self.headers
            )
            return response.status_code == 200

    async def key_exists(self, name: str) -> bool:
        """Check if a key exists in DKMS."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/key/{name}",
                headers=self.headers
            )
            return response.status_code == 200

async def example_basic_usage():
    """Demonstrate basic key management operations."""
    print("🔐 Basic Key Management Example")
    print("=" * 50)

    dkms = DKMSClient(BASE_URL, API_KEY)

    # Store some example keys
    keys_to_store = {
        "database_password": "super-secret-db-password-123!",
        "api_token": "sk-1234567890abcdefghijklmnop",
        "encryption_key": "base64-encoded-key-data-here",
        "oauth_secret": "oauth2-client-secret-value"
    }

    print("\n1. Storing keys...")
    for key_name, key_value in keys_to_store.items():
        try:
            _ = await dkms.store_key(key_name, key_value)
            print(f"   ✅ Stored: {key_name}")
        except Exception as e:
            print(f"   ❌ Failed to store {key_name}: {e}")

    print("\n2. Listing all keys...")
    try:
        all_keys = await dkms.list_keys()
        print(f"   📋 Keys in DKMS: {all_keys}")
    except Exception as e:
        print(f"   ❌ Failed to list keys: {e}")

    print("\n3. Retrieving keys...")
    for key_name in keys_to_store.keys():
        try:
            retrieved_value = await dkms.get_key(key_name)
            original_value = keys_to_store[key_name]
            if retrieved_value == original_value:
                print(f"   ✅ Retrieved {key_name}: Value matches ✓")
            else:
                print(f"   ❌ Retrieved {key_name}: Value mismatch!")
        except Exception as e:
            print(f"   ❌ Failed to retrieve {key_name}: {e}")

    print("\n4. Testing key existence...")
    test_keys = ["database_password", "nonexistent_key"]
    for key_name in test_keys:
        try:
            exists = await dkms.key_exists(key_name)
            print(f"   🔍 Key '{key_name}' exists: {exists}")
        except Exception as e:
            print(f"   ❌ Error checking {key_name}: {e}")

async def example_application_config():
    """Demonstrate managing application configuration."""
    print("\n\n🔧 Application Configuration Example")
    print("=" * 50)

    dkms = DKMSClient(BASE_URL, API_KEY)

    # Application configuration that should be kept secret
    app_config = {
        "jwt_secret": "jwt-signing-secret-key-2024",
        "stripe_secret": "sk_test_123456789",
        "sendgrid_api_key": "SG.abcdefghijk.xyz",
        "redis_url": "redis://user:password@redis-server:6379/0",
        "postgres_url": "postgresql://user:pass@db:5432/myapp"
    }

    print("\n1. Storing application secrets...")
    for config_key, config_value in app_config.items():
        try:
            await dkms.store_key(f"app.{config_key}", config_value)
            print(f"   ✅ Stored app.{config_key}")
        except Exception as e:
            print(f"   ❌ Failed to store app.{config_key}: {e}")

    print("\n2. Application startup - loading configuration...")
    loaded_config: dict[str, str] = {}

    for config_key in app_config.keys():
        try:
            loaded_config[config_key] = await dkms.get_key(f"app.{config_key}")
            print(f"   📥 Loaded {config_key}")
        except Exception as e:
            print(f"   ❌ Failed to load {config_key}: {e}")

    print("\n3. Verifying loaded configuration...")
    all_match = True
    for key, original_value in app_config.items():
        if loaded_config.get(key) == original_value:
            print(f"   ✅ {key}: Configuration matches")
        else:
            print(f"   ❌ {key}: Configuration mismatch!")
            all_match = False

    if all_match:
        print("\n   🎉 All application secrets loaded successfully!")
    else:
        print("\n   ⚠️ Some configuration values don't match!")

async def example_key_rotation():
    """Demonstrate key rotation scenario."""
    print("\n\n🔄 Key Rotation Example")
    print("=" * 50)

    dkms = DKMSClient(BASE_URL, API_KEY)

    key_name = "api_key_v1"

    print("\n1. Initial key deployment...")
    try:
        await dkms.store_key(key_name, "api-key-version-1-secret")
        print(f"   ✅ Deployed initial key: {key_name}")
    except Exception as e:
        print(f"   ❌ Failed to deploy initial key: {e}")
        return

    print("\n2. Application using the key...")
    try:
        current_key = await dkms.get_key(key_name)
        print(f"   🔑 Current key in use: {current_key[:10]}...")
    except Exception as e:
        print(f"   ❌ Failed to retrieve key: {e}")
        return

    print("\n3. Key rotation - updating to new version...")
    try:
        await dkms.store_key(key_name, "api-key-version-2-updated-secret")
        print(f"   🔄 Rotated key: {key_name}")
    except Exception as e:
        print(f"   ❌ Failed to rotate key: {e}")
        return

    print("\n4. Application picks up new key...")
    try:
        rotated_key = await dkms.get_key(key_name)
        print(f"   🔑 New key in use: {rotated_key[:10]}...")

        if rotated_key == "api-key-version-2-updated-secret":
            print("   ✅ Key rotation successful!")
        else:
            print("   ❌ Key rotation failed - value mismatch!")

    except Exception as e:
        print(f"   ❌ Failed to retrieve rotated key: {e}")

async def example_multi_environment():
    """Demonstrate multi-environment key management."""
    print("\n\n🌍 Multi-Environment Example")
    print("=" * 50)

    dkms = DKMSClient(BASE_URL, API_KEY)

    # Environment-specific configurations
    environments = {
        "dev": {
            "database_url": "postgresql://dev_user:dev_pass@dev-db:5432/myapp_dev",
            "api_base_url": "https://api-dev.example.com",
            "debug_mode": "true"
        },
        "staging": {
            "database_url": "postgresql://staging_user:staging_pass@staging-db:5432/myapp_staging",
            "api_base_url": "https://api-staging.example.com",
            "debug_mode": "false"
        },
        "prod": {
            "database_url": "postgresql://prod_user:prod_pass@prod-db:5432/myapp_prod",
            "api_base_url": "https://api.example.com",
            "debug_mode": "false"
        }
    }

    print("\n1. Storing environment-specific configurations...")
    for env_name, env_config in environments.items():
        print(f"\n   Setting up {env_name.upper()} environment:")
        for config_key, config_value in env_config.items():
            key_name = f"{env_name}.{config_key}"
            try:
                await dkms.store_key(key_name, config_value)
                print(f"     ✅ {key_name}")
            except Exception as e:
                print(f"     ❌ {key_name}: {e}")

    print("\n2. Loading configuration for PRODUCTION environment...")
    prod_config: dict[str, str] = {}
    for config_key in environments["prod"].keys():
        key_name = f"prod.{config_key}"
        try:
            prod_config[config_key] = await dkms.get_key(key_name)
            print(f"   📥 Loaded {config_key}")
        except Exception as e:
            print(f"   ❌ Failed to load {config_key}: {e}")

    print("\n3. Production configuration summary:")
    for key, value in prod_config.items():
        # Mask sensitive values
        if "pass" in key.lower() or "secret" in key.lower():
            display_value = "*" * 8
        else:
            display_value = value
        print(f"   {key}: {display_value}")

async def cleanup_example_keys():
    """Clean up keys created during examples."""
    print("\n\n🧹 Cleaning up example keys...")

    dkms = DKMSClient(BASE_URL, API_KEY)

    try:
        all_keys = await dkms.list_keys()
        example_keys = [key for key in all_keys if any(prefix in key for prefix in
                       ["database_password", "api_token", "encryption_key", "oauth_secret",
                        "app.", "api_key_v1", "dev.", "staging.", "prod."])]

        if not example_keys:
            print("   No example keys to clean up.")
            return

        print(f"   Found {len(example_keys)} example keys to clean up:")
        for key in example_keys:
            try:
                success = await dkms.delete_key(key)
                if success:
                    print(f"     ✅ Deleted: {key}")
                else:
                    print(f"     ❌ Failed to delete: {key}")
            except Exception as e:
                print(f"     ❌ Error deleting {key}: {e}")

    except Exception as e:
        print(f"   ❌ Failed to clean up: {e}")

async def main():
    """Run all examples."""
    print("🎯 DKMS Usage Examples")
    print("=" * 80)
    print("Make sure the DKMS server is running on http://127.0.0.1:8000")
    print("=" * 80)

    # Check if server is accessible
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/keys", headers=HEADERS)
            if response.status_code == 200:
                print("✅ DKMS server is accessible")
            else:
                print(f"❌ DKMS server returned status {response.status_code}")
                print("Please start the server with: uv run python start_server.py")
                return
    except Exception as e:
        print(f"❌ Cannot connect to DKMS server: {e}")
        print("Please start the server with: uv run python start_server.py")
        return

    try:
        # Run all examples
        await example_basic_usage()
        await example_application_config()
        await example_key_rotation()
        await example_multi_environment()

        # Clean up
        await cleanup_example_keys()

        print("\n" + "=" * 80)
        print("🎉 All examples completed successfully!")
        print("=" * 80)
        print("\n📚 What you learned:")
        print("• How to store and retrieve encrypted keys")
        print("• Managing application configuration securely")
        print("• Implementing key rotation workflows")
        print("• Handling multi-environment deployments")
        print("• Using the DKMS client programmatically")
        print("\n💡 Next steps:")
        print("• Integrate DKMS into your application")
        print("• Set strong API_KEY and CRYPTO_KEY in production")
        print("• Consider using HTTPS for production deployments")
        print("• Implement proper error handling and retries")

    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Examples failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
