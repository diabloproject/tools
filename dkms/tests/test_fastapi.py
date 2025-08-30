from fastapi.testclient import TestClient
import os
import asyncio
from ..src.main import app
from ..src.database import init_database

# Set test environment variables
os.environ["API_KEY"] = "test-api-key"
os.environ["CRYPTO_KEY"] = "test-crypto-key"

client = TestClient(app)

def setup_database():
    """Set up the database for testing."""
    asyncio.run(init_database())

def test_list_keys_empty():
    """Test listing keys when database is empty."""
    response = client.get("/keys", headers={"Authorization": "Bearer test-api-key"})
    assert response.status_code == 200
    data = response.json()
    assert "keys" in data
    assert isinstance(data["keys"], list)

def test_unauthorized_access():
    """Test that invalid API key returns 401."""
    response = client.get("/keys", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]

def test_missing_auth_header():
    """Test that missing auth header returns 403."""
    response = client.get("/keys")
    assert response.status_code == 403

def test_store_and_retrieve_key():
    """Test storing and retrieving a key."""
    # Store a key
    key_data = {"value": "my-test-secret"}
    response = client.post(
        "/key/test-key",
        json=key_data,
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test-key"
    assert data["value"] == "my-test-secret"

    # Retrieve the key
    response = client.get(
        "/key/test-key",
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test-key"
    assert data["value"] == "my-test-secret"

def test_retrieve_nonexistent_key():
    """Test retrieving a key that doesn't exist."""
    response = client.get(
        "/key/nonexistent",
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_store_multiple_keys_and_list():
    """Test storing multiple keys and listing them."""
    # Store first key
    response = client.post(
        "/key/api-token",
        json={"value": "sk-1234567890"},
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200

    # Store second key
    response = client.post(
        "/key/db-password",
        json={"value": "super-secret-db-pass"},
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200

    # List all keys
    response = client.get("/keys", headers={"Authorization": "Bearer test-api-key"})
    assert response.status_code == 200
    data = response.json()
    keys = data["keys"]
    assert "api-token" in keys
    assert "db-password" in keys
    assert len(keys) >= 2

def test_delete_key():
    """Test deleting a key."""
    # First store a key
    response = client.post(
        "/key/temp-key",
        json={"value": "temporary-secret"},
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200

    # Delete the key
    response = client.delete(
        "/key/temp-key",
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]

    # Verify it's gone
    response = client.get(
        "/key/temp-key",
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 404

def test_delete_nonexistent_key():
    """Test deleting a key that doesn't exist."""
    response = client.delete(
        "/key/does-not-exist",
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 404

def test_encryption_integrity():
    """Test that encryption/decryption maintains data integrity."""
    test_values = [
        "simple-password",
        "complex-p@$$w0rd!#$%",
        "unicode-测试-🔐",
        "very-long-" + "x" * 1000,
        "",  # edge case: empty string
        "newlines\nand\ttabs\r\nspecial chars"
    ]

    for i, value in enumerate(test_values):
        key_name = f"integrity-test-{i}"

        # Store
        response = client.post(
            f"/key/{key_name}",
            json={"value": value},
            headers={"Authorization": "Bearer test-api-key"}
        )
        assert response.status_code == 200

        # Retrieve and verify
        response = client.get(
            f"/key/{key_name}",
            headers={"Authorization": "Bearer test-api-key"}
        )
        assert response.status_code == 200
        assert response.json()["value"] == value

        # Clean up
        client.delete(
            f"/key/{key_name}",
            headers={"Authorization": "Bearer test-api-key"}
        )

def test_key_overwrite():
    """Test that storing a key with the same name overwrites it."""
    key_name = "overwrite-test"

    # Store initial value
    response = client.post(
        f"/key/{key_name}",
        json={"value": "original-value"},
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200

    # Overwrite with new value
    response = client.post(
        f"/key/{key_name}",
        json={"value": "new-value"},
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200

    # Verify new value
    response = client.get(
        f"/key/{key_name}",
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    assert response.json()["value"] == "new-value"

    # Clean up
    client.delete(f"/key/{key_name}", headers={"Authorization": "Bearer test-api-key"})

def test_invalid_json_payload():
    """Test handling of invalid JSON payload."""
    response = client.post(
        "/key/test",
        data="invalid-json",  # Not JSON
        headers={
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json"
        }
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_missing_value_field():
    """Test handling of missing 'value' field in request."""
    response = client.post(
        "/key/test",
        json={"not_value": "test"},  # Wrong field name
        headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 422

if __name__ == "__main__":
    # Run tests without pytest
    import sys

    # Clean up any existing test database
    try:
        os.remove("keys.db")
        print("🧹 Cleaned up existing database")
    except FileNotFoundError:
        pass

    print("🧪 Running FastAPI Tests (without pytest)")
    print("=" * 50)

    # Set up database
    asyncio.run(init_database())

    # Run tests manually
    tests = [
        test_list_keys_empty,
        test_unauthorized_access,
        test_missing_auth_header,
        test_store_and_retrieve_key,
        test_retrieve_nonexistent_key,
        test_store_multiple_keys_and_list,
        test_delete_key,
        test_delete_nonexistent_key,
        test_encryption_integrity,
        test_key_overwrite,
        test_invalid_json_payload,
        test_missing_value_field
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All FastAPI tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed.")
        sys.exit(1)
