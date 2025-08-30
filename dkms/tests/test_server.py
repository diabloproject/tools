import os
import asyncio

# Set test environment variables BEFORE importing main
os.environ["API_KEY"] = "test-api-key"
os.environ["CRYPTO_KEY"] = "test-crypto-key"

from fastapi.testclient import TestClient
from ..src.main import app
from ..src.database import init_database

async def setup_test_database():
    """Initialize database for testing."""
    await init_database()

def test_dkms_api():
    """Test the DKMS API with detailed error reporting."""

    print("🧪 Testing DKMS FastAPI Server")
    print("=" * 50)

    # Initialize database before testing
    print("Setting up test database...")
    asyncio.run(setup_test_database())

    # Initialize the test client
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-api-key"}

    try:
        # Test 1: Authentication
        print("\n1. Testing Authentication...")

        # Valid auth
        response = client.get("/keys", headers=headers)
        print(f"   Valid auth: {response.status_code}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Invalid auth
        bad_headers = {"Authorization": "Bearer wrong-key"}
        response = client.get("/keys", headers=bad_headers)
        print(f"   Invalid auth: {response.status_code}")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"

        # No auth
        response = client.get("/keys")
        print(f"   No auth: {response.status_code}")
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"

        print("   ✅ Authentication tests passed")

        # Test 2: List keys (empty)
        print("\n2. Testing Empty Key List...")
        response = client.get("/keys", headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert "keys" in data
        assert isinstance(data["keys"], list)
        print("   ✅ Empty key list test passed")

        # Test 3: Store a key
        print("\n3. Testing Key Storage...")
        key_data = {"value": "test-secret-123"}
        response = client.post("/key/test-key", json=key_data, headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-key"
        assert data["value"] == "test-secret-123"
        print("   ✅ Key storage test passed")

        # Test 4: Retrieve the key
        print("\n4. Testing Key Retrieval...")
        response = client.get("/key/test-key", headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-key"
        assert data["value"] == "test-secret-123"
        print("   ✅ Key retrieval test passed")

        # Test 5: List keys (with data)
        print("\n5. Testing Key List with Data...")
        response = client.get("/keys", headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert "test-key" in data["keys"]
        print("   ✅ Key list with data test passed")

        # Test 6: Store another key
        print("\n6. Testing Multiple Keys...")
        response = client.post("/key/api-token", json={"value": "sk-123456"}, headers=headers)
        print(f"   Status: {response.status_code}")
        assert response.status_code == 200
        print("   ✅ Multiple keys test passed")

        # Test 7: Retrieve non-existent key
        print("\n7. Testing Non-existent Key...")
        response = client.get("/key/does-not-exist", headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 404
        print("   ✅ Non-existent key test passed")

        # Test 8: Delete a key
        print("\n8. Testing Key Deletion...")
        response = client.delete("/key/test-key", headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Key deletion test passed")

        # Test 9: Verify deletion
        print("\n9. Verifying Key Deletion...")
        response = client.get("/key/test-key", headers=headers)
        print(f"   Status: {response.status_code}")
        assert response.status_code == 404
        print("   ✅ Key deletion verification passed")

        # Test 10: Key overwrite
        print("\n10. Testing Key Overwrite...")
        # Store original
        response = client.post("/key/overwrite-test", json={"value": "original"}, headers=headers)
        assert response.status_code == 200

        # Overwrite
        response = client.post("/key/overwrite-test", json={"value": "new-value"}, headers=headers)
        assert response.status_code == 200

        # Verify overwrite
        response = client.get("/key/overwrite-test", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["value"] == "new-value"
        print("   ✅ Key overwrite test passed")

        # Clean up
        client.delete("/key/overwrite-test", headers=headers)
        client.delete("/key/api-token", headers=headers)

        print("\n" + "=" * 50)
        print("🎉 All FastAPI tests passed!")
        print("Your DKMS server is working correctly!")

        return True

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner."""

    # Clean up any existing database
    try:
        os.remove("keys.db")
        print("🧹 Cleaned up existing database")
    except FileNotFoundError:
        pass

    # Run the tests
    success = test_dkms_api()

    if success:
        print("\n✅ Server tests completed successfully!")
        return 0
    else:
        print("\n❌ Server tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
