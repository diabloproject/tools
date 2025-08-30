import asyncio
import httpx
import os

# Configuration
BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("API_KEY", "default-dev-key")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

async def test_api():
    """Test the DKMS API functionality."""
    async with httpx.AsyncClient() as client:
        print("🔐 Testing DKMS API")
        print("=" * 50)

        # Test 1: List keys (should be empty initially)
        print("\n1. Testing key listing...")
        response = await client.get(f"{BASE_URL}/keys", headers=HEADERS)
        if response.status_code == 200:
            keys = response.json()
            print(f"✅ Current keys: {keys}")
        else:
            print(f"❌ Failed to list keys: {response.status_code}")
            return

        # Test 2: Store a key
        print("\n2. Testing key storage...")
        test_key_name = "test-secret"
        test_key_value = "my-super-secret-password-123"

        response = await client.post(
            f"{BASE_URL}/key/{test_key_name}",
            headers=HEADERS,
            json={"value": test_key_value}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Stored key: {result}")
        else:
            print(f"❌ Failed to store key: {response.status_code} - {response.text}")
            return

        # Test 3: Retrieve the key
        print("\n3. Testing key retrieval...")
        response = await client.get(f"{BASE_URL}/key/{test_key_name}", headers=HEADERS)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved key: {result}")
            if result["value"] == test_key_value:
                print("✅ Encryption/decryption working correctly!")
            else:
                print("❌ Value mismatch after encryption/decryption")
        else:
            print(f"❌ Failed to retrieve key: {response.status_code}")

        # Test 4: Store another key
        print("\n4. Testing multiple keys...")
        response = await client.post(
            f"{BASE_URL}/key/api-token",
            headers=HEADERS,
            json={"value": "sk-1234567890abcdef"}
        )

        if response.status_code == 200:
            print("✅ Stored second key")
        else:
            print(f"❌ Failed to store second key: {response.status_code}")

        # Test 5: List keys again
        print("\n5. Testing updated key list...")
        response = await client.get(f"{BASE_URL}/keys", headers=HEADERS)
        if response.status_code == 200:
            keys = response.json()
            print(f"✅ Updated keys: {keys}")
        else:
            print(f"❌ Failed to list keys: {response.status_code}")

        # Test 6: Test authentication
        print("\n6. Testing authentication...")
        bad_headers = {"Authorization": "Bearer wrong-key"}
        response = await client.get(f"{BASE_URL}/keys", headers=bad_headers)
        if response.status_code == 401:
            print("✅ Authentication working correctly")
        else:
            print(f"❌ Authentication failed: {response.status_code}")

        # Test 7: Try to get non-existent key
        print("\n7. Testing non-existent key...")
        response = await client.get(f"{BASE_URL}/key/does-not-exist", headers=HEADERS)
        if response.status_code == 404:
            print("✅ 404 handling working correctly")
        else:
            print(f"❌ Expected 404, got: {response.status_code}")

        # Test 8: Delete a key
        print("\n8. Testing key deletion...")
        response = await client.delete(f"{BASE_URL}/key/{test_key_name}", headers=HEADERS)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Deleted key: {result}")
        else:
            print(f"❌ Failed to delete key: {response.status_code}")

        # Test 9: Verify key is deleted
        print("\n9. Verifying key deletion...")
        response = await client.get(f"{BASE_URL}/key/{test_key_name}", headers=HEADERS)
        if response.status_code == 404:
            print("✅ Key successfully deleted")
        else:
            print(f"❌ Key still exists: {response.status_code}")

        print("\n" + "=" * 50)
        print("🎉 API testing completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_api())
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
