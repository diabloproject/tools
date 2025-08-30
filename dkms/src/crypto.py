import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class CryptoManager:
    def __init__(self, master_key: str):
        """Initialize the crypto manager with a master key."""
        self.master_key = master_key
        self._fernet = self._create_fernet()

    def _create_fernet(self) -> Fernet:
        """Create a Fernet cipher from the master key."""
        # Use a fixed salt for consistency (in production, consider storing this securely)
        salt = b"dkms_salt_2024"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)

    def encrypt(self, data: str) -> bytes:
        """Encrypt a string and return encrypted bytes."""
        return self._fernet.encrypt(data.encode('utf-8'))

    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt bytes and return the original string."""
        try:
            decrypted_bytes = self._fernet.decrypt(encrypted_data)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {str(e)}")

    def encrypt_dict(self, data: dict) -> bytes:
        """Encrypt a dictionary by converting it to JSON string first."""
        import json
        json_str = json.dumps(data, sort_keys=True)
        return self.encrypt(json_str)

    def decrypt_dict(self, encrypted_data: bytes) -> dict:
        """Decrypt bytes and return the original dictionary."""
        import json
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)
