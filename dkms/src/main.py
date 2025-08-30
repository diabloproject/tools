from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager

from .database import init_database, store_key, retrieve_key, delete_key, list_keys
from .crypto import CryptoManager

class KeyRequest(BaseModel):
    value: str

class KeyResponse(BaseModel):
    name: str
    value: str

class KeyListResponse(BaseModel):
    keys: list[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(lifespan=lifespan)

security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "default-dev-key")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


CRYPTO_KEY = os.getenv("CRYPTO_KEY", "default-crypto-key")
crypto_manager = CryptoManager(CRYPTO_KEY)


@app.get("/key/{name}", response_model=KeyResponse)
async def get_key(name: str, api_key: str = Depends(verify_api_key)):
    """Retrieve and decrypt a key from the database."""
    encrypted_data = await retrieve_key(name)
    if encrypted_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key '{name}' not found"
        )

    try:
        decrypted_value = crypto_manager.decrypt(encrypted_data)
        return KeyResponse(name=name, value=decrypted_value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decrypt key"
        )


@app.post("/key/{name}", response_model=KeyResponse)
async def post_key(name: str, key_data: KeyRequest, api_key: str = Depends(verify_api_key)):
    """Encrypt and store a key in the database."""
    try:
        encrypted_data = crypto_manager.encrypt(key_data.value)
        success = await store_key(name, encrypted_data)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store key"
            )

        return KeyResponse(name=name, value=key_data.value)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encrypt and store key: {str(e)}"
        )

@app.delete("/key/{name}")
async def delete_key_endpoint(name: str, api_key: str = Depends(verify_api_key)):
    """Delete a key from the database."""
    success = await delete_key(name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key '{name}' not found"
        )
    return {"message": f"Key '{name}' deleted successfully"}

@app.get("/keys", response_model=KeyListResponse)
async def list_keys_endpoint(api_key: str = Depends(verify_api_key)):
    """List all available key names."""
    keys = await list_keys()
    return KeyListResponse(keys=keys)
