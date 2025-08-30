from typing import Optional
import aiosqlite

DATABASE_PATH = "keys.db"

async def init_database():
    """Initialize the database with the keys table."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS keys (
                name TEXT PRIMARY KEY,
                encrypted_value BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def store_key(name: str, encrypted_value: bytes) -> bool:
    """Store an encrypted key in the database."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("""
                INSERT OR REPLACE INTO keys (name, encrypted_value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (name, encrypted_value))
            await db.commit()
            return True
    except Exception:
        return False

async def retrieve_key(name: str) -> Optional[bytes]:
    """Retrieve an encrypted key from the database."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            cursor = await db.execute(
                "SELECT encrypted_value FROM keys WHERE name = ?", (name,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None
    except Exception:
        return None

async def delete_key(name: str) -> bool:
    """Delete a key from the database."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            cursor = await db.execute(
                "DELETE FROM keys WHERE name = ?", (name,)
            )
            await db.commit()
            return cursor.rowcount > 0
    except Exception:
        return False

async def list_keys() -> list[str]:
    """List all key names in the database."""
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            cursor = await db.execute("SELECT name FROM keys ORDER BY name")
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    except Exception:
        return []