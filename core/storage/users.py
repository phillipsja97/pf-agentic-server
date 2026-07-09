import uuid
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from core.storage.db import get_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_user(email: str, hashed_password: str) -> dict:
    user_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        await db.execute(
            "INSERT INTO users (id, email, password, created_at) VALUES (?, ?, ?, ?)",
            (user_id, email, hashed_password, now),
        )
        await db.commit()
    return {"id": user_id, "email": email, "created_at": now}


async def get_user_by_email(email: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE email=?", (email,)) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row else None


async def get_user_by_id(user_id: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE id=?", (user_id,)) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row else None


async def create_refresh_token(token: str, user_id: str, expires_at: str) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO refresh_tokens (token, user_id, expires_at, revoked) VALUES (?, ?, ?, 0)",
            (token, user_id, expires_at),
        )
        await db.commit()


async def get_refresh_token(token: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM refresh_tokens WHERE token=?", (token,)) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row else None


async def revoke_refresh_token(token: str) -> None:
    async with get_db() as db:
        await db.execute("UPDATE refresh_tokens SET revoked=1 WHERE token=?", (token,))
        await db.commit()


async def upsert_push_token(user_id: str, token: str) -> None:
    token_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO push_tokens (id, user_id, token, created_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(token) DO UPDATE SET user_id=excluded.user_id, created_at=excluded.created_at""",
            (token_id, user_id, token, now),
        )
        await db.commit()


async def get_push_tokens_for_user(user_id: str) -> list[str]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT token FROM push_tokens WHERE user_id=?", (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [row["token"] for row in rows]
