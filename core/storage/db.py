import aiosqlite
from config import settings
from core.logging import logger


async def init_db() -> None:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                email       TEXT UNIQUE NOT NULL,
                password    TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token       TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                expires_at  TEXT NOT NULL,
                revoked     INTEGER NOT NULL DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                user_id     TEXT REFERENCES users(id),
                type        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                input       TEXT,
                result      TEXT,
                error       TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        # Migrate existing jobs table: add user_id column if missing
        async with db.execute("PRAGMA table_info(jobs)") as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        if "user_id" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN user_id TEXT REFERENCES users(id)")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS push_tokens (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                token       TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE(token)
            )
        """)
        await db.commit()
    logger.info(f"SQLite ready  path={settings.sqlite_path}")


def get_db() -> aiosqlite.Connection:
    return aiosqlite.connect(settings.sqlite_path)
