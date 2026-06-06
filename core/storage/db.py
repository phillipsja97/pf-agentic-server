import aiosqlite
from config import settings
from core.logging import logger


async def init_db() -> None:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                input       TEXT,
                result      TEXT,
                error       TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        await db.commit()
    logger.info(f"SQLite ready  path={settings.sqlite_path}")


def get_db() -> aiosqlite.Connection:
    return aiosqlite.connect(settings.sqlite_path)
