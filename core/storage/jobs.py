import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from core.storage.db import get_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_job(job_type: str, input_data: dict) -> str:
    job_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO jobs (id, type, status, input, created_at, updated_at) VALUES (?, ?, 'pending', ?, ?, ?)",
            (job_id, job_type, json.dumps(input_data), _now(), _now()),
        )
        await db.commit()
    return job_id


async def update_job(
    job_id: str,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    now = _now()
    async with get_db() as db:
        if result is not None:
            await db.execute(
                "UPDATE jobs SET status=?, result=?, updated_at=? WHERE id=?",
                (status, json.dumps(result), now, job_id),
            )
        elif error is not None:
            await db.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                (status, error, now, job_id),
            )
        else:
            await db.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                (status, now, job_id),
            )
        await db.commit()


async def get_job(job_id: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    record = dict(row)
    if record.get("input"):
        record["input"] = json.loads(record["input"])
    if record.get("result"):
        record["result"] = json.loads(record["result"])
    return record


async def list_jobs(job_type: Optional[str] = None, limit: int = 50) -> list[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        if job_type:
            async with db.execute(
                "SELECT * FROM jobs WHERE type=? ORDER BY created_at DESC LIMIT ?",
                (job_type, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
    records = []
    for row in rows:
        r = dict(row)
        if r.get("input"):
            r["input"] = json.loads(r["input"])
        if r.get("result"):
            r["result"] = json.loads(r["result"])
        records.append(r)
    return records
