import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from core.storage.db import get_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_job(
    job_type: str, input_data: dict, user_id: Optional[str] = None
) -> str:
    job_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO jobs (id, user_id, type, status, input, created_at, updated_at)"
            " VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            (job_id, user_id, job_type, json.dumps(input_data), _now(), _now()),
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

    if status in ("completed", "failed"):
        asyncio.create_task(_notify_job_done(job_id, status, error))


async def _notify_job_done(job_id: str, status: str, error: Optional[str]) -> None:
    try:
        job = await get_job(job_id)
        if not job or not job.get("user_id"):
            return
        from core.storage.users import get_push_tokens_for_user
        from core.push import send_push_notifications

        tokens = await get_push_tokens_for_user(job["user_id"])
        if not tokens:
            return

        job_type = job.get("type", "workflow")
        if status == "completed":
            title = f"{job_type} completed"
            body = f"Your {job_type} job finished."
        else:
            msg = error or "unknown error"
            title = f"{job_type} failed"
            body = f"Your {job_type} job failed: {msg}"

        await send_push_notifications(tokens, title, body, job_id)
    except Exception:
        pass


async def get_job(job_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        if user_id is not None:
            async with db.execute(
                "SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user_id)
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with db.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
    if row is None:
        return None
    record = dict(row)
    if record.get("input"):
        record["input"] = json.loads(record["input"])
    if record.get("result"):
        record["result"] = json.loads(record["result"])
    return record


async def list_jobs(
    job_type: Optional[str] = None,
    limit: int = 50,
    user_id: Optional[str] = None,
) -> list[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        if user_id is not None and job_type:
            async with db.execute(
                "SELECT * FROM jobs WHERE user_id=? AND type=? ORDER BY created_at DESC LIMIT ?",
                (user_id, job_type, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        elif user_id is not None:
            async with db.execute(
                "SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        elif job_type:
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
