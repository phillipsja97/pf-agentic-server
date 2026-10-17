"""Runs the sports-etl CFB refresh on demand (the UI's "Ingest latest data" button).

Unlike the other sports workflows this has side effects: it spawns
`pipeline_cfb_refresh.py` from a sports-etl checkout (settings.sports_etl_path).
One run at a time; state lives in memory, so it resets on server restart.
The refresh does not refit the model — that is a separate pipeline.
"""
import asyncio
import os
import shutil
from collections import deque
from datetime import date, datetime, timezone
from typing import Optional

from config import settings
from core.logging import logger
from schemas.models import CfbIngestStatus as IngestStatus
from workflows.cfb import matchup

LOG_LINES = 40
_PREFIX = "__main__: "
_BUDGET = "extract.budget: CFBD budget: "


class IngestBusy(Exception):
    pass


class IngestUnavailable(Exception):
    pass


_status = IngestStatus()
_log: deque[str] = deque(maxlen=LOG_LINES)
_task: Optional[asyncio.Task] = None
_proc: Optional[asyncio.subprocess.Process] = None


def default_season(today: Optional[date] = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _command(year: int) -> list[str]:
    uv = shutil.which("uv") or str(os.path.expanduser("~/.local/bin/uv"))
    return [uv, "run", "python", "pipeline_cfb_refresh.py", "--year", str(year)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reset() -> None:
    global _status, _task, _proc
    _status = IngestStatus()
    _log.clear()
    _task = None
    _proc = None


def snapshot() -> IngestStatus:
    return _status.model_copy(update={"log_tail": list(_log)})


def _observe(line: str) -> None:
    _log.append(line)
    if _BUDGET in line:
        _status.budget = line.split(_BUDGET, 1)[1]
    elif _PREFIX in line:
        _status.stage = line.split(_PREFIX, 1)[1]


async def _run(year: int) -> None:
    global _proc
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    try:
        _proc = await asyncio.create_subprocess_exec(
            *_command(year),
            cwd=settings.sports_etl_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert _proc.stdout is not None
        async for raw in _proc.stdout:
            _observe(raw.decode(errors="replace").rstrip())
        code = await _proc.wait()
        _status.exit_code = code
        _status.state = "completed" if code == 0 else "failed"
        if code == 0:
            matchup.invalidate_cache()
    except asyncio.CancelledError:
        _status.state = "failed"
        raise
    except Exception as e:  # spawn failure (uv missing, bad cwd)
        _log.append(f"could not run ingest: {e}")
        _status.state = "failed"
    finally:
        _status.finished_at = _now()
        logger.info(f"cfb ingest {_status.state}  year={year}  exit={_status.exit_code}")


def start(year: Optional[int] = None) -> IngestStatus:
    global _task, _status
    if _status.state == "running":
        raise IngestBusy("An ingest is already running")
    if not (settings.sports_etl_path / "pipeline_cfb_refresh.py").is_file():
        raise IngestUnavailable(
            f"sports-etl not found at {settings.sports_etl_path} — set SPORTS_ETL_PATH on this server"
        )
    year = year or default_season()
    _log.clear()
    _status = IngestStatus(state="running", year=year, started_at=_now())
    _task = asyncio.create_task(_run(year))
    logger.info(f"cfb ingest started  year={year}")
    return snapshot()


async def wait() -> None:
    if _task is not None:
        await asyncio.gather(_task, return_exceptions=True)


def cancel() -> None:
    if _proc is not None and _proc.returncode is None:
        _proc.terminate()
