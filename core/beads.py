import asyncio
import json
import shutil
import subprocess

from core.logging import logger

_bd_available: bool | None = None


def _check_bd() -> bool:
    global _bd_available
    if _bd_available is None:
        _bd_available = shutil.which("bd") is not None
        if not _bd_available:
            logger.info("bd not found on PATH — beads memory disabled")
    return _bd_available


async def memories() -> str:
    """Return stored project memories as a compact prompt prefix.

    Calls `bd memories --json` and formats the result as a short bullet list.
    Returns an empty string if bd is not installed or no memories exist.
    """
    if not _check_bd():
        return ""
    try:
        r = await asyncio.to_thread(
            subprocess.run,
            ["bd", "memories", "--json", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return ""
        data = json.loads(r.stdout)
        data.pop("schema_version", None)
        if not data:
            return ""
        lines = ["[Project memory]"]
        for val in data.values():
            lines.append(f"- {val}")
        return "\n".join(lines) + "\n\n"
    except Exception as e:
        logger.debug(f"bd memories failed  error={e}")
        return ""


async def remember(insight: str) -> None:
    """Persist a workflow insight to beads project memory for future runs."""
    if not _check_bd():
        return
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["bd", "remember", insight, "-q"],
            capture_output=True, text=True, timeout=5,
        )
        logger.debug(f"bd remember saved  insight={insight!r}")
    except Exception as e:
        logger.debug(f"bd remember failed  error={e}")
