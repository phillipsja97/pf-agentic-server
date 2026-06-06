from datetime import datetime, timezone
from pathlib import Path

from config import settings
from core.logging import logger


def _vault() -> Path:
    return Path(settings.obsidian_vault_path)


def write_note(folder: str, slug: str, content: str, frontmatter: dict | None = None) -> Path:
    target_dir = _vault() / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = target_dir / f"{date_str}-{slug}.md"

    fm_lines = ["---"]
    base_meta = {"created": datetime.now(timezone.utc).isoformat(), "tags": f"agent/{folder.lower()}"}
    merged = {**base_meta, **(frontmatter or {})}
    for k, v in merged.items():
        fm_lines.append(f"{k}: {v}")
    fm_lines.append("---\n")

    filepath.write_text("\n".join(fm_lines) + content, encoding="utf-8")
    logger.info(f"Obsidian note written  path={filepath}")
    return filepath


def read_note(folder: str, filename: str) -> str:
    filepath = _vault() / folder / filename
    if not filepath.exists():
        logger.warning(f"Obsidian note not found  path={filepath}")
        return ""
    return filepath.read_text(encoding="utf-8")


def read_agent_knowledge(filename: str) -> str:
    return read_note("Agent Knowledge", filename)
