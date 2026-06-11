from pathlib import Path

from schemas.models import BriefingConfig


def load_config(path: Path) -> BriefingConfig:
    if not path.exists():
        config = BriefingConfig()
        save_config(path, config)
        return config
    return BriefingConfig.model_validate_json(path.read_text())


def save_config(path: Path, config: BriefingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))
