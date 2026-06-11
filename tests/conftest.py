import pytest
from pathlib import Path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "briefing_config.json"
