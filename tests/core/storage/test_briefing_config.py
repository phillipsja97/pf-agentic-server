import json
import pytest
from pathlib import Path

from schemas.models import BriefingConfig, StockEntry
from core.storage.briefing_config import load_config, save_config


def test_load_config_creates_defaults_when_missing(config_path):
    assert not config_path.exists()
    config = load_config(config_path)
    assert config_path.exists()
    assert config.weather_location == "Nashville, Tennessee"
    assert config.news_subjects == []
    assert config.stocks == []
    assert config.stock_mode == "ticker-view"


def test_load_config_reads_existing_file(config_path):
    data = BriefingConfig(
        weather_location="Austin, Texas",
        news_subjects=["AI", "Tech"],
    )
    save_config(config_path, data)

    loaded = load_config(config_path)
    assert loaded.weather_location == "Austin, Texas"
    assert loaded.news_subjects == ["AI", "Tech"]


def test_save_config_writes_valid_json(config_path):
    config = BriefingConfig(weather_location="Chicago, Illinois")
    save_config(config_path, config)

    raw = json.loads(config_path.read_text())
    assert raw["weather_location"] == "Chicago, Illinois"


def test_save_config_creates_parent_dirs(tmp_path):
    nested_path = tmp_path / "a" / "b" / "config.json"
    config = BriefingConfig()
    save_config(nested_path, config)
    assert nested_path.exists()


def test_roundtrip_preserves_stocks(config_path):
    config = BriefingConfig(
        stocks=[StockEntry(ticker="AAPL", name="Apple Inc.", notes="flagship")]
    )
    save_config(config_path, config)
    loaded = load_config(config_path)
    assert len(loaded.stocks) == 1
    assert loaded.stocks[0].ticker == "AAPL"
    assert loaded.stocks[0].notes == "flagship"
