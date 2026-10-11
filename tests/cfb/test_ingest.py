import asyncio
import sys

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app
from workflows.cfb import ingest


@pytest.fixture(autouse=True)
def reset_state():
    ingest.reset()
    yield
    ingest.reset()


def _etl_dir(tmp_path, script: str):
    (tmp_path / "pipeline_cfb_refresh.py").write_text(script)
    return tmp_path


def _command(monkeypatch):
    # Run the stub with the test interpreter instead of `uv run python`.
    monkeypatch.setattr(ingest, "_command", lambda year: [sys.executable, "pipeline_cfb_refresh.py", "--year", str(year)])


def test_default_season_rolls_over_in_july():
    from datetime import date
    assert ingest.default_season(date(2026, 10, 11)) == 2026
    assert ingest.default_season(date(2027, 1, 10)) == 2026   # bowl season belongs to the prior year
    assert ingest.default_season(date(2027, 7, 1)) == 2027


def test_run_success_tracks_stage_and_invalidates_cache(tmp_path, monkeypatch):
    _command(monkeypatch)
    monkeypatch.setattr(ingest.settings, "sports_etl_path", _etl_dir(tmp_path, (
        "import sys\n"
        "print('INFO extract.budget: CFBD budget: 650 calls left, job needs ~60, 590 after', flush=True)\n"
        "print('INFO __main__: Refreshing 2026: games', flush=True)\n"
        "print('INFO __main__: Rebuilding profile', flush=True)\n"
    )))
    async def go():
        ingest.start(2026)
        await ingest.wait()

    with patch("workflows.cfb.ingest.matchup.invalidate_cache") as inv:
        asyncio.run(go())
    s = ingest.snapshot()
    assert s.state == "completed"
    assert s.year == 2026
    assert s.exit_code == 0
    assert s.stage == "Rebuilding profile"
    assert s.budget == "650 calls left, job needs ~60, 590 after"
    assert s.finished_at is not None
    assert any("Refreshing 2026: games" in line for line in s.log_tail)
    inv.assert_called_once()


def test_run_failure_reports_exit_code_and_keeps_cache(tmp_path, monkeypatch):
    _command(monkeypatch)
    monkeypatch.setattr(ingest.settings, "sports_etl_path", _etl_dir(tmp_path, (
        "import sys\nprint('boom', flush=True)\nsys.exit(3)\n"
    )))
    async def go():
        ingest.start(2026)
        await ingest.wait()

    with patch("workflows.cfb.ingest.matchup.invalidate_cache") as inv:
        asyncio.run(go())
    s = ingest.snapshot()
    assert s.state == "failed"
    assert s.exit_code == 3
    assert "boom" in s.log_tail[-1]
    inv.assert_not_called()


def test_second_start_while_running_is_rejected(tmp_path, monkeypatch):
    _command(monkeypatch)
    monkeypatch.setattr(ingest.settings, "sports_etl_path", _etl_dir(tmp_path, "import time\ntime.sleep(5)\n"))
    async def go():
        ingest.start(2026)
        with pytest.raises(ingest.IngestBusy):
            ingest.start(2026)
        await asyncio.sleep(0.3)   # let the child spawn so cancel() has a process to stop
        ingest.cancel()
        await ingest.wait()

    asyncio.run(go())


def test_start_without_etl_checkout_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.settings, "sports_etl_path", tmp_path / "missing")
    with pytest.raises(ingest.IngestUnavailable):
        ingest.start(2026)


def test_router_status_idle_then_post_busy_and_unavailable(tmp_path, monkeypatch):
    client = TestClient(app)
    r = client.get("/workflows/cfb/ingest")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"

    monkeypatch.setattr(ingest.settings, "sports_etl_path", tmp_path / "missing")
    r = client.post("/workflows/cfb/ingest", json={})
    assert r.status_code == 503
    assert "sports-etl" in r.json()["error"]

    with patch("routers.workflows.start_cfb_ingest", side_effect=ingest.IngestBusy("already running")):
        r = client.post("/workflows/cfb/ingest", json={})
    assert r.status_code == 409


def test_router_post_returns_running_status():
    client = TestClient(app)
    running = ingest.IngestStatus(state="running", year=2026, started_at="2026-10-11T12:00:00+00:00")
    with patch("routers.workflows.start_cfb_ingest", return_value=running) as start:
        r = client.post("/workflows/cfb/ingest", json={"year": 2026})
    assert r.status_code == 202
    assert r.json()["state"] == "running"
    start.assert_called_once_with(2026)
