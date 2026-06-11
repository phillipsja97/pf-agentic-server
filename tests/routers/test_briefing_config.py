import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from main import app
from config import settings


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "briefing_config_path", tmp_path / "config.json")


@pytest.fixture
def client():
    return TestClient(app)


def test_get_config_returns_defaults(client):
    r = client.get("/briefing/config")
    assert r.status_code == 200
    data = r.json()
    assert data["weather_location"] == "Nashville, Tennessee"
    assert data["stock_mode"] == "ticker-view"
    assert data["news_subjects"] == []


def test_update_weather_location(client):
    r = client.put("/briefing/config/weather", json={"location": "Austin, Texas"})
    assert r.status_code == 200
    assert r.json()["weather_location"] == "Austin, Texas"

    r2 = client.get("/briefing/config")
    assert r2.json()["weather_location"] == "Austin, Texas"


def test_update_stock_mode(client):
    r = client.put("/briefing/config/stock-mode", json={"mode": "portfolio-view"})
    assert r.status_code == 200
    assert r.json()["stock_mode"] == "portfolio-view"


def test_update_stock_mode_invalid_value(client):
    r = client.put("/briefing/config/stock-mode", json={"mode": "invalid-mode"})
    assert r.status_code == 422


def test_add_news_subjects(client):
    r = client.post("/briefing/config/news-subjects", json={"subjects": ["AI", "Tech"]})
    assert r.status_code == 200
    assert set(r.json()["news_subjects"]) == {"AI", "Tech"}


def test_add_news_subjects_deduplicates(client):
    client.post("/briefing/config/news-subjects", json={"subjects": ["AI"]})
    r = client.post("/briefing/config/news-subjects", json={"subjects": ["AI", "Finance"]})
    assert r.json()["news_subjects"].count("AI") == 1
    assert "Finance" in r.json()["news_subjects"]


def test_remove_news_subject(client):
    client.post("/briefing/config/news-subjects", json={"subjects": ["AI", "Tech"]})
    r = client.delete("/briefing/config/news-subjects/AI")
    assert r.status_code == 200
    assert "AI" not in r.json()["news_subjects"]
    assert "Tech" in r.json()["news_subjects"]


def test_add_rss_feeds(client):
    r = client.post(
        "/briefing/config/rss-feeds",
        json={"feeds": ["https://example.com/feed.xml"]},
    )
    assert r.status_code == 200
    assert "https://example.com/feed.xml" in r.json()["rss_feeds"]


def test_add_rss_feeds_deduplicates(client):
    url = "https://example.com/feed.xml"
    client.post("/briefing/config/rss-feeds", json={"feeds": [url]})
    r = client.post("/briefing/config/rss-feeds", json={"feeds": [url]})
    assert r.json()["rss_feeds"].count(url) == 1


def test_remove_rss_feed(client):
    url = "https://example.com/feed.xml"
    client.post("/briefing/config/rss-feeds", json={"feeds": [url]})
    r = client.request("DELETE", "/briefing/config/rss-feeds", json={"feed": url})
    assert r.status_code == 200
    assert url not in r.json()["rss_feeds"]


def test_add_stocks(client):
    r = client.post(
        "/briefing/config/stocks",
        json={"stocks": [{"ticker": "AAPL", "name": "Apple Inc."}]},
    )
    assert r.status_code == 200
    tickers = [s["ticker"] for s in r.json()["stocks"]]
    assert "AAPL" in tickers


def test_add_stocks_deduplicates_by_ticker(client):
    payload = {"stocks": [{"ticker": "AAPL", "name": "Apple Inc."}]}
    client.post("/briefing/config/stocks", json=payload)
    r = client.post("/briefing/config/stocks", json=payload)
    assert len([s for s in r.json()["stocks"] if s["ticker"] == "AAPL"]) == 1


def test_remove_stock(client):
    client.post(
        "/briefing/config/stocks",
        json={"stocks": [{"ticker": "AAPL", "name": "Apple Inc."}, {"ticker": "SPY", "name": "S&P 500 ETF"}]},
    )
    r = client.delete("/briefing/config/stocks/AAPL")
    assert r.status_code == 200
    tickers = [s["ticker"] for s in r.json()["stocks"]]
    assert "AAPL" not in tickers
    assert "SPY" in tickers


def test_remove_stock_case_insensitive(client):
    client.post(
        "/briefing/config/stocks",
        json={"stocks": [{"ticker": "AAPL", "name": "Apple Inc."}]},
    )
    r = client.delete("/briefing/config/stocks/aapl")
    assert r.status_code == 200
    assert all(s["ticker"] != "AAPL" for s in r.json()["stocks"])


def test_trigger_briefing_creates_job(client, monkeypatch):
    # Patch run_briefing to a no-op so the test doesn't need a live LLM
    import workflows.effgen.briefing as briefing_module

    async def _noop(job_id: str):
        pass

    monkeypatch.setattr(briefing_module, "run_briefing", _noop)

    r = client.post("/workflows/briefing")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert job_id

    r2 = client.get(f"/workflows/{job_id}")
    assert r2.status_code == 200
    assert r2.json()["type"] == "briefing"
    assert r2.json()["status"] == "pending"
