# Morning Briefing Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `POST /workflows/briefing` endpoint that runs an effgen agent to produce a structured morning briefing (weather, news, stocks) as a fire-and-forget async job, with config managed via `/briefing/config/*` endpoints backed by a JSON sidecar file.

**Architecture:** Single orchestrator effgen agent using the `research` preset (which includes NewsTool, HackerNewsTool, RSSFeedTool) with `extra_tools=[WeatherTool(), StockPriceTool()]`. Config lives in `data/briefing_config.json`, loaded fresh each job run. All config mutations go through `core/storage/briefing_config.py`, keeping I/O out of the router layer.

**Tech Stack:** FastAPI, effgen (`research` preset + `WeatherTool`/`StockPriceTool` from `effgen.tools.builtin`), Pydantic v2, aiosqlite (existing jobs table), pytest + FastAPI TestClient

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `config.py` | Add `briefing_config_path` setting |
| Modify | `.env.example` | Document `BRIEFING_CONFIG_PATH` |
| Modify | `schemas/models.py` | Add all briefing Pydantic models |
| Create | `core/storage/briefing_config.py` | `load_config` / `save_config` over JSON file |
| Create | `routers/briefing_config.py` | All `/briefing/config/*` endpoints |
| Modify | `main.py` | Register briefing_config router |
| Modify | `routers/workflows.py` | Add `POST /briefing` trigger |
| Create | `workflows/effgen/briefing.py` | Agent runner, prompt builder, output model |
| Create | `tests/__init__.py` | Test package marker |
| Create | `tests/conftest.py` | Shared pytest fixtures |
| Create | `tests/core/__init__.py` | Package marker |
| Create | `tests/core/storage/__init__.py` | Package marker |
| Create | `tests/core/storage/test_briefing_config.py` | Storage layer unit tests |
| Create | `tests/routers/__init__.py` | Package marker |
| Create | `tests/routers/test_briefing_config.py` | Config router integration tests |

---

## Task 1: Update Settings and .env.example

**Files:**
- Modify: `config.py`
- Modify: `.env.example`

- [ ] **Step 1: Add `briefing_config_path` to Settings**

Open `config.py`. The current content ends at line 26. Add one line inside the `Settings` class after `sqlite_path`:

```python
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    obsidian_vault_path: Path = Path("./vault")
    brain_vault_path: Path = Path("./brain")
    coding_brain_path: Path = Path.home() / ".coding-agent"
    projects_path: Path = Path.home() / "Projects"
    sqlite_path: Path = Path("./data/jobs.db")
    briefing_config_path: Path = Path("./data/briefing_config.json")

    # langfuse_secret_key: str = ""
    # langfuse_public_key: str = ""
    # langfuse_host: str = "http://localhost:3000"

    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "none"
    llm_model: str = "local"

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 2: Document in .env.example**

Add after the `SQLITE_PATH` line in `.env.example`:

```
BRIEFING_CONFIG_PATH=./data/briefing_config.json
```

- [ ] **Step 3: Verify settings load**

```bash
uv run python -c "from config import settings; print(settings.briefing_config_path)"
```

Expected output: `data/briefing_config.json`

- [ ] **Step 4: Commit**

```bash
git add config.py .env.example
git commit -m "feat: add briefing_config_path to Settings"
```

---

## Task 2: Add Briefing Schemas to models.py

**Files:**
- Modify: `schemas/models.py`

- [ ] **Step 1: Add all briefing models**

Append to `schemas/models.py` (after the existing `HealthResponse` class):

```python
from typing import Any, Literal, Optional
```

Replace the existing `from typing import Any, Optional` import at the top of the file with the line above, then append the following classes at the bottom of the file:

```python
# --- briefing config models ---

class StockEntry(BaseModel):
    ticker: str
    name: str
    notes: Optional[str] = None


class BriefingConfig(BaseModel):
    weather_location: str = "Nashville, Tennessee"
    weather_units: str = "fahrenheit"
    news_subjects: list[str] = []
    rss_feeds: list[str] = []
    stocks: list[StockEntry] = []
    stock_mode: Literal["ticker-view", "portfolio-view"] = "ticker-view"


# --- briefing config request models ---

class UpdateWeatherRequest(BaseModel):
    location: str


class UpdateStockModeRequest(BaseModel):
    mode: Literal["ticker-view", "portfolio-view"]


class AddNewsSubjectsRequest(BaseModel):
    subjects: list[str]


class AddRSSFeedsRequest(BaseModel):
    feeds: list[str]


class RemoveRSSFeedRequest(BaseModel):
    feed: str


class AddStocksRequest(BaseModel):
    stocks: list[StockEntry]


# --- briefing output models ---

class WeatherPeriod(BaseModel):
    temp_f: float
    condition: str
    summary: str


class WeatherCurrent(BaseModel):
    temp_f: float
    condition: str
    humidity: str
    wind: str


class WeatherOutput(BaseModel):
    location: str
    current: WeatherCurrent
    forecast: dict[str, WeatherPeriod]


class NewsStory(BaseModel):
    title: str
    summary: str
    source: str
    url: str


class NewsSubjectOutput(BaseModel):
    subject: str
    stories: list[NewsStory]


class TickerOutput(BaseModel):
    ticker: str
    name: str
    close: float
    change_pct: float
    summary: str


class StocksOutput(BaseModel):
    mode: Literal["ticker-view", "portfolio-view"]
    tickers: Optional[list[TickerOutput]] = None
    narrative: Optional[str] = None


class BriefingOutput(BaseModel):
    date: str
    weather: WeatherOutput
    news: Optional[list[NewsSubjectOutput]] = None
    stocks: Optional[StocksOutput] = None
```

- [ ] **Step 2: Verify models import cleanly**

```bash
uv run python -c "from schemas.models import BriefingConfig, BriefingOutput, StockEntry; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add schemas/models.py
git commit -m "feat: add briefing schemas to models.py"
```

---

## Task 3: Briefing Config Storage Layer (TDD)

**Files:**
- Create: `core/storage/briefing_config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/storage/__init__.py`
- Create: `tests/core/storage/test_briefing_config.py`

- [ ] **Step 1: Create test package structure**

```bash
mkdir -p tests/core/storage
touch tests/__init__.py tests/core/__init__.py tests/core/storage/__init__.py
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import pytest
from pathlib import Path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "briefing_config.json"
```

- [ ] **Step 3: Write failing tests**

Create `tests/core/storage/test_briefing_config.py`:

```python
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
```

- [ ] **Step 4: Run tests — verify they fail**

```bash
uv run pytest tests/core/storage/test_briefing_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.storage.briefing_config'`

- [ ] **Step 5: Implement `core/storage/briefing_config.py`**

```python
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
```

- [ ] **Step 6: Run tests — verify they pass**

```bash
uv run pytest tests/core/storage/test_briefing_config.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 7: Commit**

```bash
git add core/storage/briefing_config.py tests/
git commit -m "feat: add briefing config storage layer with tests"
```

---

## Task 4: Config Management Router (TDD)

**Files:**
- Create: `routers/briefing_config.py`
- Modify: `main.py`
- Create: `tests/routers/__init__.py`
- Create: `tests/routers/test_briefing_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/routers/__init__.py` (empty), then create `tests/routers/test_briefing_config.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/routers/test_briefing_config.py -v
```

Expected: errors like `404 Not Found` for all routes (router not registered yet)

- [ ] **Step 3: Create `routers/briefing_config.py`**

```python
from fastapi import APIRouter

from config import settings
from core.storage.briefing_config import load_config, save_config
from schemas.models import (
    AddNewsSubjectsRequest,
    AddRSSFeedsRequest,
    AddStocksRequest,
    BriefingConfig,
    RemoveRSSFeedRequest,
    UpdateStockModeRequest,
    UpdateWeatherRequest,
)

router = APIRouter(prefix="/briefing", tags=["briefing-config"])


@router.get("/config", response_model=BriefingConfig)
async def get_briefing_config() -> BriefingConfig:
    return load_config(settings.briefing_config_path)


@router.put("/config/weather", response_model=BriefingConfig)
async def update_weather(request: UpdateWeatherRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.weather_location = request.location
    save_config(settings.briefing_config_path, config)
    return config


@router.put("/config/stock-mode", response_model=BriefingConfig)
async def update_stock_mode(request: UpdateStockModeRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.stock_mode = request.mode
    save_config(settings.briefing_config_path, config)
    return config


@router.post("/config/news-subjects", response_model=BriefingConfig)
async def add_news_subjects(request: AddNewsSubjectsRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    existing = set(config.news_subjects)
    for s in request.subjects:
        if s not in existing:
            config.news_subjects.append(s)
            existing.add(s)
    save_config(settings.briefing_config_path, config)
    return config


@router.delete("/config/news-subjects/{subject}", response_model=BriefingConfig)
async def remove_news_subject(subject: str) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.news_subjects = [s for s in config.news_subjects if s != subject]
    save_config(settings.briefing_config_path, config)
    return config


@router.post("/config/rss-feeds", response_model=BriefingConfig)
async def add_rss_feeds(request: AddRSSFeedsRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    existing = set(config.rss_feeds)
    for f in request.feeds:
        if f not in existing:
            config.rss_feeds.append(f)
            existing.add(f)
    save_config(settings.briefing_config_path, config)
    return config


@router.delete("/config/rss-feeds", response_model=BriefingConfig)
async def remove_rss_feed(request: RemoveRSSFeedRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.rss_feeds = [f for f in config.rss_feeds if f != request.feed]
    save_config(settings.briefing_config_path, config)
    return config


@router.post("/config/stocks", response_model=BriefingConfig)
async def add_stocks(request: AddStocksRequest) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    existing = {s.ticker for s in config.stocks}
    for stock in request.stocks:
        if stock.ticker not in existing:
            config.stocks.append(stock)
            existing.add(stock.ticker)
    save_config(settings.briefing_config_path, config)
    return config


@router.delete("/config/stocks/{ticker}", response_model=BriefingConfig)
async def remove_stock(ticker: str) -> BriefingConfig:
    config = load_config(settings.briefing_config_path)
    config.stocks = [s for s in config.stocks if s.ticker.upper() != ticker.upper()]
    save_config(settings.briefing_config_path, config)
    return config
```

- [ ] **Step 4: Register the router in `main.py`**

Add the import and `include_router` call. The final `main.py` should look like:

```python
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from config import settings
from core.logging import logger
from core.storage.db import init_db
# from core.tracing import setup_tracing
from routers import health, workflows
from routers import briefing_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting agentic server...")
    await init_db()
    # setup_tracing()
    logger.info(f"Server ready  host={settings.host}  port={settings.port}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Agentic Workflow Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path}  status={response.status_code}  duration={duration_ms:.1f}ms"
    )
    return response


app.include_router(health.router)
app.include_router(workflows.router, prefix="/workflows")
app.include_router(briefing_config.router)
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/routers/test_briefing_config.py -v
```

Expected: 14 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add routers/briefing_config.py main.py tests/routers/
git commit -m "feat: add briefing config management router with tests"
```

---

## Task 5: Add Workflow Trigger

**Files:**
- Modify: `routers/workflows.py`

- [ ] **Step 1: Add the POST /briefing route**

Open `routers/workflows.py`. Add the import for `run_briefing` (lazy, inside the handler) and a new route. The updated file:

```python
from fastapi import APIRouter, BackgroundTasks, HTTPException

from core.logging import logger
from core.storage.jobs import create_job, get_job, list_jobs
from schemas.models import CodingRequest, JobCreatedResponse, JobStatusResponse, ResearchRequest

router = APIRouter(tags=["workflows"])


@router.post("/research", response_model=JobCreatedResponse)
async def trigger_research(
    request: ResearchRequest, background_tasks: BackgroundTasks
) -> JobCreatedResponse:
    job_id = await create_job("research", request.model_dump())
    from workflows.effgen.research import run_research
    background_tasks.add_task(run_research, job_id, request)
    logger.info(f"job {job_id} queued  workflow=research  query={request.query!r}")
    return JobCreatedResponse(job_id=job_id)


@router.post("/coding", response_model=JobCreatedResponse)
async def trigger_coding(
    request: CodingRequest, background_tasks: BackgroundTasks
) -> JobCreatedResponse:
    job_id = await create_job("coding", request.model_dump())
    from workflows.coding.app_builder import run_app_builder
    background_tasks.add_task(run_app_builder, job_id, request)
    logger.info(f"job {job_id} queued  workflow=coding  idea={request.idea[:60]!r}")
    return JobCreatedResponse(job_id=job_id)


@router.post("/briefing", response_model=JobCreatedResponse)
async def trigger_briefing(background_tasks: BackgroundTasks) -> JobCreatedResponse:
    job_id = await create_job("briefing", {})
    from workflows.effgen.briefing import run_briefing
    background_tasks.add_task(run_briefing, job_id)
    logger.info(f"job {job_id} queued  workflow=briefing")
    return JobCreatedResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_workflow_status(job_id: str) -> JobStatusResponse:
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**job)


@router.get("/", response_model=list[JobStatusResponse])
async def list_workflows(type: str | None = None, limit: int = 50) -> list[JobStatusResponse]:
    jobs = await list_jobs(job_type=type, limit=limit)
    return [JobStatusResponse(**j) for j in jobs]
```

- [ ] **Step 2: Verify server starts cleanly**

```bash
uv run uvicorn main:app --reload &
sleep 3
curl -s http://localhost:8000/docs | grep -o '"POST /workflows/briefing"' || echo "check /docs manually"
kill %1
```

Expected: server starts without errors; the `/docs` page lists `POST /workflows/briefing`

- [ ] **Step 3: Commit**

```bash
git add routers/workflows.py
git commit -m "feat: add POST /workflows/briefing trigger endpoint"
```

---

## Task 6: Briefing Agent Runner

**Files:**
- Create: `workflows/effgen/briefing.py`

- [ ] **Step 1: Create `workflows/effgen/briefing.py`**

```python
import asyncio
from datetime import date
from typing import Optional

from effgen.tools.builtin import StockPriceTool, WeatherTool

from config import settings
from core.logging import logger
from core.storage.briefing_config import load_config
from core.storage.jobs import update_job
from core.storage.obsidian import write_brain_note
from core.tracing import observe
from schemas.models import BriefingConfig, BriefingOutput


def _load_model():
    from effgen import load_model
    return load_model(
        settings.llm_model,
        provider="openai",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


_model = None
_model_lock = asyncio.Lock()


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        logger.info(f"Connecting to LLM  base_url={settings.llm_base_url}  model={settings.llm_model}")
        model = await asyncio.to_thread(_load_model)
        _model = model
        logger.info("LLM connection ready")
    return _model


def _build_prompt(config: BriefingConfig) -> str:
    today = date.today().isoformat()
    parts = [
        f"Today is {today}. Produce a complete structured morning briefing.",
        "",
        "## Phase 1: Gather",
        f"1. Fetch weather for '{config.weather_location}':",
        f"   - Call WeatherTool with operation='current', location='{config.weather_location}', units='imperial'",
        f"   - Call WeatherTool with operation='forecast', location='{config.weather_location}', days=1, units='imperial'",
    ]

    if config.news_subjects:
        subjects_str = ", ".join(f"'{s}'" for s in config.news_subjects)
        parts += [
            "",
            f"2. For each of these subjects: {subjects_str}",
            "   - Call NewsTool with operation='search', query=<subject>, max_results=2",
            "   - Call HackerNewsTool with operation='top_stories', n=20, then pick up to 2 stories relevant to <subject> (skip if none match)",
        ]
        if config.rss_feeds:
            feeds_str = ", ".join(f"'{f}'" for f in config.rss_feeds)
            parts.append(
                f"   - For each RSS feed [{feeds_str}]: call RSSFeedTool with url=<feed_url>, query=<subject>, n=2"
            )

    if config.stocks:
        tickers_str = ", ".join(f"{s.ticker} ({s.name})" for s in config.stocks)
        parts += [
            "",
            f"3. Fetch stock data for: {tickers_str}",
            "   - Call StockPriceTool once per ticker using the exact symbol string.",
        ]

    parts += [
        "",
        "## Phase 2: Synthesize",
        "Weather: organize forecast data into morning (6am–12pm), afternoon (12pm–6pm),",
        "and evening (6pm–11pm) time buckets. ALL temperatures MUST be in Fahrenheit.",
    ]

    if config.news_subjects:
        parts += [
            "News: for each subject, merge results from all tools into a single story list.",
            "If two items describe the same event, keep the one with more detail (drop the duplicate).",
        ]

    if config.stocks:
        if config.stock_mode == "ticker-view":
            parts += [
                "Stocks (ticker-view): for each ticker record yesterday's closing price,",
                "percentage change from prior close, and a 1–2 sentence commentary.",
            ]
        else:
            parts += [
                "Stocks (portfolio-view): write one narrative paragraph summarizing",
                "how the full watchlist performed together, noting standouts and overall trend.",
            ]

    parts += [
        "",
        "## Phase 3: Format",
        "Return ONLY the BriefingOutput JSON.",
        "'weather' is always present (WeatherTool always runs).",
        "Omit the 'news' key entirely if no subjects were configured.",
        "Omit the 'stocks' key entirely if no stocks were configured.",
        "The forecast dict MUST use exactly these keys: 'morning', 'afternoon', 'evening'.",
        f"Set 'date' to '{today}'.",
    ]

    return "\n".join(parts)


def _run_briefing_sync(model, config: BriefingConfig) -> tuple[BriefingOutput, int]:
    from effgen import create_agent

    extra_tools = [WeatherTool()]
    if config.stocks:
        extra_tools.append(StockPriceTool())

    agent = create_agent(
        "research",
        model,
        extra_tools=extra_tools,
        tool_calling_mode="react",
    )

    task = _build_prompt(config)
    response = agent.run(task, output_model=BriefingOutput)

    if not response.success:
        reason = (response.metadata or {}).get("reason", "unknown")
        error = (response.metadata or {}).get("error", "")
        raise RuntimeError(
            f"Agent did not succeed after {response.iterations} iterations"
            f"  reason={reason}"
            + (f"  error={error}" if error else "")
        )

    parsed: Optional[BriefingOutput] = (response.metadata or {}).get("parsed")
    if parsed is None:
        raise RuntimeError("Agent succeeded but output could not be parsed as BriefingOutput")

    return parsed, response.tokens_used


def _format_briefing_note(output: BriefingOutput) -> str:
    lines = [f"# Morning Briefing — {output.date}", ""]
    w = output.weather
    lines += [
        "## Weather", "",
        f"**Location:** {w.location}",
        f"**Now:** {w.current.temp_f}°F, {w.current.condition}, {w.current.humidity} humidity, wind {w.current.wind}",
        "",
        "| Period | Temp | Condition |",
        "|--------|------|-----------|",
    ]
    for period, data in w.forecast.items():
        lines.append(f"| {period.capitalize()} | {data.temp_f}°F | {data.condition} |")
    lines.append("")

    if output.news:
        lines += ["## News", ""]
        for subject_block in output.news:
            lines.append(f"### {subject_block.subject}")
            for story in subject_block.stories:
                lines.append(f"- [{story.title}]({story.url}) — {story.source}")
                lines.append(f"  {story.summary}")
        lines.append("")

    if output.stocks:
        lines += ["## Stocks", ""]
        if output.stocks.tickers:
            for t in output.stocks.tickers:
                sign = "+" if t.change_pct >= 0 else ""
                lines.append(f"**{t.ticker}** ({t.name}): ${t.close:.2f} ({sign}{t.change_pct:.2f}%)")
                lines.append(f"  {t.summary}")
        elif output.stocks.narrative:
            lines.append(output.stocks.narrative)

    return "\n".join(lines)


@observe(name="briefing-workflow")
async def run_briefing(job_id: str) -> None:
    logger.info(f"job {job_id} → running  workflow=briefing")
    await update_job(job_id, "running")

    try:
        config = load_config(settings.briefing_config_path)
        model = await _get_model()

        try:
            parsed, tokens_used = await asyncio.to_thread(_run_briefing_sync, model, config)
        except Exception as e:
            if "connection" in str(e).lower():
                global _model
                _model = None
                logger.warning("LLM connection lost — cleared model cache")
            raise

        note_path = write_brain_note(
            "raw/briefings",
            f"briefing-{parsed.date}",
            _format_briefing_note(parsed),
            frontmatter={"date": f'"{parsed.date}"', "type": "briefing"},
        )

        result = parsed.model_dump(exclude_none=True)
        result["tokens_used"] = tokens_used
        result["obsidian_note"] = str(note_path)

        await update_job(job_id, "completed", result=result)
        logger.info(
            f"job {job_id} → completed  workflow=briefing  tokens={tokens_used}"
        )

    except Exception as e:
        logger.error(f"job {job_id} → failed  error={e}")
        await update_job(job_id, "failed", error=str(e))
```

- [ ] **Step 2: Verify the module imports cleanly (no LLM needed)**

```bash
uv run python -c "from workflows.effgen.briefing import _build_prompt, run_briefing; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Smoke-test the prompt builder**

```bash
uv run python -c "
from schemas.models import BriefingConfig, StockEntry
from workflows.effgen.briefing import _build_prompt

config = BriefingConfig(
    weather_location='Nashville, Tennessee',
    news_subjects=['AI', 'Finance'],
    stocks=[StockEntry(ticker='AAPL', name='Apple Inc.')],
    rss_feeds=['https://example.com/feed.xml'],
)
print(_build_prompt(config))
"
```

Expected: a multi-section prompt printed to stdout containing all three phases, the configured location, both subjects, the ticker, and the RSS feed URL.

- [ ] **Step 4: Verify the full test suite still passes**

```bash
uv run pytest tests/ -v
```

Expected: all previously passing tests still PASS (19 total)

- [ ] **Step 5: Commit**

```bash
git add workflows/effgen/briefing.py
git commit -m "feat: add morning briefing agent runner"
```

---

## Task 7: Run Full Integration Smoke Test

This task verifies end-to-end wiring without a live LLM. It checks that the trigger endpoint creates a job and the status endpoint can read it back.

- [ ] **Step 1: Add smoke test to `tests/routers/test_briefing_config.py`**

Add this test to the existing file:

```python
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
```

- [ ] **Step 2: Run the new test**

```bash
uv run pytest tests/routers/test_briefing_config.py::test_trigger_briefing_creates_job -v
```

Expected: PASSED

- [ ] **Step 3: Run the full test suite one final time**

```bash
uv run pytest tests/ -v
```

Expected: all 20 tests PASSED

- [ ] **Step 4: Final commit**

```bash
git add tests/routers/test_briefing_config.py
git commit -m "test: add briefing trigger smoke test"
```
