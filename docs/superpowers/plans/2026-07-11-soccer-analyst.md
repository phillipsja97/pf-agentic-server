# Soccer Analyst Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `POST /workflows/soccer-analyst` endpoint that accepts a natural language soccer question, generates SQL via a local LLM, runs it against Delta Lake tables on MinIO, and returns raw data rows plus a Plotly figure JSON.

**Architecture:** Delta tables are loaded from MinIO via the `deltalake` Python library into PyArrow tables and registered as in-memory DuckDB tables on first request (lazy singleton). The LLM is called via the OpenAI-compatible HTTP client with a tight structured-output prompt; the response is parsed into `SoccerQueryPlan`. One self-correction retry on SQL failure. Plotly JSON is built server-side from query results.

**Tech Stack:** FastAPI, DuckDB, deltalake (delta-rs Python bindings), polars, openai (Python client), pydantic

## Global Constraints

- Follow existing agentic-server patterns: lazy singletons with `asyncio.Lock`, `asyncio.to_thread` for sync work, lazy imports in route handlers
- All new settings go in `config.py` as optional `str` fields with `""` defaults — server must start without MinIO vars set
- Tests use `TestClient(app)` and `monkeypatch` on `settings` — follow `tests/routers/test_briefing_config.py` pattern
- Use `uv add <package>` to add dependencies, not manual pyproject.toml edits
- Season value in all tables is integer `2024`
- Table names for SQL: `fixtures`, `player_match`, `player_season`, `team_stats` (plain names, not delta_scan paths)
- MinIO endpoint: `http://192.168.1.189:9000`, bucket: `soccer`, table prefix: `soccer/<table_name>/`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `workflows/soccer/__init__.py` | Create | Package marker |
| `workflows/soccer/analyst.py` | Create | All workflow logic: DuckDB, LLM, Plotly |
| `schemas/models.py` | Modify | Add SoccerQueryPlan, SoccerAnalystRequest, SoccerAnalystResponse |
| `routers/workflows.py` | Modify | Add POST /soccer-analyst route |
| `config.py` | Modify | Add minio_endpoint, minio_access_key, minio_secret_key, minio_bucket |
| `.env.example` | Modify | Add MinIO vars block |
| `tests/soccer/__init__.py` | Create | Test package marker |
| `tests/soccer/test_analyst.py` | Create | Unit tests for analyst helpers |
| `tests/routers/test_soccer_analyst.py` | Create | Integration test for the endpoint |

---

## Task 1: Scaffolding — deps, config, package init

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Create: `workflows/soccer/__init__.py`
- Create: `tests/soccer/__init__.py`

**Interfaces:**
- Produces:
  - `settings.minio_endpoint: str` (default `""`)
  - `settings.minio_access_key: str` (default `""`)
  - `settings.minio_secret_key: str` (default `""`)
  - `settings.minio_bucket: str` (default `"soccer"`)

- [ ] **Step 1: Add dependencies**

```bash
cd /home/jphill/Projects/agentic-server
uv add duckdb deltalake polars openai
```

Expected: installs without errors. `uv.lock` updates.

- [ ] **Step 2: Add MinIO settings to `config.py`**

Add these four fields to the `Settings` class (after `openai_api_key`):

```python
minio_endpoint: str = ""
minio_access_key: str = ""
minio_secret_key: str = ""
minio_bucket: str = "soccer"
```

Full updated `config.py`:

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

    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "none"
    llm_model: str = "local"
    llm_max_tokens: int | None = None

    host: str = "0.0.0.0"
    port: int = 8060

    jwt_secret: str = "change-me-in-production"

    anthropic_api_key: str = ""
    openai_api_key: str = ""

    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "soccer"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

- [ ] **Step 3: Add MinIO block to `.env.example`**

Append to the end of `/home/jphill/Projects/agentic-server/.env.example`:

```
# Soccer analyst (MinIO / S3-compatible)
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET=soccer
```

- [ ] **Step 4: Create package init files**

Create empty `/home/jphill/Projects/agentic-server/workflows/soccer/__init__.py`

Create empty `/home/jphill/Projects/agentic-server/tests/soccer/__init__.py`

- [ ] **Step 5: Verify server still starts**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8060 &
sleep 3
curl -s http://localhost:8060/health | python3 -m json.tool
kill %1
```

Expected: `{"status": "ok", ...}` — server starts without the MinIO vars set.

- [ ] **Step 6: Commit**

```bash
git add config.py .env.example workflows/soccer/__init__.py tests/soccer/__init__.py uv.lock pyproject.toml
git commit -m "chore: add soccer analyst scaffolding, MinIO config, and dependencies"
```

---

## Task 2: Schemas

**Files:**
- Modify: `schemas/models.py`
- Test: `tests/soccer/test_analyst.py` (partial — schema tests only)

**Interfaces:**
- Produces:
  - `SoccerQueryPlan(sql: str, chart_type: str, x_column: str, y_column: str, title: str)`
  - `SoccerAnalystRequest(question: str)`
  - `SoccerAnalystResponse(question: str, sql: str, data: list[dict], chart: dict, row_count: int)`

- [ ] **Step 1: Write the failing tests**

Create `/home/jphill/Projects/agentic-server/tests/soccer/test_analyst.py`:

```python
import pytest
from schemas.models import SoccerQueryPlan, SoccerAnalystRequest, SoccerAnalystResponse


def test_soccer_query_plan_parses_json():
    raw = '{"sql": "SELECT player FROM player_season LIMIT 5", "chart_type": "bar", "x_column": "player", "y_column": "assists", "title": "Top Assisters"}'
    plan = SoccerQueryPlan.model_validate_json(raw)
    assert plan.sql == "SELECT player FROM player_season LIMIT 5"
    assert plan.chart_type == "bar"
    assert plan.x_column == "player"
    assert plan.y_column == "assists"
    assert plan.title == "Top Assisters"


def test_soccer_analyst_request_requires_question():
    req = SoccerAnalystRequest(question="who scored the most goals?")
    assert req.question == "who scored the most goals?"


def test_soccer_analyst_response_shape():
    resp = SoccerAnalystResponse(
        question="test",
        sql="SELECT 1",
        data=[{"player": "Salah", "goals": 20}],
        chart={"data": [], "layout": {}},
        row_count=1,
    )
    assert resp.row_count == 1
    assert resp.data[0]["player"] == "Salah"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/soccer/test_analyst.py -v
```

Expected: FAIL — `ImportError: cannot import name 'SoccerQueryPlan'`

- [ ] **Step 3: Add schemas to `schemas/models.py`**

Append to the end of `schemas/models.py`:

```python
class SoccerQueryPlan(BaseModel):
    sql: str
    chart_type: str
    x_column: str
    y_column: str
    title: str


class SoccerAnalystRequest(BaseModel):
    question: str


class SoccerAnalystResponse(BaseModel):
    question: str
    sql: str
    data: list[dict]
    chart: dict
    row_count: int
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/soccer/test_analyst.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add schemas/models.py tests/soccer/test_analyst.py
git commit -m "feat: add SoccerQueryPlan, SoccerAnalystRequest, SoccerAnalystResponse schemas"
```

---

## Task 3: DuckDB Data Layer

**Files:**
- Create: `workflows/soccer/analyst.py` (partial — data layer only)
- Test: `tests/soccer/test_analyst.py` (append)

**Interfaces:**
- Consumes: `settings.minio_endpoint`, `settings.minio_access_key`, `settings.minio_secret_key`, `settings.minio_bucket`
- Produces:
  - `_build_connection() -> duckdb.DuckDBPyConnection` — loads 4 Delta tables from MinIO, registers as in-memory DuckDB tables
  - `_get_con() -> duckdb.DuckDBPyConnection` — async lazy singleton wrapping `_build_connection`

- [ ] **Step 1: Write the failing test**

Append to `tests/soccer/test_analyst.py`:

```python
import duckdb
import polars as pl
from unittest.mock import patch, MagicMock
import pyarrow as pa


def _make_arrow_table():
    return pa.table({"player": ["Salah", "Haaland"], "goals": [20, 25], "season": [2024, 2024]})


def test_build_connection_registers_tables(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "minio_endpoint", "http://fake:9000")
    monkeypatch.setattr(settings, "minio_access_key", "key")
    monkeypatch.setattr(settings, "minio_secret_key", "secret")
    monkeypatch.setattr(settings, "minio_bucket", "soccer")

    mock_dt = MagicMock()
    mock_dt.to_pyarrow_table.return_value = _make_arrow_table()

    with patch("workflows.soccer.analyst.DeltaTable", return_value=mock_dt):
        from workflows.soccer.analyst import _build_connection
        con = _build_connection()

    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    assert "fixtures" in tables
    assert "player_match" in tables
    assert "player_season" in tables
    assert "team_stats" in tables
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/soccer/test_analyst.py::test_build_connection_registers_tables -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'workflows.soccer.analyst'`

- [ ] **Step 3: Create `workflows/soccer/analyst.py` with the data layer**

```python
# workflows/soccer/analyst.py
import asyncio
from typing import Optional

import duckdb
import polars as pl
from deltalake import DeltaTable

from config import settings
from core.logging import logger
from schemas.models import SoccerAnalystRequest, SoccerAnalystResponse, SoccerQueryPlan

_TABLE_PATHS = {
    "fixtures": "soccer/fixtures/",
    "player_match": "soccer/player_match/",
    "player_season": "soccer/player_season/",
    "team_stats": "soccer/team_stats/",
}

_TABLE_SCHEMAS = {
    "fixtures": (
        "game_id (TEXT), date (DATE), home_team (TEXT), away_team (TEXT), "
        "home_goals (INT), away_goals (INT), home_xg (FLOAT), away_xg (FLOAT), "
        "season (INT)"
    ),
    "player_match": (
        "player (TEXT), team (TEXT), position (TEXT), goals (INT), assists (INT), "
        "xg (FLOAT), xa (FLOAT), shots (INT), key_passes (INT), minutes (INT), "
        "yellow_cards (INT), red_cards (INT), season (INT), game_id (TEXT)"
    ),
    "player_season": (
        "player (TEXT), team (TEXT), position (TEXT), matches (INT), minutes (INT), "
        "goals (INT), xg (FLOAT), np_goals (INT), np_xg (FLOAT), assists (INT), "
        "xa (FLOAT), shots (INT), key_passes (INT), yellow_cards (INT), "
        "red_cards (INT), season (INT)"
    ),
    "team_stats": (
        "home_team (TEXT), away_team (TEXT), date (DATE), "
        "home_goals (INT), away_goals (INT), home_xg (FLOAT), away_xg (FLOAT), "
        "home_expected_points (FLOAT), away_expected_points (FLOAT), "
        "home_ppda (FLOAT), away_ppda (FLOAT), "
        "home_deep_completions (INT), away_deep_completions (INT), "
        "season (INT), game_id (TEXT)"
    ),
}

_con: Optional[duckdb.DuckDBPyConnection] = None
_con_lock = asyncio.Lock()


def _build_connection() -> duckdb.DuckDBPyConnection:
    storage_options = {
        "endpoint_url": settings.minio_endpoint,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }

    con = duckdb.connect()
    for table_name, path_suffix in _TABLE_PATHS.items():
        path = f"s3://{settings.minio_bucket}/{path_suffix}"
        dt = DeltaTable(path, storage_options=storage_options)
        arrow_table = dt.to_pyarrow_table()
        con.register(table_name, arrow_table)
        logger.info(f"soccer analyst: loaded {table_name} ({arrow_table.num_rows} rows)")

    return con


async def _get_con() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is not None:
        return _con
    async with _con_lock:
        if _con is not None:
            return _con
        _con = await asyncio.to_thread(_build_connection)
    return _con
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/soccer/test_analyst.py::test_build_connection_registers_tables -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add workflows/soccer/analyst.py tests/soccer/test_analyst.py
git commit -m "feat: add soccer analyst DuckDB data layer with Delta table loading"
```

---

## Task 4: Prompt Builder + LLM Layer

**Files:**
- Modify: `workflows/soccer/analyst.py` (append)
- Test: `tests/soccer/test_analyst.py` (append)

**Interfaces:**
- Consumes: `SoccerQueryPlan` from Task 2, `settings.llm_base_url`, `settings.llm_api_key`, `settings.llm_model`
- Produces:
  - `_build_prompt(question: str) -> str`
  - `_call_llm_sync(client: openai.OpenAI, prompt: str) -> SoccerQueryPlan`
  - `_get_client() -> openai.OpenAI` (async lazy singleton)

- [ ] **Step 1: Write the failing tests**

Append to `tests/soccer/test_analyst.py`:

```python
from workflows.soccer.analyst import _build_prompt, _call_llm_sync


def test_build_prompt_contains_question():
    prompt = _build_prompt("who scored the most goals?")
    assert "who scored the most goals?" in prompt


def test_build_prompt_contains_all_table_names():
    prompt = _build_prompt("test")
    assert "fixtures" in prompt
    assert "player_match" in prompt
    assert "player_season" in prompt
    assert "team_stats" in prompt


def test_build_prompt_mentions_season_type():
    prompt = _build_prompt("test")
    assert "2024" in prompt


def test_call_llm_sync_parses_clean_json(monkeypatch):
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        '{"sql": "SELECT player FROM player_season LIMIT 5", '
        '"chart_type": "bar", "x_column": "player", '
        '"y_column": "goals", "title": "Top Scorers"}'
    )
    result = _call_llm_sync(mock_client, "test prompt")
    assert result.sql == "SELECT player FROM player_season LIMIT 5"
    assert result.chart_type == "bar"


def test_call_llm_sync_strips_markdown_fences(monkeypatch):
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "```json\n"
        '{"sql": "SELECT 1", "chart_type": "bar", '
        '"x_column": "a", "y_column": "b", "title": "T"}\n'
        "```"
    )
    result = _call_llm_sync(mock_client, "test prompt")
    assert result.sql == "SELECT 1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/soccer/test_analyst.py::test_build_prompt_contains_question tests/soccer/test_analyst.py::test_call_llm_sync_parses_clean_json -v
```

Expected: FAIL — `ImportError: cannot import name '_build_prompt'`

- [ ] **Step 3: Append LLM layer to `workflows/soccer/analyst.py`**

Add these imports at the top of analyst.py (after existing imports):

```python
import openai as openai_lib
```

Then append to the bottom of `workflows/soccer/analyst.py`:

```python
_SYSTEM_PROMPT = (
    "You are a soccer data analyst. You write DuckDB SQL queries. "
    "Tables are already loaded in memory — use plain table names, not delta_scan(). "
    "Return only valid JSON with no explanation or markdown."
)

_client: Optional[openai_lib.OpenAI] = None
_client_lock = asyncio.Lock()


async def _get_client() -> openai_lib.OpenAI:
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is not None:
            return _client
        _client = openai_lib.OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    return _client


def _build_prompt(question: str) -> str:
    schema_lines = [
        f"{table}: {columns}"
        for table, columns in _TABLE_SCHEMAS.items()
    ]
    schema_str = "\n".join(schema_lines)

    return (
        f"Answer this question about Premier League 2024 data: {question}\n\n"
        f"Available tables (season column is integer 2024):\n{schema_str}\n\n"
        "Return JSON with exactly these fields:\n"
        '- sql: DuckDB SQL using plain table names (e.g. SELECT * FROM player_season)\n'
        '- chart_type: one of "bar", "line", "scatter"\n'
        "- x_column: column name for x axis\n"
        "- y_column: column name for y axis\n"
        "- title: descriptive chart title"
    )


def _call_llm_sync(client: openai_lib.OpenAI, prompt: str) -> SoccerQueryPlan:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return SoccerQueryPlan.model_validate_json(raw.strip())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/soccer/test_analyst.py -k "prompt or llm" -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/soccer/analyst.py tests/soccer/test_analyst.py
git commit -m "feat: add soccer analyst prompt builder and LLM call layer"
```

---

## Task 5: SQL Execution + Plotly + Full Workflow

**Files:**
- Modify: `workflows/soccer/analyst.py` (append)
- Test: `tests/soccer/test_analyst.py` (append)

**Interfaces:**
- Consumes: `_get_con()`, `_get_client()`, `_build_prompt()`, `_call_llm_sync()`, `SoccerQueryPlan`, `SoccerAnalystRequest`, `SoccerAnalystResponse`
- Produces:
  - `_execute_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame`
  - `_build_plotly_figure(df: pl.DataFrame, plan: SoccerQueryPlan) -> dict`
  - `run_soccer_analyst(request: SoccerAnalystRequest) -> SoccerAnalystResponse` (async)

- [ ] **Step 1: Write the failing tests**

Append to `tests/soccer/test_analyst.py`:

```python
import duckdb
import polars as pl
from workflows.soccer.analyst import _execute_sql, _build_plotly_figure


def _make_in_memory_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE player_season AS SELECT 'Salah' AS player, 20 AS goals, "
        "'FW' AS position, 2024 AS season"
    )
    return con


def test_execute_sql_returns_polars_df():
    con = _make_in_memory_con()
    df = _execute_sql(con, "SELECT player, goals FROM player_season")
    assert isinstance(df, pl.DataFrame)
    assert df["player"][0] == "Salah"
    assert df["goals"][0] == 20


def test_execute_sql_raises_on_bad_sql():
    con = _make_in_memory_con()
    with pytest.raises(Exception):
        _execute_sql(con, "SELECT * FROM nonexistent_table")


def test_build_plotly_figure_bar_chart():
    df = pl.DataFrame({"player": ["Salah", "Haaland"], "goals": [20, 25]})
    plan = SoccerQueryPlan(
        sql="SELECT player, goals FROM player_season LIMIT 2",
        chart_type="bar",
        x_column="player",
        y_column="goals",
        title="Top Scorers",
    )
    fig = _build_plotly_figure(df, plan)
    assert fig["data"][0]["type"] == "bar"
    assert fig["data"][0]["x"] == ["Salah", "Haaland"]
    assert fig["data"][0]["y"] == [20, 25]
    assert fig["layout"]["title"] == "Top Scorers"


def test_build_plotly_figure_missing_column_returns_nones():
    df = pl.DataFrame({"player": ["Salah"], "goals": [20]})
    plan = SoccerQueryPlan(
        sql="SELECT 1",
        chart_type="bar",
        x_column="player",
        y_column="nonexistent",
        title="T",
    )
    fig = _build_plotly_figure(df, plan)
    assert fig["data"][0]["y"] == [None]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/soccer/test_analyst.py::test_execute_sql_returns_polars_df tests/soccer/test_analyst.py::test_build_plotly_figure_bar_chart -v
```

Expected: FAIL — `ImportError: cannot import name '_execute_sql'`

- [ ] **Step 3: Append execution + Plotly + workflow to `workflows/soccer/analyst.py`**

```python
def _execute_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame:
    return con.execute(sql).pl()


def _build_plotly_figure(df: pl.DataFrame, plan: SoccerQueryPlan) -> dict:
    rows = df.to_dicts()
    x_vals = [r.get(plan.x_column) for r in rows]
    y_vals = [r.get(plan.y_column) for r in rows]
    return {
        "data": [{
            "type": plan.chart_type,
            "x": x_vals,
            "y": y_vals,
            "name": plan.y_column,
        }],
        "layout": {
            "title": plan.title,
            "xaxis": {"title": plan.x_column},
            "yaxis": {"title": plan.y_column},
        },
    }


async def run_soccer_analyst(request: SoccerAnalystRequest) -> SoccerAnalystResponse:
    client = await _get_client()
    con = await _get_con()
    prompt = _build_prompt(request.question)

    plan = await asyncio.to_thread(_call_llm_sync, client, prompt)

    try:
        df = await asyncio.to_thread(_execute_sql, con, plan.sql)
    except Exception as e:
        retry_prompt = (
            prompt
            + f"\n\nThe previous query failed with this error: {e}\n"
            "Fix the SQL and return corrected JSON."
        )
        plan = await asyncio.to_thread(_call_llm_sync, client, retry_prompt)
        df = await asyncio.to_thread(_execute_sql, con, plan.sql)

    chart = _build_plotly_figure(df, plan)
    return SoccerAnalystResponse(
        question=request.question,
        sql=plan.sql,
        data=df.to_dicts(),
        chart=chart,
        row_count=len(df),
    )
```

- [ ] **Step 4: Run all analyst unit tests**

```bash
uv run pytest tests/soccer/test_analyst.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/soccer/analyst.py tests/soccer/test_analyst.py
git commit -m "feat: add SQL execution, Plotly builder, and run_soccer_analyst workflow"
```

---

## Task 6: Router + Integration Test

**Files:**
- Modify: `routers/workflows.py`
- Modify: `schemas/models.py` (add import to router)
- Create: `tests/routers/test_soccer_analyst.py`

**Interfaces:**
- Consumes: `run_soccer_analyst(request: SoccerAnalystRequest) -> SoccerAnalystResponse`
- Produces: `POST /workflows/soccer-analyst` endpoint

- [ ] **Step 1: Write the failing integration test**

Create `/home/jphill/Projects/agentic-server/tests/routers/test_soccer_analyst.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app
from schemas.models import SoccerAnalystResponse


@pytest.fixture
def client():
    return TestClient(app)


def _mock_response():
    return SoccerAnalystResponse(
        question="who scored the most goals?",
        sql="SELECT player, goals FROM player_season ORDER BY goals DESC LIMIT 5",
        data=[{"player": "Haaland", "goals": 25}],
        chart={
            "data": [{"type": "bar", "x": ["Haaland"], "y": [25], "name": "goals"}],
            "layout": {"title": "Top Scorers", "xaxis": {"title": "player"}, "yaxis": {"title": "goals"}},
        },
        row_count=1,
    )


def test_soccer_analyst_returns_200(client):
    with patch(
        "routers.workflows.run_soccer_analyst",
        new=AsyncMock(return_value=_mock_response()),
    ):
        r = client.post(
            "/workflows/soccer-analyst",
            json={"question": "who scored the most goals?"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "who scored the most goals?"
    assert "sql" in body
    assert "chart" in body
    assert "data" in body
    assert body["row_count"] == 1


def test_soccer_analyst_requires_question(client):
    r = client.post("/workflows/soccer-analyst", json={})
    assert r.status_code == 422


def test_soccer_analyst_returns_422_on_workflow_error(client):
    with patch(
        "routers.workflows.run_soccer_analyst",
        new=AsyncMock(side_effect=Exception("DuckDB error: table not found")),
    ):
        r = client.post(
            "/workflows/soccer-analyst",
            json={"question": "bad question"},
        )
    assert r.status_code == 422
    assert "error" in r.json()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/routers/test_soccer_analyst.py -v
```

Expected: FAIL — endpoint not registered yet

- [ ] **Step 3: Add route to `routers/workflows.py`**

Add these imports to the existing import block in `routers/workflows.py`:

```python
from schemas.models import (
    ...existing imports...
    SoccerAnalystRequest,
    SoccerAnalystResponse,
)
```

Then append the route to the bottom of `routers/workflows.py`:

```python
@router.post("/soccer-analyst", response_model=SoccerAnalystResponse)
async def soccer_analyst(request: SoccerAnalystRequest) -> SoccerAnalystResponse:
    from workflows.soccer.analyst import run_soccer_analyst
    try:
        return await run_soccer_analyst(request)
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})
```

- [ ] **Step 4: Add `SoccerAnalystRequest` to the import in `routers/workflows.py`**

The full updated import block at the top of `routers/workflows.py`:

```python
from schemas.models import (
    CodingRequest,
    JobCreatedResponse,
    JobStatusResponse,
    LearningPlanRequest,
    RagChatRequest,
    RagIngestRequest,
    ResearchRequest,
    SoccerAnalystRequest,
    SoccerAnalystResponse,
)
```

- [ ] **Step 5: Run integration tests**

```bash
uv run pytest tests/routers/test_soccer_analyst.py -v
```

Expected: 3 tests PASS

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add routers/workflows.py tests/routers/test_soccer_analyst.py schemas/models.py
git commit -m "feat: add POST /workflows/soccer-analyst endpoint"
```

---

## Smoke Test Against Live Data

Once the server is running with MinIO creds in `.env`:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8060 --reload
```

```bash
curl -s -X POST http://localhost:8060/workflows/soccer-analyst \
  -H "Content-Type: application/json" \
  -d '{"question": "which 5 players had the most assists in the 2024 season?"}' \
  | python3 -m json.tool
```

Expected response shape:
```json
{
  "question": "which 5 players had the most assists in the 2024 season?",
  "sql": "SELECT player, assists FROM player_season WHERE season = 2024 ORDER BY assists DESC LIMIT 5",
  "data": [{"player": "...", "assists": ...}, ...],
  "chart": {"data": [...], "layout": {...}},
  "row_count": 5
}
```

If the LLM returns malformed JSON or bad SQL, the error message will indicate whether to retry the question or pivot to the ReAct agent approach (Option B from the spec).
