# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands use `uv` for environment management.

```bash
# Run the development server
uv run uvicorn main:app --reload

# Run with explicit host/port overrides
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Install LangGraph optional deps
uv sync --extra langgraph

# Add a new dependency
uv add <package>
```

`uv run pytest` runs the suite under `tests/`, organised by package (`tests/cfb/`, `tests/soccer/`, `tests/routers/`, `tests/core/`). Route tests use `fastapi.testclient.TestClient(app)` and patch `routers.workflows.run_<name>` with `AsyncMock`.

`tests/routers/test_briefing_config.py::test_trigger_briefing_creates_job` is a known pre-existing failure unrelated to the sports workflows.

## Architecture

This is a **fire-and-forget async job server**. The pattern is consistent across all workflows:

1. `POST /workflows/<name>` — creates a job row in SQLite (`status=pending`), enqueues the real work via FastAPI `BackgroundTasks`, returns `job_id` immediately
2. `GET /workflows/{job_id}` — polls job status; result and error are stored on the job row when done
3. The background task transitions the job through `pending → running → completed/failed` via `core/storage/jobs.py:update_job`

### Key layers

- **`config.py`** — single `Settings` (pydantic-settings, reads `.env`). All configurable values live here; import `settings` anywhere.
- **`core/storage/db.py`** — SQLite init and `get_db()` context manager (aiosqlite). Schema is a single `jobs` table.
- **`core/storage/jobs.py`** — CRUD over the jobs table: `create_job`, `update_job`, `get_job`, `list_jobs`.
- **`core/storage/obsidian.py`** — writes/reads Markdown notes to an Obsidian vault on disk. Workflow outputs are persisted here as notes under dated filenames.
- **`core/tracing.py`** — wraps Langfuse. Provides `@observe` decorator that no-ops gracefully when `LANGFUSE_*` keys are absent.
- **`schemas/models.py`** — all Pydantic request/response models.
- **`routers/workflows.py`** — registers workflow endpoints; imports workflow runner lazily inside the route handler to avoid loading the model at startup.
- **`workflows/effgen/research.py`** — loads a local GGUF model (lazy singleton, thread-safe lock), runs it via `effgen.create_agent`, and writes results to both the jobs table and an Obsidian note.

### Sports workflows (the exception to the job pattern)

These respond **synchronously** rather than creating a job, and read Delta
tables that `sports-etl` writes to MinIO. This server never calls the
CollegeFootballData API — it has no `cfbd` dependency and no API key.

| Route | Shape |
|---|---|
| `POST /workflows/soccer-analyst` | LLM → SQL → DuckDB; returns rows + a Plotly figure, or a `comparison` RadarFigure for player-comparison questions |
| `POST /workflows/cfb-analyst` | same, for college football |
| `POST /workflows/cfb/matchup` | **no LLM** — linear model over stored coefficients |
| `GET /workflows/cfb/slate` | this week's games, each predicted |
| `GET /workflows/cfb/backtest` | scores stored predictions against finals; replays any game with none stored, and flags each row `stored` or `replayed` |
| `GET /workflows/cfb/teams` | team list for pickers |
| `POST` / `GET /workflows/cfb/ingest` | **side effect:** runs `sports-etl`'s `pipeline_cfb_refresh.py` as a subprocess (`SPORTS_ETL_PATH`), one at a time; GET polls state, stage and log tail. Does not refit the model. Unauthenticated, like the rest of the sports routes |

The two analysts load whole Delta tables into memory as PyArrow at first
request and register them with DuckDB, so generated SQL uses plain table names,
not `delta_scan()`. They cache forever.

`GET /workflows/cfb/slate` also **writes**: every prediction it serves is
appended to `cfb_predictions` (`workflows/cfb/predictions.py`), append-only and
idempotent on (game_id, model fitted_at, through_week). This is the one table
agentic-server writes — a prediction is an artifact of serving, so the ETL
cannot know what the server showed. The write can never fail the request, and
tests must not reach it (`tests/cfb/conftest.py` blocks the I/O).

`workflows/cfb/matchup.py` is different: it reads `cfb_team_profile`,
`cfb_model` and `cfb_games` into a **30-minute TTL** cache, because the weekly
ETL refresh must become visible without a server restart. Every response stamps
`data_through_date` so a stale answer is visible rather than silent. Prediction
is one linear formula per head applied to floats stored in `cfb_model` — no
scikit-learn here; the model is fitted in `sports-etl`.

### Adding a new workflow

1. Create `workflows/<engine>/<name>.py` with an `async def run_<name>(job_id, request)` function that calls `update_job` to manage status.
2. Add request/response schemas to `schemas/models.py`.
3. Register a route in `routers/workflows.py` following the existing `POST /research` pattern.
4. Read-only endpoints must be nested under a prefix (e.g. `/cfb/teams`) — `GET /workflows/{job_id}` captures any single-segment GET path, so `GET /workflows/slate` would be treated as a job-id lookup. A path parameter never matches across a `/`, so nesting is immune regardless of declaration order.

### LLM / model

The model client is a lazy singleton in `workflows/effgen/research.py`, initialized on the first job via `effgen.load_model(..., provider="openai", base_url=..., api_key=...)`. This connects to any OpenAI-compatible endpoint (llama.cpp server, vLLM, etc.). The connection call runs in a thread via `asyncio.to_thread` to avoid blocking the event loop.

### Environment

Copy `.env.example` to `.env`. The server starts without Langfuse keys (tracing silently disabled). `OBSIDIAN_VAULT_PATH` must point to a real directory; notes are written under `<vault>/Research/`. Set `LLM_BASE_URL` to your llama.cpp (or other OpenAI-compatible) server's `/v1` endpoint before running research jobs.
