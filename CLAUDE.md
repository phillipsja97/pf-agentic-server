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

No test suite is configured yet. `httpx` is available as a dev dependency for writing integration tests against the running server.

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
- **`workflows/effgen/research.py`** — the only implemented workflow. Loads a local GGUF model (lazy singleton, thread-safe lock), runs it via `effgen.create_agent`, and writes results to both the jobs table and an Obsidian note.

### Adding a new workflow

1. Create `workflows/<engine>/<name>.py` with an `async def run_<name>(job_id, request)` function that calls `update_job` to manage status.
2. Add request/response schemas to `schemas/models.py`.
3. Register a route in `routers/workflows.py` following the existing `POST /research` pattern.

### LLM / model

The model client is a lazy singleton in `workflows/effgen/research.py`, initialized on the first job via `effgen.load_model(..., provider="openai", base_url=..., api_key=...)`. This connects to any OpenAI-compatible endpoint (llama.cpp server, vLLM, etc.). The connection call runs in a thread via `asyncio.to_thread` to avoid blocking the event loop.

### Environment

Copy `.env.example` to `.env`. The server starts without Langfuse keys (tracing silently disabled). `OBSIDIAN_VAULT_PATH` must point to a real directory; notes are written under `<vault>/Research/`. Set `LLM_BASE_URL` to your llama.cpp (or other OpenAI-compatible) server's `/v1` endpoint before running research jobs.
