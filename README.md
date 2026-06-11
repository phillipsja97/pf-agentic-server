# Agentic Workflow Server

A fire-and-forget async job server for running AI workflows. Clients submit a job and get a `job_id` back immediately; the work runs in the background and results are polled via a status endpoint.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- An OpenAI-compatible LLM endpoint (e.g. llama.cpp server, vLLM) for the research workflow
- An [Obsidian](https://obsidian.md/) vault directory for persisting workflow output as notes

## Setup

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Description |
|---|---|
| `OBSIDIAN_VAULT_PATH` | Absolute path to your Obsidian vault directory |
| `LLM_BASE_URL` | Base URL of your OpenAI-compatible LLM server (e.g. `http://localhost:8080/v1`) |
| `LLM_MODEL` | Model name to pass to the endpoint |
| `SQLITE_PATH` | Path for the SQLite jobs database (default: `./data/jobs.db`) |

Langfuse tracing is optional — the server starts fine without those keys.

## Running

```bash
# Development (auto-reload on file changes)
uv run uvicorn main:app --reload

# With explicit host/port
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API docs are available at `http://localhost:8000/docs` once the server is running.

## API

### Health

```
GET /health
```

Returns `{ "status": "ok", "version": "0.1.0" }`.

### Research workflow

**Submit a job**

```
POST /workflows/research
Content-Type: application/json

{
  "query": "What are the latest advances in transformer architectures?",
  "depth": "standard"
}
```

Returns `{ "job_id": "<uuid>" }` immediately.

**Poll status**

```
GET /workflows/{job_id}
```

Returns job status and result when complete:

```json
{
  "id": "...",
  "type": "research",
  "status": "completed",
  "input": { "query": "...", "depth": "standard" },
  "result": "...",
  "error": null,
  "created_at": "...",
  "updated_at": "..."
}
```

Status values: `pending` → `running` → `completed` / `failed`.

**List jobs**

```
GET /workflows/?type=research&limit=50
```

## Testing

No automated test suite exists yet. `httpx` is available as a dev dependency for writing integration tests against the running server.

Manual end-to-end test against a running server:

```bash
# Submit a research job
curl -s -X POST http://localhost:8000/workflows/research \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "depth": "standard"}' | tee /tmp/job.json

# Poll until done (replace JOB_ID)
JOB_ID=$(cat /tmp/job.json | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
curl -s http://localhost:8000/workflows/$JOB_ID | python3 -m json.tool
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for a full description of the codebase layout and how to add new workflows.
