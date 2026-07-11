# Soccer Analyst Workflow — Design Spec

**Date:** 2026-07-11
**Status:** Approved

---

## Overview

A synchronous FastAPI endpoint that accepts a natural language soccer question, uses a local LLM (via effgen) to generate DuckDB SQL, executes the query against Delta Lake tables on MinIO, and returns both raw data rows and a Plotly figure JSON for frontend rendering.

**Scope:** Premier League 2024 season (Understat data). Single structured LLM prompt approach — if the 8B model proves unreliable, pivot to a ReAct agent with DuckDB tools (Option B from brainstorming).

---

## Architecture

### New files
```
workflows/soccer/
├── __init__.py
└── analyst.py        # all workflow logic: prompt, DuckDB, Plotly
```

### Modified files
```
schemas/models.py     # + SoccerAnalystRequest, SoccerAnalystResponse, SoccerQueryPlan
routers/workflows.py  # + POST /soccer-analyst
config.py             # + minio_endpoint, minio_access_key, minio_secret_key, minio_bucket,
                      #   soccer_table_prefix
```

### Pattern
Synchronous endpoint — no job queue. `POST /soccer-analyst` does the work and returns directly. This diverges from the fire-and-forget pattern used by other workflows, which is intentional: text-to-SQL is fast enough (~2-5s) and the frontend needs the result synchronously to render the chart.

---

## Data Flow

```
POST /soccer-analyst  { "question": "..." }
  │
  ├─ 1. Build prompt (hardcoded schema + question)
  ├─ 2. Call LLM → SoccerQueryPlan { sql, chart_type, x_column, y_column, title }
  ├─ 3. Run SQL via DuckDB against MinIO Delta tables
  │       └─ on failure: retry once (append error to prompt, call LLM again)
  │       └─ on second failure: return HTTP 422 { error, sql }
  ├─ 4. Build Plotly figure JSON from result + SoccerQueryPlan
  └─ 5. Return SoccerAnalystResponse
```

---

## LLM Integration

Reuses the lazy model singleton pattern from `workflows/effgen/research.py` — the same `_get_model()` / `_model_lock` pattern. No new model loading code needed.

**Structured output model:**
```python
class SoccerQueryPlan(BaseModel):
    sql: str
    chart_type: str        # "bar" | "line" | "scatter"
    x_column: str          # column name for x axis
    y_column: str          # column name for y axis
    title: str             # chart title
```

Passed as `output_model=SoccerQueryPlan` to `effgen.create_agent`. If effgen requires a named preset for `output_model` to work (as in `research.py`), fall back to calling the model directly via the OpenAI-compatible HTTP client (`openai.OpenAI(base_url=...).chat.completions.create(...)`) and parsing the JSON response manually.

**System prompt:**
```
You are a soccer data analyst. You write DuckDB SQL queries against Delta Lake tables.
Always use delta_scan() to read tables. Return only valid JSON matching the requested schema.
Do not explain your reasoning. Return JSON only.
```

**User prompt template:**
```
Answer this question using SQL: {question}

Available tables (use delta_scan with full s3 paths shown):

fixtures ({fixtures_path}):
  {fixtures_columns}

player_match ({player_match_path}):
  {player_match_columns}

player_season ({player_season_path}):
  {player_season_columns}

team_stats ({team_stats_path}):
  {team_stats_columns}

Rules:
- Use delta_scan('s3://...') for every table reference
- The season column value is an integer (e.g. 2024)
- Limit results to a reasonable number (e.g. LIMIT 10 unless specified)

Return JSON with exactly these fields:
- sql: the complete DuckDB SQL query
- chart_type: one of "bar", "line", "scatter"
- x_column: column name to use for the x axis
- y_column: column name to use for the y axis
- title: a descriptive chart title
```

**Column names:** Must be verified at implementation time by running `SELECT * FROM delta_scan('...') LIMIT 1` against each live Delta table on MinIO and recording the actual normalized column names. The prompt template uses `{fixtures_columns}` etc. as placeholders that get filled from a hardcoded dict in `analyst.py`.

---

## Self-Correction Retry

```python
def _run_query_with_retry(plan: SoccerQueryPlan, prompt: str, model) -> pl.DataFrame:
    try:
        return _execute_sql(plan.sql)
    except Exception as e:
        retry_prompt = prompt + f"\n\nThe query failed: {e}\nFix the SQL and return corrected JSON."
        plan = _call_llm(model, retry_prompt)
        return _execute_sql(plan.sql)  # raises if still fails
```

On second failure the endpoint returns HTTP 422:
```json
{ "error": "DuckDB error: ...", "sql": "SELECT ..." }
```

---

## DuckDB Connection

Module-level singleton in `analyst.py`, initialized once on first request:

```python
def _get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL delta; LOAD delta;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{_minio_host_port}'")
    con.execute(f"SET s3_access_key_id='{settings.minio_access_key}'")
    con.execute(f"SET s3_secret_access_key='{settings.minio_secret_key}'")
    con.execute("SET s3_use_ssl=false")
    con.execute("SET s3_url_style='path'")
    return con
```

Config values added to `config.py` as optional fields with empty string defaults (server starts without them; soccer analyst fails gracefully if absent).

---

## Plotly Chart Building

Server-side, deterministic — LLM never touches Plotly:

```python
def _build_plotly_figure(df: pl.DataFrame, plan: SoccerQueryPlan) -> dict:
    rows = df.to_dicts()
    x_vals = [r[plan.x_column] for r in rows]
    y_vals = [r[plan.y_column] for r in rows]

    trace = {"type": plan.chart_type, "x": x_vals, "y": y_vals, "name": plan.y_column}
    layout = {"title": plan.title, "xaxis": {"title": plan.x_column},
              "yaxis": {"title": plan.y_column}}
    return {"data": [trace], "layout": layout}
```

---

## API

**Request:**
```python
class SoccerAnalystRequest(BaseModel):
    question: str
```

**Success response:**
```python
class SoccerAnalystResponse(BaseModel):
    question: str
    sql: str
    data: list[dict]        # raw result rows
    chart: dict             # Plotly figure JSON { data, layout }
    row_count: int
```

**Error response (HTTP 422):**
```json
{ "error": "...", "sql": "..." }
```

**Route:**
```
POST /workflows/soccer-analyst
```

Follows the existing router prefix convention. No auth required for the POC (can be added later following the `Depends(get_current_user)` pattern).

---

## Config Additions

`config.py`:
```python
minio_endpoint: str = ""          # e.g. http://localhost:9000
minio_access_key: str = ""
minio_secret_key: str = ""
minio_bucket: str = "soccer"
```

`.env.example` additions:
```
# Soccer analyst
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
MINIO_BUCKET=soccer
```

---

## Dependencies

Add to `agentic-server` pyproject.toml:
```
duckdb>=0.10
deltalake>=0.17
polars>=0.20
```

---

## Pivot Criteria

If the 8B model consistently fails to produce valid SQL (>50% failure rate after retry), pivot to **Option B: ReAct agent with DuckDB tools** — replace the single prompt with an effgen agent that calls `describe_table()` and `run_query()` tools in a loop. The DuckDB connection, Plotly builder, and API shape remain unchanged.
