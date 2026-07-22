# Soccer Player Comparison Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `/workflows/soccer-analyst` to detect player comparison queries and return a `RadarFigure` in the `comparison` field of `SoccerAnalystResponse`, which the frontend radar chart already consumes.

**Architecture:** The LLM prompt is extended to return a discriminated plan — either a regular query plan (`is_comparison: false`) or a comparison plan (`is_comparison: true`, with player names and YYZZ-encoded seasons). A new `_parse_plan` function dispatches to the right Pydantic model. Comparison queries run a fixed DuckDB `PERCENT_RANK()` window-function query with no extra LLM call; the regular path is completely unchanged.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, DuckDB, Polars, `uv` for running tests

## Global Constraints

- Tests run with: `cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/ -v`
- Use `unittest.mock` (not `pytest-mock`) — already established pattern in existing tests
- `SoccerQueryPlan` must not be modified — existing tests must continue to pass
- The regular query path in `run_soccer_analyst` must remain unchanged
- All new code lives in `schemas/models.py` and `workflows/soccer/analyst.py`
- No new dependencies required

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `schemas/models.py` | Modify | Add `ComparisonQueryPlan`; add `comparison: Optional[dict]` to `SoccerAnalystResponse` |
| `workflows/soccer/analyst.py` | Modify | Add `_parse_plan`, extend `_build_prompt`, add `_build_comparison`, branch `run_soccer_analyst` |
| `tests/soccer/test_analyst.py` | Modify | Add 7 new tests (TDD — written before implementation in each task) |

---

## Task 1: Schema additions

**Files:**
- Modify: `schemas/models.py`
- Test: `tests/soccer/test_analyst.py`

**Interfaces:**
- Produces: `ComparisonQueryPlan` — consumed by Task 2 (`_parse_plan`) and Task 3 (`_build_comparison`, `run_soccer_analyst`)
- Produces: `SoccerAnalystResponse.comparison: Optional[dict]` — consumed by Task 3

- [ ] **Step 1: Write the failing tests**

Add to `tests/soccer/test_analyst.py` (after the existing `test_soccer_analyst_response_shape` test):

```python
from typing import Literal


def test_comparison_query_plan_parses():
    from schemas.models import ComparisonQueryPlan
    plan = ComparisonQueryPlan(
        is_comparison=True,
        player_a="N'Golo Kante",
        season_a=1617,
        player_b="Moises Caicedo",
        season_b=2425,
        title="Kante 2016-17 vs Caicedo 2024-25",
    )
    assert plan.player_a == "N'Golo Kante"
    assert plan.season_a == 1617
    assert plan.is_comparison is True


def test_soccer_analyst_response_accepts_comparison_field():
    from schemas.models import SoccerAnalystResponse
    resp_no_comparison = SoccerAnalystResponse(
        question="test", sql="SELECT 1", data=[], chart={}, row_count=0
    )
    assert resp_no_comparison.comparison is None

    resp_with_comparison = SoccerAnalystResponse(
        question="test", sql="SELECT 1", data=[], chart={}, row_count=0,
        comparison={"title": "A vs B", "axes": ["Goals/90"], "series": []},
    )
    assert resp_with_comparison.comparison["title"] == "A vs B"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/test_analyst.py::test_comparison_query_plan_parses tests/soccer/test_analyst.py::test_soccer_analyst_response_accepts_comparison_field -v
```

Expected: FAIL — `ImportError: cannot import name 'ComparisonQueryPlan'`

- [ ] **Step 3: Add `ComparisonQueryPlan` and update `SoccerAnalystResponse` in `schemas/models.py`**

Add after the existing `SoccerQueryPlan` class (around line 196):

```python
class ComparisonQueryPlan(BaseModel):
    is_comparison: Literal[True]
    player_a: str
    season_a: int
    player_b: str
    season_b: int
    title: str
```

Update `SoccerAnalystResponse` (replace the existing class):

```python
class SoccerAnalystResponse(BaseModel):
    question: str
    sql: str
    data: list[dict]
    chart: dict
    row_count: int
    comparison: Optional[dict] = None
```

- [ ] **Step 4: Run the new tests to confirm they pass**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/test_analyst.py::test_comparison_query_plan_parses tests/soccer/test_analyst.py::test_soccer_analyst_response_accepts_comparison_field -v
```

Expected: 2 PASS

- [ ] **Step 5: Run the full soccer test suite to confirm nothing regressed**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/ -v
```

Expected: 15 passed (13 existing + 2 new)

- [ ] **Step 6: Commit**

```bash
git -C /home/jphill/Projects/agentic-repos/agentic-server add schemas/models.py tests/soccer/test_analyst.py
git -C /home/jphill/Projects/agentic-repos/agentic-server commit -m "feat: add ComparisonQueryPlan model and comparison field to SoccerAnalystResponse"
```

---

## Task 2: Prompt extension and plan parsing

**Files:**
- Modify: `workflows/soccer/analyst.py`
- Test: `tests/soccer/test_analyst.py`

**Interfaces:**
- Consumes: `ComparisonQueryPlan` from `schemas/models.py` (Task 1)
- Produces: `_parse_plan(raw: str) -> ComparisonQueryPlan | SoccerQueryPlan` — consumed by `_call_llm_sync` (this task) and indirectly by Task 3 via the return type of `_call_llm_sync`

- [ ] **Step 1: Write the failing tests**

Add to `tests/soccer/test_analyst.py`:

```python
def test_parse_plan_returns_regular_plan():
    from workflows.soccer.analyst import _parse_plan
    raw = '{"is_comparison": false, "sql": "SELECT 1", "chart_type": "bar", "x_column": "a", "y_column": "b", "title": "T"}'
    from schemas.models import SoccerQueryPlan
    plan = _parse_plan(raw)
    assert isinstance(plan, SoccerQueryPlan)
    assert plan.sql == "SELECT 1"


def test_parse_plan_returns_comparison_plan():
    from workflows.soccer.analyst import _parse_plan
    raw = '{"is_comparison": true, "player_a": "Kante", "season_a": 1617, "player_b": "Caicedo", "season_b": 2425, "title": "K vs C"}'
    from schemas.models import ComparisonQueryPlan
    plan = _parse_plan(raw)
    assert isinstance(plan, ComparisonQueryPlan)
    assert plan.player_a == "Kante"
    assert plan.season_a == 1617


def test_build_prompt_includes_is_comparison_instruction():
    from workflows.soccer.analyst import _build_prompt
    prompt = _build_prompt("compare kante and caicedo", league=None)
    assert "is_comparison" in prompt
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/test_analyst.py::test_parse_plan_returns_regular_plan tests/soccer/test_analyst.py::test_parse_plan_returns_comparison_plan tests/soccer/test_analyst.py::test_build_prompt_includes_is_comparison_instruction -v
```

Expected: FAIL — `ImportError: cannot import name '_parse_plan'`

- [ ] **Step 3: Add `import json` to `workflows/soccer/analyst.py`**

Add at the top of the imports block (after `import asyncio`):

```python
import json
```

- [ ] **Step 4: Add `_parse_plan` to `workflows/soccer/analyst.py`**

Add after the `_client_lock` declaration (around line 101, before `_get_client`):

```python
def _parse_plan(raw: str) -> "ComparisonQueryPlan | SoccerQueryPlan":
    from schemas.models import ComparisonQueryPlan as _CQP
    data = json.loads(raw)
    if data.get("is_comparison"):
        return _CQP.model_validate(data)
    return SoccerQueryPlan.model_validate(data)
```

Update the import at the top of `workflows/soccer/analyst.py` to include `ComparisonQueryPlan`:

```python
from schemas.models import SoccerAnalystRequest, SoccerAnalystResponse, SoccerQueryPlan, ComparisonQueryPlan
```

Then rewrite `_parse_plan` without the local import:

```python
def _parse_plan(raw: str) -> ComparisonQueryPlan | SoccerQueryPlan:
    data = json.loads(raw)
    if data.get("is_comparison"):
        return ComparisonQueryPlan.model_validate(data)
    return SoccerQueryPlan.model_validate(data)
```

- [ ] **Step 5: Update `_call_llm_sync` to use `_parse_plan`**

Replace the last line of `_call_llm_sync` (the `return SoccerQueryPlan.model_validate_json(raw.strip())` line) with:

```python
    return _parse_plan(raw.strip())
```

Also update the return type annotation on `_call_llm_sync`:

```python
def _call_llm_sync(client: openai_lib.OpenAI, prompt: str) -> ComparisonQueryPlan | SoccerQueryPlan:
```

- [ ] **Step 6: Extend `_build_prompt` with the comparison instruction**

At the end of `_build_prompt`, replace the final `return (...)` block's last few lines. Currently the return ends with:

```python
        "- title: descriptive chart title"
    )
```

Change it to:

```python
        "- title: descriptive chart title\n\n"
        "If the question asks to compare two specific players across seasons, return this shape instead:\n"
        '  {"is_comparison": true, "player_a": "...", "season_a": YYZZ, "player_b": "...", "season_b": YYZZ, "title": "..."}\n'
        "Use the same YYZZ season encoding described above (e.g. 2016-17 → 1617).\n"
        "For all other questions, include \"is_comparison\": false alongside the existing fields."
    )
```

- [ ] **Step 7: Run the new tests**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/test_analyst.py::test_parse_plan_returns_regular_plan tests/soccer/test_analyst.py::test_parse_plan_returns_comparison_plan tests/soccer/test_analyst.py::test_build_prompt_includes_is_comparison_instruction -v
```

Expected: 3 PASS

- [ ] **Step 8: Run the full soccer test suite**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/ -v
```

Expected: 18 passed (15 from Task 1 + 3 new). The existing `test_call_llm_sync_*` tests still pass because mock LLM responses without `is_comparison` route to `SoccerQueryPlan` via `_parse_plan`.

- [ ] **Step 9: Commit**

```bash
git -C /home/jphill/Projects/agentic-repos/agentic-server add workflows/soccer/analyst.py tests/soccer/test_analyst.py
git -C /home/jphill/Projects/agentic-repos/agentic-server commit -m "feat: add _parse_plan dispatcher and extend _build_prompt with comparison instruction"
```

---

## Task 3: Comparison execution and `run_soccer_analyst` branch

**Files:**
- Modify: `workflows/soccer/analyst.py`
- Test: `tests/soccer/test_analyst.py`

**Interfaces:**
- Consumes: `ComparisonQueryPlan` from `schemas/models.py` (Task 1); `_parse_plan` / updated `_call_llm_sync` from Task 2
- Produces: `_build_comparison(con, plan) -> tuple[str, list[dict], dict]`; updated `run_soccer_analyst` that returns `comparison` field in response

- [ ] **Step 1: Write the failing tests**

Add a helper and three tests to `tests/soccer/test_analyst.py`:

```python
def _make_comparison_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE player_season (
            player TEXT, team TEXT, position TEXT,
            matches INT, minutes INT,
            goals INT, xg FLOAT, np_goals INT, np_xg FLOAT,
            assists INT, xa FLOAT,
            shots INT, key_passes INT,
            yellow_cards INT, red_cards INT,
            season INT, league TEXT
        )
    """)
    con.execute("""
        INSERT INTO player_season VALUES
            ('Kante',       'Chelsea',  'MF', 30, 2700, 3,  0.5, 3,  0.4, 7,  1.0, 45, 90, 1, 0, 1617, 'ENG-Premier League'),
            ('Caicedo',     'Brighton', 'MF', 32, 2880, 5,  0.8, 5,  0.7, 4,  0.6, 35, 60, 3, 0, 2425, 'ENG-Premier League'),
            ('OtherPlayer', 'Arsenal',  'FW', 28, 2520, 15, 2.1, 12, 1.8, 8,  1.2, 20, 40, 2, 0, 1617, 'ENG-Premier League')
    """)
    return con


def test_build_comparison_returns_radar_figure():
    from workflows.soccer.analyst import _build_comparison
    from schemas.models import ComparisonQueryPlan
    con = _make_comparison_con()
    plan = ComparisonQueryPlan(
        is_comparison=True,
        player_a="Kante",
        season_a=1617,
        player_b="Caicedo",
        season_b=2425,
        title="Kante vs Caicedo",
    )
    sql, data, radar = _build_comparison(con, plan)
    assert radar["title"] == "Kante vs Caicedo"
    assert len(radar["axes"]) == 6
    assert len(radar["series"]) == 2
    assert all(len(s["values"]) == 6 for s in radar["series"])
    assert all(0 <= v <= 100 for s in radar["series"] for v in s["values"])


def test_build_comparison_raises_if_both_players_not_found():
    from workflows.soccer.analyst import _build_comparison, SoccerAnalystError
    from schemas.models import ComparisonQueryPlan
    con = _make_comparison_con()
    plan = ComparisonQueryPlan(
        is_comparison=True,
        player_a="Nobody",
        season_a=1617,
        player_b="AlsoNobody",
        season_b=2425,
        title="Test",
    )
    with pytest.raises(SoccerAnalystError):
        _build_comparison(con, plan)


def test_build_comparison_raises_if_only_one_player_found():
    from workflows.soccer.analyst import _build_comparison, SoccerAnalystError
    from schemas.models import ComparisonQueryPlan
    con = _make_comparison_con()
    plan = ComparisonQueryPlan(
        is_comparison=True,
        player_a="Kante",
        season_a=1617,
        player_b="Nobody",
        season_b=2425,
        title="Test",
    )
    with pytest.raises(SoccerAnalystError):
        _build_comparison(con, plan)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/test_analyst.py::test_build_comparison_returns_radar_figure tests/soccer/test_analyst.py::test_build_comparison_raises_if_both_players_not_found tests/soccer/test_analyst.py::test_build_comparison_raises_if_only_one_player_found -v
```

Expected: FAIL — `ImportError: cannot import name '_build_comparison'`

- [ ] **Step 3: Add constants and `_build_comparison` to `workflows/soccer/analyst.py`**

Add these constants after the `_TABLE_SCHEMAS` dict (around line 48):

```python
_COMPARISON_SQL = """\
WITH base AS (
    SELECT player, season,
        goals       * 90.0 / NULLIF(minutes, 0) AS goals_p90,
        assists     * 90.0 / NULLIF(minutes, 0) AS assists_p90,
        xg          * 90.0 / NULLIF(minutes, 0) AS xg_p90,
        xa          * 90.0 / NULLIF(minutes, 0) AS xa_p90,
        key_passes  * 90.0 / NULLIF(minutes, 0) AS key_passes_p90,
        shots       * 90.0 / NULLIF(minutes, 0) AS shots_p90
    FROM player_season
    WHERE minutes > 0
),
ranked AS (
    SELECT player, season,
        ROUND(PERCENT_RANK() OVER (ORDER BY goals_p90)       * 100) AS goals_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY assists_p90)     * 100) AS assists_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY xg_p90)          * 100) AS xg_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY xa_p90)          * 100) AS xa_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY key_passes_p90)  * 100) AS key_passes_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY shots_p90)       * 100) AS shots_p90
    FROM base
)
SELECT * FROM ranked
WHERE (player ILIKE ? AND season = ?)
   OR (player ILIKE ? AND season = ?)
"""

_RADAR_AXES = ["Goals/90", "Assists/90", "xG/90", "xA/90", "Key Passes/90", "Shots/90"]
_RADAR_COLS = ["goals_p90", "assists_p90", "xg_p90", "xa_p90", "key_passes_p90", "shots_p90"]
```

Add the `_build_comparison` function after `_build_plotly_figure` (around line 206):

```python
def _build_comparison(
    con: duckdb.DuckDBPyConnection,
    plan: ComparisonQueryPlan,
) -> tuple[str, list[dict], dict]:
    params = [plan.player_a, plan.season_a, plan.player_b, plan.season_b]
    try:
        df = con.execute(_COMPARISON_SQL, params).pl()
    except Exception as e:
        raise SoccerAnalystError(str(e), _COMPARISON_SQL)

    if len(df) == 0:
        raise SoccerAnalystError(
            "No data found for one or both players", _COMPARISON_SQL
        )
    if len(df) < 2:
        raise SoccerAnalystError(
            "Could not find data for both players in the specified seasons",
            _COMPARISON_SQL,
        )

    series = [
        {
            "name": row["player"],
            "values": [int(row.get(col) or 0) for col in _RADAR_COLS],
        }
        for row in df.to_dicts()
    ]
    radar_figure = {"title": plan.title, "axes": _RADAR_AXES, "series": series}
    return _COMPARISON_SQL, df.to_dicts(), radar_figure
```

- [ ] **Step 4: Run the new tests**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/test_analyst.py::test_build_comparison_returns_radar_figure tests/soccer/test_analyst.py::test_build_comparison_raises_if_both_players_not_found tests/soccer/test_analyst.py::test_build_comparison_raises_if_only_one_player_found -v
```

Expected: 3 PASS

- [ ] **Step 5: Branch `run_soccer_analyst` on plan type**

In `run_soccer_analyst` (around line 208), after the line `plan = await asyncio.to_thread(_call_llm_sync, client, prompt)`, add the comparison branch **before** the existing `try:` block:

```python
    if isinstance(plan, ComparisonQueryPlan):
        sql, data, comparison = await asyncio.to_thread(_build_comparison, con, plan)
        return SoccerAnalystResponse(
            question=request.question,
            sql=sql,
            data=data,
            chart={"data": [], "layout": {}},
            row_count=len(data),
            comparison=comparison,
        )
```

The existing `try: df = await asyncio.to_thread(_execute_sql, ...)` block and everything after it remains completely unchanged.

- [ ] **Step 6: Run the full soccer test suite**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest tests/soccer/ -v
```

Expected: 21 passed (18 from Tasks 1–2 + 3 new)

- [ ] **Step 7: Run the full test suite**

```bash
cd /home/jphill/Projects/agentic-repos/agentic-server && uv run pytest -v
```

Expected: all existing tests pass plus the 8 new soccer tests.

- [ ] **Step 8: Commit**

```bash
git -C /home/jphill/Projects/agentic-repos/agentic-server add workflows/soccer/analyst.py tests/soccer/test_analyst.py
git -C /home/jphill/Projects/agentic-repos/agentic-server commit -m "feat: add _build_comparison with percentile SQL and branch run_soccer_analyst for comparison queries"
```

---

## Self-Review

**Spec coverage:**
- ✅ `ComparisonQueryPlan` with `is_comparison: Literal[True]`, `player_a`, `season_a`, `player_b`, `season_b`, `title` — Task 1
- ✅ `SoccerAnalystResponse.comparison: Optional[dict] = None` — Task 1
- ✅ `SoccerQueryPlan` unchanged — not touched in any task
- ✅ `_build_prompt` extended with `is_comparison` instruction — Task 2
- ✅ `_parse_plan` dispatches on `is_comparison` flag — Task 2
- ✅ `_call_llm_sync` updated to return `ComparisonQueryPlan | SoccerQueryPlan` — Task 2
- ✅ Fixed DuckDB `PERCENT_RANK()` window SQL in `_COMPARISON_SQL` — Task 3
- ✅ Fixed axes: `["Goals/90", "Assists/90", "xG/90", "xA/90", "Key Passes/90", "Shots/90"]` — Task 3
- ✅ `ILIKE` for player name matching — Task 3
- ✅ 0-row error → `SoccerAnalystError` — Task 3
- ✅ 1-row error → `SoccerAnalystError` — Task 3
- ✅ Regular path in `run_soccer_analyst` unchanged — Task 3 adds branch before existing code, does not modify it
- ✅ `comparison` field populated in response for comparison queries — Task 3
- ✅ `chart` stub `{"data": [], "layout": {}}` in comparison response — Task 3
- ✅ All 7 spec tests covered: `test_comparison_query_plan_parses`, `test_soccer_analyst_response_accepts_comparison_field`, `test_parse_plan_returns_regular_plan`, `test_parse_plan_returns_comparison_plan`, `test_build_prompt_includes_is_comparison_instruction`, `test_build_comparison_returns_radar_figure`, `test_build_comparison_raises_if_both_players_not_found`, `test_build_comparison_raises_if_only_one_player_found`

**Placeholder scan:** None found. All code, SQL, test data, and commands are fully specified.

**Type consistency:**
- `ComparisonQueryPlan` defined in Task 1 → imported in Task 2 (`_parse_plan`) and Task 3 (`_build_comparison` signature, `isinstance` check)
- `_build_comparison` return type `tuple[str, list[dict], dict]` defined in Task 3 → consumed in same task by `run_soccer_analyst`
- `_COMPARISON_SQL`, `_RADAR_AXES`, `_RADAR_COLS` defined as module-level constants in Task 3 → used in `_build_comparison` in same task
