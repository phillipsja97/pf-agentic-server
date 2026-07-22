# Soccer Player Comparison Backend — Design Spec

**Date:** 2026-07-21
**Status:** Approved

## Overview

When a user asks the soccer analyst to compare two players across seasons (e.g. "Compare Ngolo Kante 2017 stats with Moises Caicedo 2025 in the Premier League"), the `/workflows/soccer-analyst` endpoint detects the comparison intent in its single LLM call and returns a `RadarFigure` in the `comparison` field of the response. The frontend already handles this field — it renders a dither-kit radar chart overlay. The existing regular query path is completely unchanged.

---

## Approach

**One-shot LLM discriminated plan.** `_build_prompt` tells the LLM it may return either a regular query plan (`is_comparison: false`) or a comparison plan (`is_comparison: true`). The backend parses the flag and branches: regular queries follow the existing path; comparison queries run a fixed DuckDB percentile window-function query — no extra LLM call needed.

---

## Data Shape (`schemas/models.py`)

### New: `ComparisonQueryPlan`

```python
class ComparisonQueryPlan(BaseModel):
    is_comparison: Literal[True]
    player_a: str    # e.g. "N'Golo Kante"
    season_a: int    # YYZZ encoded, e.g. 1617 for 2016-17
    player_b: str
    season_b: int
    title: str       # e.g. "Kante 2016-17 vs Caicedo 2024-25 — Premier League"
```

### Updated: `SoccerAnalystResponse`

One new optional field — all existing fields and their types are unchanged:

```python
class SoccerAnalystResponse(BaseModel):
    question: str
    sql: str
    data: list[dict]
    chart: dict
    row_count: int
    comparison: Optional[dict] = None   # RadarFigure when present, None otherwise
```

`SoccerQueryPlan` is **unchanged**. Existing tests remain valid.

### RadarFigure shape (what goes in `comparison`)

```json
{
  "title": "Kante 2016-17 vs Caicedo 2024-25 — Premier League",
  "axes": ["Goals/90", "Assists/90", "xG/90", "xA/90", "Key Passes/90", "Shots/90"],
  "series": [
    { "name": "N'Golo Kante", "values": [45, 60, 42, 55, 78, 30] },
    { "name": "Moises Caicedo", "values": [38, 55, 36, 48, 71, 28] }
  ]
}
```

Values are 0–100 integer percentile ranks within the scoped dataset.

---

## Prompt Extension (`_build_prompt`)

The existing prompt is extended with one new paragraph at the end (after the existing JSON field instructions):

```
If the question asks to compare two specific players across seasons, return this shape instead:
  {"is_comparison": true, "player_a": "...", "season_a": YYZZ, "player_b": "...", "season_b": YYZZ, "title": "..."}
Use the same YYZZ season encoding described above.
For all other questions, return {"is_comparison": false, ...} with the existing fields.
```

---

## Plan Parsing (`_parse_plan`)

New helper function replaces direct `SoccerQueryPlan.model_validate_json`:

```python
def _parse_plan(raw: str) -> ComparisonQueryPlan | SoccerQueryPlan:
    data = json.loads(raw)
    if data.get("is_comparison"):
        return ComparisonQueryPlan.model_validate(data)
    return SoccerQueryPlan.model_validate(data)
```

`_call_llm_sync` returns `ComparisonQueryPlan | SoccerQueryPlan` and calls `_parse_plan` instead of `SoccerQueryPlan.model_validate_json`.

---

## Comparison Execution (`_build_comparison`)

```python
def _build_comparison(
    con: duckdb.DuckDBPyConnection,
    plan: ComparisonQueryPlan,
) -> tuple[str, list[dict], dict]:
```

Returns `(sql, data_rows, radar_figure)`.

**SQL** — one query using window functions, scoped to the connection's loaded data:

```sql
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
WHERE (player ILIKE :player_a AND season = :season_a)
   OR (player ILIKE :player_b AND season = :season_b)
```

**Axes (fixed):** `["Goals/90", "Assists/90", "xG/90", "xA/90", "Key Passes/90", "Shots/90"]`

**Error handling:**
- 0 rows returned → `SoccerAnalystError("No data found for one or both players", sql)`
- 1 row returned (one player missing) → `SoccerAnalystError("Could not find data for both players in the specified seasons", sql)`
- 2+ rows returned → build `RadarFigure` (take first row per player if duplicates)

**`run_soccer_analyst` branch:**

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
# existing regular path follows unchanged
```

---

## Error Handling

All errors follow the existing pattern — `SoccerAnalystError(message, sql)` propagates to the router, which returns 422 with `{"error": ..., "sql": ...}`. No new error types, no router changes.

| Scenario | Behavior |
|---|---|
| Player not found or no data for season | `SoccerAnalystError` → 422 |
| Only one player found | `SoccerAnalystError` → 422 |
| LLM returns malformed JSON | Existing `model_validate_json` exception → router 422 |
| Regular query (non-comparison) | Existing path, `comparison` field is `None` |

---

## Testing (`tests/soccer/test_analyst.py`)

New tests added to the existing file, using `unittest.mock` (no pytest-mock):

| Test | What it verifies |
|---|---|
| `test_parse_plan_returns_comparison_plan` | `_parse_plan` with `is_comparison: true` → `ComparisonQueryPlan` |
| `test_parse_plan_returns_regular_plan` | `_parse_plan` with `is_comparison: false` → `SoccerQueryPlan` |
| `test_build_prompt_includes_is_comparison_instruction` | `_build_prompt` output contains `"is_comparison"` |
| `test_soccer_analyst_response_has_comparison_field` | `SoccerAnalystResponse` accepts `comparison=None` and `comparison={...}` |
| `test_build_comparison_returns_radar_figure` | In-memory DuckDB fixture → correct `RadarFigure` shape |
| `test_build_comparison_raises_if_player_not_found` | 0-row result → `SoccerAnalystError` |
| `test_build_comparison_raises_if_only_one_player_found` | 1-row result → `SoccerAnalystError` |

---

## Out of Scope

- CFB comparison (soccer only)
- Team vs. team comparison
- More than two players on one radar
- Fuzzy name matching beyond `ILIKE` (exact prefix/suffix matching)
- Percentile scope customisation (always scoped to the loaded connection's dataset)
