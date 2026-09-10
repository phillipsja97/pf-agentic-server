# CFB Matchup Predictor — Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three endpoints — `POST /workflows/cfb/matchup`, `GET /workflows/cfb/slate`, `GET /workflows/cfb/teams` — that serve predictions from the `cfb_team_profile`, `cfb_model`, and `cfb_games` Delta tables with no LLM in the path.

**Architecture:** `workflows/cfb/matchup.py` holds pure functions (prediction formula over stored coefficients, radar/edge assembly, team suggestions) and a TTL-cached loader that reads the three tables into Polars. Three async runners assemble Pydantic responses. Routes live in `routers/workflows.py` nested under `/cfb/`, immune to the `GET /{job_id}` catch-all.

**Tech Stack:** FastAPI, Pydantic, Polars, `deltalake`, pytest + `TestClient` + `unittest.mock`.

**Spec:** `~/Projects/sports-stats/sports-etl/docs/superpowers/specs/2026-09-09-cfb-matchup-predictor-design.md`

**Prerequisite:** the ETL plan (`sports-etl/docs/superpowers/plans/2026-09-09-cfb-matchup-etl.md`) has been executed and its backfill run, so the three tables exist in MinIO. Tests here do not need MinIO; the manual smoke check in Task 5 does.

## Global Constraints

- Run every command from `~/Projects/agentic-repos/agentic-server/`. `uv run pytest` is the suite; `tests/routers/test_briefing_config.py::test_trigger_briefing_creates_job` fails before this work starts and is unrelated — do not chase it.
- Tests live under `tests/cfb/`, use `fastapi.testclient.TestClient(app)` for routes, and patch `routers.workflows.run_<name>` with `AsyncMock` (pattern: `tests/cfb/test_cfb_analyst_router.py`).
- Settings are read from `config.settings`; tests override with `monkeypatch.setattr(settings, ...)`.
- No LLM, no DuckDB, no scikit-learn in this path. Coefficients are plain floats from the `cfb_model` row.
- Route paths are `/workflows/cfb/matchup` (POST), `/workflows/cfb/slate` (GET), `/workflows/cfb/teams` (GET). No auth dependency, matching the analyst routes.
- Cache TTL is 1800 seconds. Every response carries `data_through_date`.
- Team-name miss → HTTP 422 with `{"error": ..., "suggestions": [...]}`.
- Commit after every task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|---|---|
| `schemas/models.py` (modify) | request/response models for the three endpoints |
| `workflows/cfb/matchup.py` (create) | pure functions, TTL loader, three runners, `CfbMatchupError` |
| `routers/workflows.py` (modify) | three routes |
| `workflows/cfb/analyst.py` (modify) | season-range text in the LLM prompt |
| `tests/cfb/test_models.py` (modify) | request defaults |
| `tests/cfb/test_matchup.py` (create) | pure functions + runners against in-memory frames |
| `tests/cfb/test_matchup_router.py` (create) | route wiring and error shaping |
| `CLAUDE.md` (modify) | remove the stale "no test suite" line; add the endpoints |

---

### Task 1: Schemas and the analyst prompt range

**Files:**
- Modify: `schemas/models.py`
- Modify: `workflows/cfb/analyst.py` (two string literals)
- Modify: `tests/cfb/test_models.py`

**Interfaces:**
- Produces (all in `schemas.models`): `CfbMatchupRequest`, `CfbTeamRecord`, `CfbPrediction`, `CfbCalibrationBin`, `CfbModelRecord`, `CfbEdge`, `CfbH2HGame`, `CfbMatchupResponse`, `CfbSlateGame`, `CfbSlateResponse`, `CfbTeam`

- [ ] **Step 1: Write the failing test**

Append to `tests/cfb/test_models.py`:

```python
from schemas.models import CfbMatchupRequest, CfbMatchupResponse, CfbEdge


def test_cfb_matchup_request_defaults():
    req = CfbMatchupRequest(home_team="Georgia", away_team="Alabama")
    assert req.neutral_site is False
    assert req.season is None and req.through_week is None


def test_cfb_edge_favors_is_constrained():
    CfbEdge(stat="points", home_value=30.0, away_value=20.0, diff=10.0, favors="home", unit="pts")
    import pytest
    with pytest.raises(Exception):
        CfbEdge(stat="points", home_value=30.0, away_value=20.0, diff=10.0, favors="left", unit="pts")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cfb/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'CfbMatchupRequest'`

- [ ] **Step 3: Add the models**

Append to `schemas/models.py` after `CfbAnalystRequest`:

```python
# --- cfb matchup predictor models ---

class CfbMatchupRequest(BaseModel):
    home_team: str
    away_team: str
    neutral_site: bool = False
    season: Optional[int] = None        # default: latest season in cfb_team_profile
    through_week: Optional[int] = None  # default: latest fetched week of that season


class CfbTeamRecord(BaseModel):
    team: str
    conference: Optional[str] = None
    wins: int
    losses: int
    games_played: int


class CfbPrediction(BaseModel):
    home_win_prob: float
    away_win_prob: float
    projected_home_points: float
    projected_away_points: float
    expected_margin: float


class CfbCalibrationBin(BaseModel):
    lo: float
    hi: float
    n: int
    predicted: Optional[float] = None
    actual: Optional[float] = None


class CfbModelRecord(BaseModel):
    accuracy: float
    brier: float
    log_loss: float
    elo_baseline_accuracy: float
    elo_baseline_brier: float
    calibration_bins: list[CfbCalibrationBin]
    trained_through_season: int
    shrink_k: int


class CfbEdge(BaseModel):
    stat: str
    home_value: Optional[float]
    away_value: Optional[float]
    diff: Optional[float]           # home - away, in the stat's own unit
    favors: Literal["home", "away", "even"]
    unit: str


class CfbH2HGame(BaseModel):
    season: int
    week: int
    season_type: str
    date: Optional[str] = None
    home_team: str
    away_team: str
    home_points: int
    away_points: int


class CfbMatchupResponse(BaseModel):
    season: int
    through_week: int
    data_through_date: Optional[str]
    neutral_site: bool
    home: CfbTeamRecord
    away: CfbTeamRecord
    prediction: CfbPrediction
    model: CfbModelRecord
    comparison: dict                # RadarFigure: {title, axes, series[{name, values}]}
    edges: list[CfbEdge]
    head_to_head: list[CfbH2HGame]


class CfbSlateGame(BaseModel):
    game_id: int
    week: int
    start_date: Optional[str] = None
    home_team: str
    away_team: str
    neutral_site: bool
    home_win_prob: float
    projected_home_points: float
    projected_away_points: float


class CfbSlateResponse(BaseModel):
    season: int
    week: int
    through_week: int
    data_through_date: Optional[str]
    games: list[CfbSlateGame]


class CfbTeam(BaseModel):
    team: str
    conference: Optional[str] = None
```

`Literal` and `Optional` are already imported at the top of `schemas/models.py`.

- [ ] **Step 4: Update the analyst prompt's season range**

In `workflows/cfb/analyst.py`, change both literals — in `_SYSTEM_PROMPT` `"Available seasons: 2015 through 2024. "` → `"Available seasons: 2015 through 2026. "`, and in `_build_prompt` `"Available seasons: 2015–2024 only."` → `"Available seasons: 2015–2026 only."`. The ETL backfill has ingested 2025 and 2026 into the same tables the analyst loads.

- [ ] **Step 5: Run and commit**

Run: `uv run pytest tests/cfb -v`
Expected: PASS

```bash
git add schemas/models.py workflows/cfb/analyst.py tests/cfb/test_models.py
git commit -m "feat: add CFB matchup request/response models; extend analyst season range

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Pure functions — predict, radar, edges, suggestions, week resolution

**Files:**
- Create: `workflows/cfb/matchup.py`
- Create: `tests/cfb/test_matchup.py`

**Interfaces:**
- Produces (in `workflows.cfb.matchup`):
  - `predict(model_row: dict, elo_diff: float, adj_diff: float, home_field: float) -> CfbPrediction`
  - `build_radar(home: dict, away: dict, title: str) -> dict`
  - `build_edges(home: dict, away: dict) -> list[CfbEdge]`
  - `suggest_teams(name: str, names: list[str]) -> list[str]`
  - `latest_through_week(profile: pl.DataFrame, season: int) -> int`
  - `class CfbMatchupError(Exception)` with `.suggestions: list[str]`
  - `RADAR_AXES`, `EDGE_STATS` constants

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_matchup.py`:

```python
import math
import polars as pl
import pytest
from workflows.cfb.matchup import (
    predict, build_radar, build_edges, suggest_teams, latest_through_week, CfbMatchupError, RADAR_AXES,
)


MODEL_ROW = {
    "win_intercept": 0.0, "win_b_elo": 0.01, "win_b_adj": 0.1, "win_b_home": 0.5,
    "margin_intercept": 0.0, "margin_b_elo": 0.04, "margin_b_adj": 0.6, "margin_b_home": 2.5,
    "total_intercept": 50.0, "total_b_elo": 0.0, "total_b_adj": 0.0, "total_b_home": 0.0,
}


def _row(team, **kw):
    base = {"team": team, "conference": "SEC", "games": 4, "wins": 3, "losses": 1,
            "points_pg": 30.0, "points_allowed_pg": 20.0, "rush_yds_pg": 200.0, "rush_yds_allowed_pg": 120.0,
            "pass_yds_pg": 250.0, "pass_yds_allowed_pg": 200.0, "turnover_margin_pg": 0.5, "third_down_pct": 45.0,
            "pct_points": 80.0, "pct_points_allowed": 70.0, "pct_rush": 90.0, "pct_rush_allowed": 60.0,
            "pct_pass": 50.0, "pct_pass_allowed": 40.0, "pct_turnover_margin": 55.0, "pct_third_down": 65.0,
            "elo": 1700.0, "adj_margin": 8.0}
    return {**base, **kw}


def test_predict_applies_all_three_heads():
    p = predict(MODEL_ROW, elo_diff=100.0, adj_diff=5.0, home_field=1.0)
    logit = 0.01 * 100 + 0.1 * 5 + 0.5          # 2.0
    assert p.home_win_prob == pytest.approx(1 / (1 + math.exp(-logit)))
    assert p.away_win_prob == pytest.approx(1 - p.home_win_prob)
    assert p.expected_margin == pytest.approx(0.04 * 100 + 0.6 * 5 + 2.5)   # 9.5
    assert p.projected_home_points == pytest.approx((50 + 9.5) / 2)
    assert p.projected_away_points == pytest.approx((50 - 9.5) / 2)


def test_predict_neutral_site_has_no_home_term():
    p = predict(MODEL_ROW, 0.0, 0.0, home_field=0.0)
    assert p.home_win_prob == pytest.approx(0.5)


def test_build_radar_is_a_radar_figure_of_percentiles():
    fig = build_radar(_row("Georgia"), _row("Alabama", pct_points=20.0), "Georgia vs Alabama")
    assert fig["title"] == "Georgia vs Alabama"
    assert fig["axes"] == [label for label, _ in RADAR_AXES]
    assert len(fig["axes"]) == 6
    assert fig["series"][0] == {"name": "Georgia", "values": [80.0, 70.0, 90.0, 60.0, 50.0, 40.0]}
    assert fig["series"][1]["values"][0] == 20.0


def test_build_radar_nulls_become_zero():
    fig = build_radar(_row("A", pct_points=None), _row("B"), "t")
    assert fig["series"][0]["values"][0] == 0.0


def test_build_edges_sign_and_favors():
    edges = build_edges(_row("A", points_pg=35.0, points_allowed_pg=10.0, pass_yds_pg=200.0),
                        _row("B", points_pg=25.0, points_allowed_pg=20.0, pass_yds_pg=200.0))
    by = {e.stat: e for e in edges}
    assert len(edges) == 8
    assert by["points"].diff == 10.0 and by["points"].favors == "home" and by["points"].unit == "pts"
    # allowed: lower is better, so home allowing 10 vs 20 favors home even though diff is negative
    assert by["points allowed"].diff == -10.0 and by["points allowed"].favors == "home"
    assert by["pass yds"].favors == "even"


def test_build_edges_null_values():
    edges = build_edges(_row("A", points_pg=None), _row("B"))
    pts = [e for e in edges if e.stat == "points"][0]
    assert pts.home_value is None and pts.diff is None and pts.favors == "even"


def test_suggest_teams_returns_close_matches_not_exact():
    names = ["Georgia", "Georgia Tech", "Georgia Southern", "Alabama", "Ohio State"]
    assert suggest_teams("georgia", names)[:3] == ["Georgia", "Georgia Tech", "Georgia Southern"]
    assert suggest_teams("Ohio St", names) == ["Ohio State"]
    assert suggest_teams("zzzz", names) == []


def test_latest_through_week_is_max_fetched_week_for_season():
    profile = pl.DataFrame({"season": [2026, 2026, 2026, 2025], "through_week": [0, 1, 2, 15], "team": ["A"] * 4})
    assert latest_through_week(profile, 2026) == 2
    assert latest_through_week(profile, 2025) == 15
    with pytest.raises(CfbMatchupError):
        latest_through_week(profile, 2030)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/cfb/test_matchup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.cfb.matchup'`

- [ ] **Step 3: Implement the pure functions**

Create `workflows/cfb/matchup.py`:

```python
"""CFB matchup predictor serving layer.

Reads cfb_team_profile / cfb_model / cfb_games from MinIO (TTL-cached) and
applies the stored linear coefficients. No LLM, no SQL generation.
"""
import asyncio
import difflib
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

import polars as pl
from deltalake import DeltaTable

from config import settings
from core.logging import logger
from schemas.models import (
    CfbCalibrationBin, CfbEdge, CfbH2HGame, CfbMatchupRequest, CfbMatchupResponse, CfbModelRecord,
    CfbPrediction, CfbSlateGame, CfbSlateResponse, CfbTeam, CfbTeamRecord,
)

CACHE_TTL_SECONDS = 1800

# (axis label, profile column) — percentile columns, allowed-stats already inverted upstream.
RADAR_AXES = [
    ("Points/gm", "pct_points"),
    ("Pts allowed", "pct_points_allowed"),
    ("Rush yds", "pct_rush"),
    ("Rush allowed", "pct_rush_allowed"),
    ("Pass yds", "pct_pass"),
    ("Pass allowed", "pct_pass_allowed"),
]

# (stat label, profile column, unit, lower_is_better)
EDGE_STATS = [
    ("points",           "points_pg",           "pts", False),
    ("points allowed",   "points_allowed_pg",   "pts", True),
    ("rush yds",         "rush_yds_pg",         "yds", False),
    ("rush yds allowed", "rush_yds_allowed_pg", "yds", True),
    ("pass yds",         "pass_yds_pg",         "yds", False),
    ("pass yds allowed", "pass_yds_allowed_pg", "yds", True),
    ("turnover margin",  "turnover_margin_pg",  "/gm", False),
    ("3rd down %",       "third_down_pct",      "%",   False),
]


class CfbMatchupError(Exception):
    def __init__(self, message: str, suggestions: Optional[list[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


def _linear(model_row: dict, head: str, elo_diff: float, adj_diff: float, home_field: float) -> float:
    return (
        model_row[f"{head}_intercept"]
        + model_row[f"{head}_b_elo"] * elo_diff
        + model_row[f"{head}_b_adj"] * adj_diff
        + model_row[f"{head}_b_home"] * home_field
    )


def predict(model_row: dict, elo_diff: float, adj_diff: float, home_field: float) -> CfbPrediction:
    p_home = 1.0 / (1.0 + math.exp(-_linear(model_row, "win", elo_diff, adj_diff, home_field)))
    margin = _linear(model_row, "margin", elo_diff, adj_diff, home_field)
    total = _linear(model_row, "total", elo_diff, adj_diff, home_field)
    return CfbPrediction(
        home_win_prob=p_home,
        away_win_prob=1.0 - p_home,
        projected_home_points=(total + margin) / 2.0,
        projected_away_points=(total - margin) / 2.0,
        expected_margin=margin,
    )


def build_radar(home: dict, away: dict, title: str) -> dict:
    def values(row: dict) -> list[float]:
        return [float(row.get(col) or 0.0) for _, col in RADAR_AXES]
    return {
        "title": title,
        "axes": [label for label, _ in RADAR_AXES],
        "series": [{"name": home["team"], "values": values(home)}, {"name": away["team"], "values": values(away)}],
    }


def build_edges(home: dict, away: dict) -> list[CfbEdge]:
    edges = []
    for label, col, unit, lower_is_better in EDGE_STATS:
        h, a = home.get(col), away.get(col)
        if h is None or a is None:
            edges.append(CfbEdge(stat=label, home_value=h, away_value=a, diff=None, favors="even", unit=unit))
            continue
        diff = float(h) - float(a)
        if abs(diff) < 1e-9:
            favors = "even"
        elif (diff > 0) != lower_is_better:
            favors = "home"
        else:
            favors = "away"
        edges.append(CfbEdge(stat=label, home_value=float(h), away_value=float(a), diff=diff, favors=favors, unit=unit))
    return edges


def suggest_teams(name: str, names: list[str], limit: int = 5) -> list[str]:
    lowered = {n.lower(): n for n in names}
    prefix = [lowered[k] for k in sorted(lowered) if k.startswith(name.lower())]
    fuzzy = [lowered[k] for k in difflib.get_close_matches(name.lower(), list(lowered), n=limit, cutoff=0.6)]
    out: list[str] = []
    for n in prefix + fuzzy:
        if n not in out:
            out.append(n)
    return out[:limit]


def latest_through_week(profile: pl.DataFrame, season: int) -> int:
    weeks = profile.filter(pl.col("season") == season)["through_week"]
    if weeks.is_empty():
        raise CfbMatchupError(f"No profile data for season {season}")
    return int(weeks.max())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/cfb/test_matchup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/cfb/matchup.py tests/cfb/test_matchup.py
git commit -m "feat: CFB matchup pure functions — predict, radar, edges, suggestions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: TTL store and the three runners

**Files:**
- Modify: `workflows/cfb/matchup.py`
- Modify: `tests/cfb/test_matchup.py`

**Interfaces:**
- Produces:
  - `@dataclass Store(profile: pl.DataFrame, games: pl.DataFrame, model_row: dict, loaded_at: float)`
  - `_load_store() -> Store` (sync; reads MinIO via `DeltaTable`) and `async _get_store() -> Store` (TTL + lock)
  - `async run_matchup(req: CfbMatchupRequest) -> CfbMatchupResponse`
  - `async run_slate(week: Optional[int]) -> CfbSlateResponse`
  - `async list_teams() -> list[CfbTeam]`
  - `model_record(model_row: dict) -> CfbModelRecord`

- [ ] **Step 1: Write the failing tests**

Append to `tests/cfb/test_matchup.py`:

```python
import asyncio
import json
from datetime import date
from unittest.mock import patch, MagicMock
import pyarrow as pa
import workflows.cfb.matchup as m
from workflows.cfb.matchup import Store, run_matchup, run_slate, list_teams, model_record
from schemas.models import CfbMatchupRequest


def _profile_df():
    rows = []
    for tw in (0, 1, 2):
        rows.append(_row("Georgia", season=2026, through_week=tw, games=tw, elo=1800.0 + tw, adj_margin=10.0))
        rows.append(_row("Alabama", season=2026, through_week=tw, games=tw, elo=1700.0 + tw, adj_margin=4.0))
        rows.append(_row("Georgia Tech", season=2026, through_week=tw, games=tw, elo=1500.0, adj_margin=-2.0, conference="ACC"))
    rows.append(_row("Georgia", season=2025, through_week=15, games=13, elo=1790.0, adj_margin=9.0))
    return pl.DataFrame(rows)


def _games_df():
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "season": [2026, 2026, 2026, 2024],
        "week": [1, 3, 3, 1],
        "season_type": ["regular", "regular", "regular", "postseason"],
        "completed": [True, False, False, True],
        "neutral_site": [False, False, True, True],
        "start_date": [date(2026, 9, 5), date(2026, 9, 19), date(2026, 9, 19), date(2025, 1, 1)],
        "home_team": ["Georgia", "Georgia", "Alabama", "Alabama"],
        "away_team": ["Alabama", "Georgia Tech", "Georgia", "Georgia"],
        "home_classification": ["fbs"] * 4,
        "away_classification": ["fbs"] * 4,
        "home_points": [27, None, None, 20],
        "away_points": [20, None, None, 23],
        "home_pregame_elo": [1800.0, 1802.0, 1702.0, 1750.0],
        "away_pregame_elo": [1700.0, 1500.0, 1802.0, 1760.0],
    })


def _model_row():
    return {**MODEL_ROW, "shrink_k": 4, "trained_through_season": 2025,
            "accuracy": 0.74, "brier": 0.18, "log_loss": 0.53,
            "elo_accuracy": 0.708, "elo_brier": 0.2, "elo_log_loss": 0.56,
            "calibration_bins": json.dumps([{"lo": 0.5, "hi": 0.6, "n": 10, "predicted": 0.55, "actual": 0.5}])}


def _store():
    return Store(profile=_profile_df(), games=_games_df(), model_row=_model_row(), loaded_at=time_now())


def time_now():
    import time
    return time.monotonic()


@pytest.fixture
def fake_store(monkeypatch):
    store = _store()
    monkeypatch.setattr(m, "_store", store)
    return store


def test_model_record_parses_bins_and_baseline():
    rec = model_record(_model_row())
    assert rec.accuracy == 0.74 and rec.elo_baseline_accuracy == 0.708 and rec.shrink_k == 4
    assert rec.calibration_bins[0].n == 10 and rec.calibration_bins[0].actual == 0.5


def test_run_matchup_uses_latest_week_and_profile_elo(fake_store):
    resp = asyncio.run(run_matchup(CfbMatchupRequest(home_team="Georgia", away_team="Alabama")))
    assert resp.season == 2026 and resp.through_week == 2
    assert resp.home.team == "Georgia" and resp.home.games_played == 2
    assert resp.data_through_date == "2026-09-05"
    expected = predict(_model_row(), elo_diff=100.0, adj_diff=6.0, home_field=1.0)
    assert resp.prediction.home_win_prob == pytest.approx(expected.home_win_prob)
    assert resp.comparison["series"][0]["name"] == "Georgia"
    assert len(resp.edges) == 8
    assert [g.season for g in resp.head_to_head] == [2026, 2024]     # completed only, newest first
    assert resp.model.trained_through_season == 2025


def test_run_matchup_neutral_and_explicit_week(fake_store):
    resp = asyncio.run(run_matchup(CfbMatchupRequest(home_team="Georgia", away_team="Alabama", neutral_site=True, through_week=0)))
    assert resp.through_week == 0 and resp.neutral_site is True
    expected = predict(_model_row(), elo_diff=100.0, adj_diff=6.0, home_field=0.0)
    assert resp.prediction.home_win_prob == pytest.approx(expected.home_win_prob)
    assert resp.data_through_date is None


def test_run_matchup_unknown_team_raises_with_suggestions(fake_store):
    with pytest.raises(m.CfbMatchupError) as ei:
        asyncio.run(run_matchup(CfbMatchupRequest(home_team="Georgi", away_team="Alabama")))
    assert "Georgia" in ei.value.suggestions and "Georgia Tech" in ei.value.suggestions


def test_run_slate_defaults_to_next_week_and_uses_game_elo(fake_store):
    resp = asyncio.run(run_slate(None))
    assert resp.season == 2026 and resp.through_week == 2 and resp.week == 3
    assert [g.home_team for g in resp.games] == ["Georgia", "Alabama"]
    g1 = resp.games[0]
    expected = predict(_model_row(), elo_diff=302.0, adj_diff=12.0, home_field=1.0)   # 1802-1500, 10-(-2)
    assert g1.home_win_prob == pytest.approx(expected.home_win_prob)
    assert resp.games[1].neutral_site is True


def test_list_teams_is_current_season_sorted(fake_store):
    teams = asyncio.run(list_teams())
    assert [t.team for t in teams] == ["Alabama", "Georgia", "Georgia Tech"]
    assert teams[2].conference == "ACC"


def test_get_store_reloads_after_ttl(monkeypatch):
    calls = []
    def fake_load():
        calls.append(1)
        return _store()
    monkeypatch.setattr(m, "_load_store", fake_load)
    monkeypatch.setattr(m, "_store", None)
    asyncio.run(m._get_store())
    asyncio.run(m._get_store())
    assert len(calls) == 1
    m._store.loaded_at -= m.CACHE_TTL_SECONDS + 1
    asyncio.run(m._get_store())
    assert len(calls) == 2


def test_load_store_reads_three_tables_and_latest_model_row(monkeypatch):
    monkeypatch.setattr(settings_module(), "minio_bucket", "b")
    tables = {
        "cfb/team_profile": pa.table({"season": [2026], "through_week": [1], "team": ["Georgia"]}),
        "cfb/games": pa.table({"season": [2026]}),
        "cfb/model": pa.table({"fitted_at": [1, 2], "win_intercept": [0.0, 9.0]}),
    }
    def fake_dt(path, storage_options):
        key = next(k for k in tables if k in path)
        dt = MagicMock(); dt.to_pyarrow_table.return_value = tables[key]; return dt
    with patch("workflows.cfb.matchup.DeltaTable", side_effect=fake_dt):
        store = m._load_store()
    assert store.model_row["win_intercept"] == 9.0
    assert store.profile["team"].to_list() == ["Georgia"]


def settings_module():
    from config import settings
    return settings
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/cfb/test_matchup.py -v -k "store or run_ or list_teams or model_record"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the store and runners**

Append to `workflows/cfb/matchup.py`:

```python
@dataclass
class Store:
    profile: pl.DataFrame
    games: pl.DataFrame
    model_row: dict
    loaded_at: float


_store: Optional[Store] = None
_store_lock = asyncio.Lock()

_PATHS = {"profile": "cfb/team_profile", "games": "cfb/games", "model": "cfb/model"}


def _storage_options() -> dict:
    return {
        "endpoint_url": settings.minio_endpoint,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }


def _read(key: str) -> pl.DataFrame:
    path = f"s3://{settings.minio_bucket}/{_PATHS[key]}/"
    dt = DeltaTable(path, storage_options=_storage_options())
    return pl.from_arrow(dt.to_pyarrow_table())


def _load_store() -> Store:
    profile = _read("profile")
    games = _read("games")
    model = _read("model").sort("fitted_at", descending=True)
    if model.is_empty():
        raise CfbMatchupError("cfb_model is empty — run pipeline_cfb_model.py")
    logger.info(f"cfb matchup: loaded profile={len(profile)} games={len(games)} model rows={len(model)}")
    return Store(profile=profile, games=games, model_row=model.row(0, named=True), loaded_at=time.monotonic())


async def _get_store() -> Store:
    global _store
    if _store is not None and time.monotonic() - _store.loaded_at < CACHE_TTL_SECONDS:
        return _store
    async with _store_lock:
        if _store is None or time.monotonic() - _store.loaded_at >= CACHE_TTL_SECONDS:
            _store = await asyncio.to_thread(_load_store)
    return _store


def model_record(model_row: dict) -> CfbModelRecord:
    bins_raw = model_row.get("calibration_bins") or "[]"
    bins = json.loads(bins_raw) if isinstance(bins_raw, str) else bins_raw
    return CfbModelRecord(
        accuracy=model_row["accuracy"], brier=model_row["brier"], log_loss=model_row["log_loss"],
        elo_baseline_accuracy=model_row["elo_accuracy"], elo_baseline_brier=model_row["elo_brier"],
        calibration_bins=[CfbCalibrationBin(**b) for b in bins],
        trained_through_season=int(model_row["trained_through_season"]),
        shrink_k=int(model_row["shrink_k"]),
    )


def _current_season(profile: pl.DataFrame) -> int:
    return int(profile["season"].max())


def _profile_row(profile: pl.DataFrame, season: int, through_week: int, team: str) -> dict:
    hit = profile.filter((pl.col("season") == season) & (pl.col("through_week") == through_week) & (pl.col("team") == team))
    if hit.is_empty():
        names = profile.filter(pl.col("season") == season)["team"].unique().to_list()
        raise CfbMatchupError(f"Unknown team {team!r} for {season}", suggestions=suggest_teams(team, names))
    return hit.row(0, named=True)


def _team_record(row: dict) -> CfbTeamRecord:
    return CfbTeamRecord(team=row["team"], conference=row.get("conference"),
                         wins=int(row.get("wins") or 0), losses=int(row.get("losses") or 0),
                         games_played=int(row.get("games") or 0))


def _data_through_date(games: pl.DataFrame, season: int, through_week: int) -> Optional[str]:
    played = games.filter(
        (pl.col("season") == season) & (pl.col("season_type") == "regular")
        & (pl.col("week") <= through_week) & pl.col("completed") & pl.col("start_date").is_not_null()
    )
    if played.is_empty():
        return None
    return str(played["start_date"].max())[:10]


def _head_to_head(games: pl.DataFrame, a: str, b: str, limit: int = 10) -> list[CfbH2HGame]:
    pair = games.filter(
        pl.col("completed") & pl.col("home_points").is_not_null()
        & (((pl.col("home_team") == a) & (pl.col("away_team") == b)) | ((pl.col("home_team") == b) & (pl.col("away_team") == a)))
    ).sort("start_date", descending=True, nulls_last=True).head(limit)
    return [
        CfbH2HGame(season=int(r["season"]), week=int(r["week"]), season_type=r["season_type"],
                   date=str(r["start_date"])[:10] if r["start_date"] is not None else None,
                   home_team=r["home_team"], away_team=r["away_team"],
                   home_points=int(r["home_points"]), away_points=int(r["away_points"]))
        for r in pair.to_dicts()
    ]


async def run_matchup(req: CfbMatchupRequest) -> CfbMatchupResponse:
    store = await _get_store()
    season = req.season or _current_season(store.profile)
    through_week = req.through_week if req.through_week is not None else latest_through_week(store.profile, season)
    home = _profile_row(store.profile, season, through_week, req.home_team)
    away = _profile_row(store.profile, season, through_week, req.away_team)
    home_field = 0.0 if req.neutral_site else 1.0
    prediction = predict(store.model_row, float(home["elo"]) - float(away["elo"]),
                         float(home["adj_margin"]) - float(away["adj_margin"]), home_field)
    return CfbMatchupResponse(
        season=season, through_week=through_week,
        data_through_date=_data_through_date(store.games, season, through_week),
        neutral_site=req.neutral_site,
        home=_team_record(home), away=_team_record(away),
        prediction=prediction,
        model=model_record(store.model_row),
        comparison=build_radar(home, away, f"{home['team']} vs {away['team']} — {season} through week {through_week}"),
        edges=build_edges(home, away),
        head_to_head=_head_to_head(store.games, home["team"], away["team"]),
    )


async def run_slate(week: Optional[int]) -> CfbSlateResponse:
    store = await _get_store()
    season = _current_season(store.profile)
    through_week = latest_through_week(store.profile, season)
    target_week = week if week is not None else through_week + 1
    games = store.games.filter(
        (pl.col("season") == season) & (pl.col("season_type") == "regular") & (pl.col("week") == target_week)
        & (pl.col("home_classification") == "fbs") & (pl.col("away_classification") == "fbs")
    ).sort("start_date", nulls_last=True)
    out = []
    for g in games.to_dicts():
        try:
            home = _profile_row(store.profile, season, through_week, g["home_team"])
            away = _profile_row(store.profile, season, through_week, g["away_team"])
        except CfbMatchupError:
            continue   # a team with no profile row (rare) is skipped rather than failing the slate
        elo_diff = float(g["home_pregame_elo"] or home["elo"]) - float(g["away_pregame_elo"] or away["elo"])
        p = predict(store.model_row, elo_diff, float(home["adj_margin"]) - float(away["adj_margin"]),
                    0.0 if g["neutral_site"] else 1.0)
        out.append(CfbSlateGame(
            game_id=int(g["id"]), week=int(g["week"]),
            start_date=str(g["start_date"])[:10] if g["start_date"] is not None else None,
            home_team=g["home_team"], away_team=g["away_team"], neutral_site=bool(g["neutral_site"]),
            home_win_prob=p.home_win_prob,
            projected_home_points=p.projected_home_points, projected_away_points=p.projected_away_points,
        ))
    return CfbSlateResponse(season=season, week=target_week, through_week=through_week,
                            data_through_date=_data_through_date(store.games, season, through_week), games=out)


async def list_teams() -> list[CfbTeam]:
    store = await _get_store()
    season = _current_season(store.profile)
    through_week = latest_through_week(store.profile, season)
    rows = (store.profile.filter((pl.col("season") == season) & (pl.col("through_week") == through_week))
            .select("team", "conference").unique().sort("team"))
    return [CfbTeam(team=r["team"], conference=r["conference"]) for r in rows.to_dicts()]
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/cfb/test_matchup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/cfb/matchup.py tests/cfb/test_matchup.py
git commit -m "feat: TTL-cached store and matchup/slate/teams runners

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Routes

**Files:**
- Modify: `routers/workflows.py`
- Create: `tests/cfb/test_matchup_router.py`

**Interfaces:**
- Produces: `POST /workflows/cfb/matchup` → `CfbMatchupResponse` | 422 `{error, suggestions}`; `GET /workflows/cfb/slate?week=` → `CfbSlateResponse`; `GET /workflows/cfb/teams` → `list[CfbTeam]`

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_matchup_router.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app
from schemas.models import (
    CfbMatchupResponse, CfbTeamRecord, CfbPrediction, CfbModelRecord, CfbSlateResponse, CfbTeam,
)
from workflows.cfb.matchup import CfbMatchupError


@pytest.fixture
def client():
    return TestClient(app)


def _matchup():
    rec = lambda t: CfbTeamRecord(team=t, conference="SEC", wins=2, losses=0, games_played=2)
    return CfbMatchupResponse(
        season=2026, through_week=2, data_through_date="2026-09-05", neutral_site=False,
        home=rec("Georgia"), away=rec("Alabama"),
        prediction=CfbPrediction(home_win_prob=0.7, away_win_prob=0.3, projected_home_points=31, projected_away_points=21, expected_margin=10),
        model=CfbModelRecord(accuracy=0.74, brier=0.18, log_loss=0.53, elo_baseline_accuracy=0.708, elo_baseline_brier=0.2,
                             calibration_bins=[], trained_through_season=2025, shrink_k=4),
        comparison={"title": "t", "axes": ["a"], "series": []}, edges=[], head_to_head=[],
    )


def test_matchup_returns_200(client):
    with patch("routers.workflows.run_cfb_matchup", new=AsyncMock(return_value=_matchup())):
        r = client.post("/workflows/cfb/matchup", json={"home_team": "Georgia", "away_team": "Alabama"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["home_win_prob"] == 0.7
    assert body["home"]["team"] == "Georgia"
    assert body["model"]["elo_baseline_accuracy"] == 0.708


def test_matchup_unknown_team_is_422_with_suggestions(client):
    err = CfbMatchupError("Unknown team 'Georgi'", suggestions=["Georgia", "Georgia Tech"])
    with patch("routers.workflows.run_cfb_matchup", new=AsyncMock(side_effect=err)):
        r = client.post("/workflows/cfb/matchup", json={"home_team": "Georgi", "away_team": "Alabama"})
    assert r.status_code == 422
    assert r.json() == {"error": "Unknown team 'Georgi'", "suggestions": ["Georgia", "Georgia Tech"]}


def test_slate_is_get_and_not_swallowed_by_job_id_route(client):
    resp = CfbSlateResponse(season=2026, week=3, through_week=2, data_through_date="2026-09-05", games=[])
    with patch("routers.workflows.run_cfb_slate", new=AsyncMock(return_value=resp)) as mock_run:
        r = client.get("/workflows/cfb/slate?week=3")
    assert r.status_code == 200
    assert r.json()["week"] == 3
    mock_run.assert_awaited_once_with(3)


def test_slate_week_optional(client):
    resp = CfbSlateResponse(season=2026, week=3, through_week=2, data_through_date=None, games=[])
    with patch("routers.workflows.run_cfb_slate", new=AsyncMock(return_value=resp)) as mock_run:
        r = client.get("/workflows/cfb/slate")
    assert r.status_code == 200
    mock_run.assert_awaited_once_with(None)


def test_teams_is_get(client):
    with patch("routers.workflows.list_cfb_teams", new=AsyncMock(return_value=[CfbTeam(team="Georgia", conference="SEC")])):
        r = client.get("/workflows/cfb/teams")
    assert r.status_code == 200
    assert r.json() == [{"team": "Georgia", "conference": "SEC"}]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/cfb/test_matchup_router.py -v`
Expected: FAIL — `AttributeError: <module 'routers.workflows'> does not have the attribute 'run_cfb_matchup'`

- [ ] **Step 3: Add the routes**

In `routers/workflows.py`, extend the schema import and add the matchup import:

```python
from schemas.models import (
    CfbAnalystRequest,
    CfbMatchupRequest,
    CfbMatchupResponse,
    CfbSlateResponse,
    CfbTeam,
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
from workflows.soccer.analyst import run_soccer_analyst, SoccerAnalystError
from workflows.cfb.analyst import run_cfb_analyst, CfbAnalystError
from workflows.cfb.matchup import (
    run_matchup as run_cfb_matchup,
    run_slate as run_cfb_slate,
    list_teams as list_cfb_teams,
    CfbMatchupError,
)
```

Then add, directly after the `cfb_analyst` route:

```python
# Nested under /cfb/ so the GET routes are not captured by GET /{job_id}
# (a path parameter never matches across a "/").
@router.post("/cfb/matchup", response_model=CfbMatchupResponse)
async def cfb_matchup(request: CfbMatchupRequest):
    try:
        return await run_cfb_matchup(request)
    except CfbMatchupError as e:
        return JSONResponse(status_code=422, content={"error": str(e), "suggestions": e.suggestions})


@router.get("/cfb/slate", response_model=CfbSlateResponse)
async def cfb_slate(week: int | None = None):
    try:
        return await run_cfb_slate(week)
    except CfbMatchupError as e:
        return JSONResponse(status_code=422, content={"error": str(e), "suggestions": e.suggestions})


@router.get("/cfb/teams", response_model=list[CfbTeam])
async def cfb_teams():
    return await list_cfb_teams()
```

- [ ] **Step 4: Run route tests and the whole suite**

Run: `uv run pytest tests/cfb -v`
Expected: PASS

Run: `uv run pytest -q`
Expected: only the pre-existing `test_trigger_briefing_creates_job` failure.

- [ ] **Step 5: Commit**

```bash
git add routers/workflows.py tests/cfb/test_matchup_router.py
git commit -m "feat: add /workflows/cfb/{matchup,slate,teams} routes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Smoke check against MinIO and doc fix

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the server and hit each endpoint (requires MinIO + a populated `cfb_model`)**

```bash
uv run uvicorn main:app --port 8060
```

In another shell:

```bash
curl -s http://localhost:8060/workflows/cfb/teams | head -c 300
```

```bash
curl -s "http://localhost:8060/workflows/cfb/slate" | python -m json.tool | head -40
```

```bash
curl -s -X POST http://localhost:8060/workflows/cfb/matchup -H 'Content-Type: application/json' -d '{"home_team":"Georgia","away_team":"Alabama"}' | python -m json.tool | head -60
```

```bash
curl -s -X POST http://localhost:8060/workflows/cfb/matchup -H 'Content-Type: application/json' -d '{"home_team":"Georgi","away_team":"Alabama"}'
```

Expected: teams list sorted; slate `week == through_week + 1` with `home_win_prob` in (0, 1); matchup with six radar axes, eight edges, and `data_through_date` set; the last call returns 422 with `suggestions` containing `"Georgia"`.

- [ ] **Step 2: Fix CLAUDE.md**

In this repo's `CLAUDE.md`, replace `No test suite is configured yet. \`httpx\` is available as a dev dependency for writing integration tests against the running server.` with:

```
`uv run pytest` runs the suite under `tests/`, organised by package (`tests/cfb/`, `tests/soccer/`, `tests/routers/`). Route tests use `fastapi.testclient.TestClient(app)` and patch `routers.workflows.run_<name>` with `AsyncMock`.
```

And under *Adding a new workflow*, append a fourth point:

```
4. Read-only endpoints must be nested under a prefix (e.g. `/cfb/teams`) — `GET /workflows/{job_id}` captures any single-segment GET path.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: correct test-suite note; document the GET catch-all trap

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage:** models (T1), no-LLM prediction from stored floats (T2), TTL store with `data_through_date` (T3), radar reusing the `RadarFigure` shape and edges as a separate raw-value array (T2/T3), head-to-head from `cfb_games` (T3), nested GET routes + `teams` endpoint + 422 with suggestions (T4), analyst prompt range (T1), server tests where the formula runs (T2–T4). The spec's `teams: [...]` list became named `home`/`away` fields — same data, clearer for the UI.

**Placeholder scan:** none.

**Type consistency:** `predict(model_row, elo_diff, adj_diff, home_field)` identical in T2 tests, T3 runners, and T3 tests. `CfbMatchupError(message, suggestions)` shape matches T4's error handler. `run_matchup`/`run_slate`/`list_teams` are aliased in T4's import exactly as the router tests patch them (`run_cfb_matchup`, `run_cfb_slate`, `list_cfb_teams`). `Store` fields match the fixture in T3.
