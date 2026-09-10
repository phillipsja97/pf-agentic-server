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
    CfbCalibrationBin, CfbEdge, CfbH2HGame, CfbMatchupRequest, CfbMatchupResponse,
    CfbModelRecord, CfbPrediction, CfbSlateGame, CfbSlateResponse, CfbTeam, CfbTeamRecord,
)

CACHE_TTL_SECONDS = 1800

# (axis label, profile column) — percentile columns, allowed-stats already
# inverted upstream so a bigger shape is always a better team.
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
        "series": [
            {"name": home["team"], "values": values(home)},
            {"name": away["team"], "values": values(away)},
        ],
    }


def build_edges(home: dict, away: dict) -> list[CfbEdge]:
    edges = []
    for label, col, unit, lower_is_better in EDGE_STATS:
        h, a = home.get(col), away.get(col)
        if h is None or a is None:
            edges.append(CfbEdge(stat=label, home_value=h, away_value=a, diff=None,
                                 favors="even", unit=unit))
            continue
        diff = float(h) - float(a)
        if abs(diff) < 1e-9:
            favors = "even"
        elif (diff > 0) != lower_is_better:
            favors = "home"
        else:
            favors = "away"
        edges.append(CfbEdge(stat=label, home_value=float(h), away_value=float(a),
                             diff=diff, favors=favors, unit=unit))
    return edges


def suggest_teams(name: str, names: list[str], limit: int = 5) -> list[str]:
    """Near matches for a team name that did not resolve.

    Prefix hits first (in name order), then fuzzy matches — "Georgi" should
    surface Georgia, Georgia Southern and Georgia Tech rather than nothing.
    """
    lowered = {n.lower(): n for n in names}
    prefix = [lowered[k] for k in sorted(lowered) if k.startswith(name.lower())]
    fuzzy = [lowered[k] for k in difflib.get_close_matches(name.lower(), list(lowered),
                                                           n=limit, cutoff=0.6)]
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
