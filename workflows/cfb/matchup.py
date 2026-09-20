"""CFB matchup predictor serving layer.

Reads cfb_team_profile / cfb_model / cfb_games / cfb_teams from MinIO (TTL-cached) and
applies the stored linear coefficients. No LLM, no SQL generation.
"""
import asyncio
import difflib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import polars as pl
from deltalake import DeltaTable

from config import settings
from core.logging import logger
from workflows.cfb.colors import resolve_pair
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


# Feature lists must match sports-etl model/cfb_model.py HEAD_FEATURES; the
# stored coefficient columns are named from them. The total head takes sums:
# a total cannot be predicted from differences alone.
FEATURES = ("elo_diff", "adj_diff", "eff_diff", "home_field")
TOTAL_FEATURES = ("home_field", "off_sum", "def_sum")
HEAD_FEATURES = {"win": FEATURES, "margin": FEATURES, "total": TOTAL_FEATURES}


def _linear(model_row: dict, head: str, features: dict[str, float]) -> float:
    total = model_row[f"{head}_intercept"]
    for name in HEAD_FEATURES[head]:
        # A model row fitted before a feature existed simply lacks its column;
        # treating that as a zero coefficient keeps old rows usable.
        total += model_row.get(f"{head}_b_{name}", 0.0) * features.get(name, 0.0)
    return total


def predict(model_row: dict, elo_diff: float, adj_diff: float, home_field: float,
            eff_diff: float = 0.0, off_sum: float = 0.0, def_sum: float = 0.0) -> CfbPrediction:
    feats = {"elo_diff": elo_diff, "adj_diff": adj_diff,
             "eff_diff": eff_diff, "home_field": home_field,
             "off_sum": off_sum, "def_sum": def_sum}
    p_home = 1.0 / (1.0 + math.exp(-_linear(model_row, "win", feats)))
    margin = _linear(model_row, "margin", feats)
    total = _linear(model_row, "total", feats)
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


@dataclass
class Store:
    profile: pl.DataFrame
    games: pl.DataFrame
    model_row: dict
    loaded_at: float
    # school -> (primary, alternate). Empty if cfb_teams has not been built
    # yet, in which case every side renders in the neutral.
    colors: dict[str, tuple[Optional[str], Optional[str]]] = field(default_factory=dict)
    # CORE efficiency by (season, through_week, team). Empty if not built.
    core: Optional[pl.DataFrame] = None
    # Weekly SP+/FPI snapshots by (season, through_week, team). Empty if not built.
    ratings: Optional[pl.DataFrame] = None


_store: Optional[Store] = None
_store_lock = asyncio.Lock()

_PATHS = {"profile": "cfb/team_profile", "games": "cfb/games", "model": "cfb/model",
          "teams": "cfb/teams", "core": "cfb/core_ratings",
          "ratings": "cfb/ratings_snapshots"}


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


def _load_colors() -> dict[str, tuple[Optional[str], Optional[str]]]:
    """Latest season's brand colours, keyed by school.

    Optional: the matchup still works without cfb_teams, just in neutral grey,
    so a missing table must not take the endpoint down.
    """
    try:
        teams = _read("teams")
    except Exception as e:
        logger.warning(f"cfb matchup: no team colours available ({e})")
        return {}
    if teams.is_empty():
        return {}
    latest = teams.filter(pl.col("year") == teams["year"].max())
    return {
        r["school"]: (r.get("color"), r.get("alternate_color"))
        for r in latest.to_dicts()
    }


def _load_ratings() -> Optional[pl.DataFrame]:
    """As-of SP+/FPI snapshots. Optional, like CORE."""
    try:
        return _read("ratings")
    except Exception as e:
        logger.warning(f"cfb matchup: no SP+/FPI snapshots available ({e})")
        return None


RATING_FIELDS = ("sp_rating", "sp_rank", "fpi", "fpi_rank")


def ratings_for(store: Store, season: int, through_week: int, team: str) -> dict:
    """The latest SP+/FPI snapshot for a team at or before `through_week`.

    Snapshots are taken weekly, so the most recent one at or before the
    prediction's own as-of week is what the ratings said before kickoff.
    """
    empty = {f: None for f in RATING_FIELDS}
    r = store.ratings
    if r is None or r.is_empty():
        return empty
    rows = r.filter((pl.col("season") == season) & (pl.col("through_week") <= through_week)
                    & (pl.col("team") == team)).sort("through_week", descending=True)
    if rows.is_empty():
        return empty
    row = rows.row(0, named=True)
    return {f: row.get(f) for f in RATING_FIELDS}


def _load_core() -> Optional[pl.DataFrame]:
    """As-of CORE ratings. Optional — the matchup works fine without them."""
    try:
        return _read("core")
    except Exception as e:
        logger.warning(f"cfb matchup: no CORE ratings available ({e})")
        return None


CORE_FIELDS = ("core_off", "core_def", "core_overall",
               "core_off_pct", "core_def_pct", "core_overall_pct")


def core_for(store: Store, season: int, through_week: int, team: str) -> dict:
    """One team's CORE ratings plus its percentile among FBS at the same week.

    Percentiles are stated outward-is-better for all three, so the UI never has
    to remember that `core_def` counts points allowed and therefore ranks in
    reverse.
    """
    empty = {f: None for f in CORE_FIELDS}
    core = store.core
    if core is None or core.is_empty():
        return empty
    week_rows = core.filter((pl.col("season") == season) & (pl.col("through_week") == through_week))
    if week_rows.is_empty():
        return empty
    mine = week_rows.filter(pl.col("team") == team)
    if mine.is_empty():
        return empty
    row = mine.row(0, named=True)

    def pct(column: str, higher_is_better: bool) -> Optional[float]:
        """Rank percentile: best in the field is 100, worst is 0.

        Ranking rather than "fraction strictly below" so the leader reaches the
        rim of a 0-100 radar instead of stopping at (n-1)/n. A field of one has
        no ranking to express, so it sits at the midpoint.
        """
        value = row.get(column)
        if value is None:
            return None
        series = week_rows[column].drop_nulls()
        if series.is_empty():
            return None
        if series.len() == 1:
            return 50.0
        beaten = (series < value).sum() if higher_is_better else (series > value).sum()
        return round(100.0 * beaten / (series.len() - 1), 1)

    return {
        "core_off": row["core_off"], "core_def": row["core_def"],
        "core_overall": row["core_overall"],
        "core_off_pct": pct("core_off", True),
        "core_def_pct": pct("core_def", False),     # points allowed: lower is better
        "core_overall_pct": pct("core_overall", True),
    }


def _load_store() -> Store:
    profile = _read("profile")
    games = _read("games")
    model = _read("model").sort("fitted_at", descending=True)
    if model.is_empty():
        raise CfbMatchupError("cfb_model is empty — run pipeline_cfb_model.py")
    colors = _load_colors()
    core = _load_core()
    ratings = _load_ratings()
    logger.info(f"cfb matchup: loaded profile={len(profile)} games={len(games)} "
                f"model rows={len(model)} colours={len(colors)} "
                f"core={0 if core is None else len(core)} ratings={0 if ratings is None else len(ratings)}")
    return Store(profile=profile, games=games, model_row=model.row(0, named=True),
                 loaded_at=time.monotonic(), colors=colors, core=core, ratings=ratings)


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
        # Benchmarks are optional columns: a row fitted before they existed
        # simply lacks them, and a null leaves that comparison off the page.
        **{k: model_row.get(k) for k in (
            "early_accuracy", "early_elo_accuracy", "market_accuracy",
            "sp_prior_accuracy", "sp_final_accuracy", "fpi_prior_accuracy", "fpi_final_accuracy")},
    )


def _current_season(profile: pl.DataFrame) -> int:
    return int(profile["season"].max())


def _profile_row(profile: pl.DataFrame, season: int, through_week: int, team: str) -> dict:
    hit = profile.filter(
        (pl.col("season") == season) & (pl.col("through_week") == through_week) & (pl.col("team") == team)
    )
    if hit.is_empty():
        names = profile.filter(pl.col("season") == season)["team"].unique().to_list()
        raise CfbMatchupError(f"Unknown team {team!r} for {season}",
                              suggestions=suggest_teams(team, names))
    return hit.row(0, named=True)


def _team_record(row: dict, color: str, core: Optional[dict] = None,
                 ratings: Optional[dict] = None) -> CfbTeamRecord:
    return CfbTeamRecord(team=row["team"], conference=row.get("conference"),
                         wins=int(row.get("wins") or 0), losses=int(row.get("losses") or 0),
                         games_played=int(row.get("games") or 0), color=color,
                         **(core or {}), **(ratings or {}))


def _pair_colors(store: Store, home_team: str, away_team: str) -> tuple[str, str]:
    return resolve_pair(store.colors.get(home_team, (None, None)),
                        store.colors.get(away_team, (None, None)))


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
        & (((pl.col("home_team") == a) & (pl.col("away_team") == b))
           | ((pl.col("home_team") == b) & (pl.col("away_team") == a)))
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
    through_week = (req.through_week if req.through_week is not None
                    else latest_through_week(store.profile, season))
    home = _profile_row(store.profile, season, through_week, req.home_team)
    away = _profile_row(store.profile, season, through_week, req.away_team)
    home_field = 0.0 if req.neutral_site else 1.0
    home_color, away_color = _pair_colors(store, home["team"], away["team"])
    prediction = predict(
        store.model_row,
        float(home["elo"]) - float(away["elo"]),
        float(home["adj_margin"]) - float(away["adj_margin"]),
        home_field,
        eff_diff=float(home.get("net_eff") or 0.0) - float(away.get("net_eff") or 0.0),
        off_sum=float(home.get("adj_off") or 0.0) + float(away.get("adj_off") or 0.0),
        def_sum=float(home.get("adj_def") or 0.0) + float(away.get("adj_def") or 0.0),
    )
    return CfbMatchupResponse(
        season=season, through_week=through_week,
        data_through_date=_data_through_date(store.games, season, through_week),
        neutral_site=req.neutral_site,
        home=_team_record(home, home_color, core_for(store, season, through_week, home["team"]),
                          ratings_for(store, season, through_week, home["team"])),
        away=_team_record(away, away_color, core_for(store, season, through_week, away["team"]),
                          ratings_for(store, season, through_week, away["team"])),
        prediction=prediction,
        model=model_record(store.model_row),
        comparison=build_radar(
            home, away, f"{home['team']} vs {away['team']} — {season} through week {through_week}"
        ),
        edges=build_edges(home, away),
        head_to_head=_head_to_head(store.games, home["team"], away["team"]),
    )


async def run_slate(week: Optional[int]) -> CfbSlateResponse:
    store = await _get_store()
    season = _current_season(store.profile)
    through_week = latest_through_week(store.profile, season)
    target_week = week if week is not None else through_week + 1
    games = store.games.filter(
        (pl.col("season") == season) & (pl.col("season_type") == "regular")
        & (pl.col("week") == target_week)
        & (pl.col("home_classification") == "fbs") & (pl.col("away_classification") == "fbs")
    ).sort("start_date", nulls_last=True)
    out = []
    for g in games.to_dicts():
        try:
            home = _profile_row(store.profile, season, through_week, g["home_team"])
            away = _profile_row(store.profile, season, through_week, g["away_team"])
        except CfbMatchupError:
            # A team with no profile row (rare) is skipped rather than failing the slate.
            continue
        elo_diff = float(g["home_pregame_elo"] or home["elo"]) - float(g["away_pregame_elo"] or away["elo"])
        p = predict(store.model_row, elo_diff,
                    float(home["adj_margin"]) - float(away["adj_margin"]),
                    0.0 if g["neutral_site"] else 1.0,
                    eff_diff=float(home.get("net_eff") or 0.0) - float(away.get("net_eff") or 0.0),
                    off_sum=float(home.get("adj_off") or 0.0) + float(away.get("adj_off") or 0.0),
                    def_sum=float(home.get("adj_def") or 0.0) + float(away.get("adj_def") or 0.0))
        slate_home_color, slate_away_color = _pair_colors(store, g["home_team"], g["away_team"])
        out.append(CfbSlateGame(
            game_id=int(g["id"]), week=int(g["week"]),
            start_date=str(g["start_date"])[:10] if g["start_date"] is not None else None,
            home_team=g["home_team"], away_team=g["away_team"], neutral_site=bool(g["neutral_site"]),
            home_win_prob=p.home_win_prob,
            projected_home_points=p.projected_home_points,
            projected_away_points=p.projected_away_points,
            home_color=slate_home_color, away_color=slate_away_color,
        ))
    return CfbSlateResponse(
        season=season, week=target_week, through_week=through_week,
        data_through_date=_data_through_date(store.games, season, through_week),
        games=out,
    )


async def list_teams() -> list[CfbTeam]:
    store = await _get_store()
    season = _current_season(store.profile)
    through_week = latest_through_week(store.profile, season)
    rows = (
        store.profile
        .filter((pl.col("season") == season) & (pl.col("through_week") == through_week))
        .select("team", "conference").unique().sort("team")
    )
    return [CfbTeam(team=r["team"], conference=r["conference"]) for r in rows.to_dicts()]
