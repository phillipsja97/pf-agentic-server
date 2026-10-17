"""Scoring the CFB model against what it said before kickoff.

Two sources of truth, in order of preference:

1. **Stored** — a row in `cfb_predictions`, written by the slate when it served
   the game (see workflows.cfb.predictions). This is a record of what the UI
   actually showed, so it is used whenever one exists.
2. **Replayed** — rebuilt from the profile snapshot at `through_week = W-1`
   plus the game row's pregame Elo, run through the current model row. Every
   game played before predictions were ever stored falls here, and each row
   says which it is.

A stored row only counts as pregame if its `through_week` is below the game's
week: a slate pulled up after kickoff stamps the latest week and knows the
result, so including it would flatter the model. Among the pregame rows for a
game the highest `through_week` wins — the most informed look before kickoff.
"""
from statistics import median
from typing import Optional

import polars as pl

from core.logging import logger
from schemas.models import (
    CfbBacktestGame, CfbBacktestMetrics, CfbBacktestResponse, CfbBacktestWeek,
)
from workflows.cfb import predictions
from workflows.cfb.matchup import (
    CfbMatchupError, _current_season, _data_through_date, _get_store, _profile_row,
    latest_through_week, predict,
)

WITHIN_THRESHOLDS = (3, 7, 10, 14)


def _finished_games(games: pl.DataFrame, season: int, week: Optional[int]) -> pl.DataFrame:
    """Completed regular-season FBS-vs-FBS games — the same population the slate covers."""
    rows = games.filter(
        (pl.col("season") == season) & (pl.col("season_type") == "regular")
        & (pl.col("home_classification") == "fbs") & (pl.col("away_classification") == "fbs")
        & pl.col("home_points").is_not_null() & pl.col("away_points").is_not_null()
    )
    if week is not None:
        rows = rows.filter(pl.col("week") == week)
    return rows.sort("week", "start_date", nulls_last=True)


def pregame_predictions(stored: Optional[pl.DataFrame], season: int) -> dict[int, dict]:
    """The best pregame stored prediction per game id, keyed by game id."""
    if stored is None or stored.is_empty():
        return {}
    required = {"game_id", "season", "week", "through_week"}
    if not required.issubset(stored.columns):
        logger.warning("cfb backtest: stored predictions missing columns — ignoring them")
        return {}
    rows = stored.filter(
        (pl.col("season") == season) & (pl.col("through_week") < pl.col("week"))
    ).sort("through_week", descending=True)
    best: dict[int, dict] = {}
    for r in rows.to_dicts():
        # Normalised to the shape _replay returns, so score() reads one shape.
        best.setdefault(int(r["game_id"]), {**r, "fitted_at": r.get("model_fitted_at")})
    return best


def _replay(store, season: int, game: dict, model_row: dict) -> Optional[dict]:
    """What the slate would have shown for this game, or None if it could not have."""
    through_week = int(game["week"]) - 1
    try:
        home = _profile_row(store.profile, season, through_week, game["home_team"])
        away = _profile_row(store.profile, season, through_week, game["away_team"])
    except CfbMatchupError:
        return None
    elo_diff = (float(game["home_pregame_elo"] or home["elo"])
                - float(game["away_pregame_elo"] or away["elo"]))
    p = predict(
        model_row, elo_diff,
        float(home["adj_margin"]) - float(away["adj_margin"]),
        0.0 if game["neutral_site"] else 1.0,
        eff_diff=float(home.get("net_eff") or 0.0) - float(away.get("net_eff") or 0.0),
        off_sum=float(home.get("adj_off") or 0.0) + float(away.get("adj_off") or 0.0),
        def_sum=float(home.get("adj_def") or 0.0) + float(away.get("adj_def") or 0.0),
    )
    return {
        "through_week": through_week,
        "fitted_at": predictions.model_fitted_at(model_row),
        "home_win_prob": p.home_win_prob,
        "projected_home_points": p.projected_home_points,
        "projected_away_points": p.projected_away_points,
        "expected_margin": p.expected_margin,
    }


def score(game: dict, pred: dict, source: str) -> CfbBacktestGame:
    """Join one prediction to one final score."""
    home_points, away_points = int(game["home_points"]), int(game["away_points"])
    actual_margin = home_points - away_points
    projected_margin = float(pred["expected_margin"])
    projected_total = float(pred["projected_home_points"]) + float(pred["projected_away_points"])
    prob = float(pred["home_win_prob"])
    # Ties and a coin-flip probability both count as a miss, deliberately:
    # the pick has to land on the right side to be right.
    correct = (prob > 0.5 and actual_margin > 0) or (prob < 0.5 and actual_margin < 0)
    return CfbBacktestGame(
        game_id=int(game["id"]), week=int(game["week"]),
        date=str(game["start_date"])[:10] if game["start_date"] is not None else None,
        home_team=game["home_team"], away_team=game["away_team"],
        neutral_site=bool(game["neutral_site"]),
        source=source, through_week=int(pred["through_week"]),
        fitted_at=pred.get("fitted_at"),
        home_win_prob=prob,
        projected_home_points=float(pred["projected_home_points"]),
        projected_away_points=float(pred["projected_away_points"]),
        expected_margin=projected_margin,
        actual_home_points=home_points, actual_away_points=away_points,
        actual_margin=actual_margin, actual_total=home_points + away_points,
        margin_error=projected_margin - actual_margin,
        total_error=projected_total - (home_points + away_points),
        straight_up_correct=correct,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def metrics(games: list[CfbBacktestGame]) -> dict:
    """The metric set, as a dict of the CfbBacktestMetrics fields.

    An empty set reports only `games: 0` — a zero accuracy over no games would
    read as a model that never gets anything right.
    """
    if not games:
        return {"games": 0}
    margin_err = [g.margin_error for g in games]
    margin_abs = [abs(e) for e in margin_err]
    total_err = [g.total_error for g in games]
    home_err = [g.projected_home_points - g.actual_home_points for g in games]
    away_err = [g.projected_away_points - g.actual_away_points for g in games]
    brier = _mean([(g.home_win_prob - (1.0 if g.actual_margin > 0 else 0.0)) ** 2 for g in games])
    out = {
        "games": len(games),
        "straight_up_accuracy": round(_mean([1.0 if g.straight_up_correct else 0.0
                                             for g in games]), 4),
        "brier": round(brier, 4),
        "margin_mae": round(_mean(margin_abs), 2),
        "margin_median_abs_error": round(median(margin_abs), 2),
        "margin_bias": round(_mean(margin_err), 2),
        "total_mae": round(_mean([abs(e) for e in total_err]), 2),
        "total_bias": round(_mean(total_err), 2),
        "home_score_mae": round(_mean([abs(e) for e in home_err]), 2),
        "home_score_bias": round(_mean(home_err), 2),
        "away_score_mae": round(_mean([abs(e) for e in away_err]), 2),
        "away_score_bias": round(_mean(away_err), 2),
    }
    for t in WITHIN_THRESHOLDS:
        out[f"within_{t}"] = round(_mean([1.0 if e <= t else 0.0 for e in margin_abs]), 4)
    return out


def by_week(games: list[CfbBacktestGame]) -> list[CfbBacktestWeek]:
    weeks: dict[int, list[CfbBacktestGame]] = {}
    for g in games:
        weeks.setdefault(g.week, []).append(g)
    return [CfbBacktestWeek(week=w, **metrics(weeks[w])) for w in sorted(weeks)]


async def run_backtest(season: Optional[int] = None,
                       week: Optional[int] = None) -> CfbBacktestResponse:
    store = await _get_store()
    season = season or _current_season(store.profile)
    model_row = store.model_row
    stored = pregame_predictions(predictions.read(), season)

    scored: list[CfbBacktestGame] = []
    skipped = 0
    for game in _finished_games(store.games, season, week).to_dicts():
        hit = stored.get(int(game["id"]))
        if hit is not None:
            scored.append(score(game, hit, "stored"))
            continue
        replayed = _replay(store, season, game, model_row)
        if replayed is None:
            # No profile snapshot as of W-1 — week 1 of a season whose
            # through_week 0 was never written, mostly.
            skipped += 1
            continue
        scored.append(score(game, replayed, "replayed"))

    stored_count = sum(1 for g in scored if g.source == "stored")
    logger.info(f"cfb backtest: season={season} week={week} scored={len(scored)} "
                f"stored={stored_count} skipped={skipped}")
    return CfbBacktestResponse(
        season=season, week=week,
        fitted_at=predictions.model_fitted_at(model_row),
        trained_through_season=int(model_row["trained_through_season"]),
        data_through_date=_data_through_date(store.games, season,
                                             latest_through_week(store.profile, season)),
        stored_games=stored_count,
        replayed_games=len(scored) - stored_count,
        skipped_games=skipped,
        overall=CfbBacktestMetrics(**metrics(scored)),
        by_week=by_week(scored),
        games=scored,
    )
