"""Persisting the predictions the slate serves.

`GET /workflows/cfb/slate` computes a prediction for every FBS game of a week
and, before this module, threw them away: the only record of what the model
said before kickoff was the HTTP response. These writes make the served
prediction durable so `GET /workflows/cfb/backtest` can score the model
against what it actually showed rather than against a replay.

The write lives here, in the serving layer, rather than in sports-etl: a
prediction is an artifact of *serving* — it exists because a request came in,
and it is made from the model row and profile snapshot this process had cached
at that moment. The ETL has no way to know what the server showed. This is the
one table in the bucket agentic-server writes, and it writes nothing else, so
there is still exactly one writer per table.

Append-only, idempotent on (game_id, model_fitted_at, through_week): the same
slate call made ten times stores one row per game. Nothing here is allowed to
fail a request — every entry point logs and returns.
"""
from datetime import datetime, timezone
from typing import Optional

import polars as pl
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from config import settings
from core.logging import logger
from schemas.models import CfbSlateGame

PREDICTIONS_PATH = "cfb/predictions"

# Written explicitly so the first write fixes the column types, instead of
# letting the types of whichever slate happened to be served first decide.
SCHEMA = pa.schema([
    pa.field("game_id", pa.int64()),
    pa.field("season", pa.int64()),
    pa.field("week", pa.int64()),
    pa.field("through_week", pa.int64()),
    pa.field("model_fitted_at", pa.string()),
    pa.field("home_team", pa.string()),
    pa.field("away_team", pa.string()),
    pa.field("neutral_site", pa.bool_()),
    pa.field("home_win_prob", pa.float64()),
    pa.field("projected_home_points", pa.float64()),
    pa.field("projected_away_points", pa.float64()),
    pa.field("expected_margin", pa.float64()),
    pa.field("generated_at", pa.string()),
])

KEY = ("game_id", "model_fitted_at", "through_week")


def _uri() -> str:
    return f"s3://{settings.minio_bucket}/{PREDICTIONS_PATH}/"


def _storage_options() -> dict:
    # Same shape as workflows.cfb.matchup._storage_options; imported lazily to
    # keep this module independent of the serving layer that calls it.
    from workflows.cfb.matchup import _storage_options as opts
    return opts()


def model_fitted_at(model_row: dict) -> str:
    """The model identity stamped on a row, as a string.

    `fitted_at` is a timestamp in some writes and an ISO string in others, so
    it is normalised here — the idempotency key must compare equal across both.
    """
    return str(model_row.get("fitted_at"))


def rows_for_slate(games: list[CfbSlateGame], *, season: int, through_week: int,
                   model_row: dict) -> list[dict]:
    """One storable row per slate game."""
    fitted_at = model_fitted_at(model_row)
    generated_at = datetime.now(timezone.utc).isoformat()
    return [
        {
            "game_id": int(g.game_id),
            "season": int(season),
            "week": int(g.week),
            "through_week": int(through_week),
            "model_fitted_at": fitted_at,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "neutral_site": bool(g.neutral_site),
            "home_win_prob": float(g.home_win_prob),
            "projected_home_points": float(g.projected_home_points),
            "projected_away_points": float(g.projected_away_points),
            "expected_margin": float(g.projected_home_points - g.projected_away_points),
            "generated_at": generated_at,
        }
        for g in games
    ]


def read() -> Optional[pl.DataFrame]:
    """Every stored prediction, or None when the table does not exist yet.

    None and an empty frame mean different things to the backtest — "nothing
    has ever been stored" versus "stored, nothing matched" — but both make it
    fall back to the replay, so callers may treat them alike.
    """
    try:
        dt = DeltaTable(_uri(), storage_options=_storage_options())
        return pl.from_arrow(dt.to_pyarrow_table())
    except Exception as e:
        logger.info(f"cfb predictions: no stored predictions yet ({e})")
        return None


def _existing_keys(stored: Optional[pl.DataFrame]) -> set[tuple]:
    if stored is None or stored.is_empty():
        return set()
    missing = [c for c in KEY if c not in stored.columns]
    if missing:
        logger.warning(f"cfb predictions: stored table lacks {missing} — skipping dedupe")
        return set()
    return {
        (int(r["game_id"]), str(r["model_fitted_at"]), int(r["through_week"]))
        for r in stored.select(KEY).to_dicts()
    }


def write(rows: list[dict]) -> int:
    """Append the rows not already stored. Returns how many were written.

    Blocking (S3 round trips) — call it off the event loop. Never raises: a
    prediction that cannot be stored must not cost the caller its response.
    """
    if not rows:
        return 0
    try:
        seen = _existing_keys(read())
        fresh, batch_keys = [], set()
        for r in rows:
            k = (int(r["game_id"]), str(r["model_fitted_at"]), int(r["through_week"]))
            if k in seen or k in batch_keys:
                continue
            batch_keys.add(k)
            fresh.append(r)
        if not fresh:
            logger.info(f"cfb predictions: all {len(rows)} already stored")
            return 0
        write_deltalake(_uri(), pa.Table.from_pylist(fresh, schema=SCHEMA),
                        mode="append", storage_options=_storage_options())
        logger.info(f"cfb predictions: stored {len(fresh)} of {len(rows)}")
        return len(fresh)
    except Exception as e:
        logger.warning(f"cfb predictions: could not store {len(rows)} predictions ({e})")
        return 0
