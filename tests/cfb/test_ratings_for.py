import polars as pl
import pytest
from workflows.cfb.matchup import Store, ratings_for


def _snap(rows):
    return pl.DataFrame(rows, schema={"season": pl.Int32, "through_week": pl.Int32, "team": pl.Utf8,
                                      "sp_rating": pl.Float64, "sp_rank": pl.Float64,
                                      "fpi": pl.Float64, "fpi_rank": pl.Float64})


def _row(team, week, sp, fpi):
    return {"season": 2026, "through_week": week, "team": team, "sp_rating": sp, "sp_rank": 1.0, "fpi": fpi, "fpi_rank": 1.0}


def _store(snap):
    return Store(profile=None, games=None, model_row={}, loaded_at=0.0, ratings=snap)


def test_returns_the_latest_snapshot_at_or_before_the_week():
    snap = _snap([_row("Tennessee", 2, 18.0, 16.0), _row("Tennessee", 3, 20.2, 18.5), _row("Tennessee", 4, 22.0, 20.0)])
    got = ratings_for(_store(snap), 2026, 3, "Tennessee")
    assert got["sp_rating"] == pytest.approx(20.2) and got["fpi"] == pytest.approx(18.5)


def test_never_reads_a_snapshot_taken_after_the_prediction_week():
    snap = _snap([_row("Tennessee", 4, 22.0, 20.0)])
    assert ratings_for(_store(snap), 2026, 3, "Tennessee")["sp_rating"] is None


def test_falls_back_to_an_older_week_when_this_one_was_not_snapshotted():
    snap = _snap([_row("Tennessee", 1, 15.0, 14.0)])
    assert ratings_for(_store(snap), 2026, 3, "Tennessee")["sp_rating"] == pytest.approx(15.0)


def test_nulls_when_the_table_is_absent():
    assert all(v is None for v in ratings_for(_store(None), 2026, 3, "Tennessee").values())
