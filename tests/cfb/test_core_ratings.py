"""CORE on the matchup response: points per 100 plays, plus league context."""
import polars as pl
import pytest
from workflows.cfb.matchup import Store, core_for


def _core(rows):
    return pl.DataFrame(rows, schema={"season": pl.Int32, "through_week": pl.Int32,
                                      "team": pl.Utf8, "core_off": pl.Float64,
                                      "core_def": pl.Float64, "core_overall": pl.Float64})


def _row(team, off, dfn, season=2026, week=1):
    return {"season": season, "through_week": week, "team": team,
            "core_off": off, "core_def": dfn, "core_overall": off - dfn}


def _store(core):
    return Store(profile=None, games=None, model_row={}, loaded_at=0.0, core=core)


class TestCoreFor:
    def _sample(self):
        return _core([_row("Best", 20.0, -15.0), _row("Mid", 0.0, 0.0),
                      _row("Worst", -18.0, 12.0), _row("Other", 5.0, -2.0)])

    def test_returns_the_team_s_ratings_for_that_week(self):
        got = core_for(_store(self._sample()), 2026, 1, "Best")
        assert got["core_off"] == pytest.approx(20.0)
        assert got["core_def"] == pytest.approx(-15.0)
        assert got["core_overall"] == pytest.approx(35.0)

    def test_percentiles_are_outward_is_better_for_offence(self):
        got = core_for(_store(self._sample()), 2026, 1, "Best")
        assert got["core_off_pct"] == pytest.approx(100.0)   # best of the field
        worst = core_for(_store(self._sample()), 2026, 1, "Worst")
        assert worst["core_off_pct"] == pytest.approx(0.0)

    def test_a_suppressing_defence_scores_a_high_percentile_despite_being_negative(self):
        """core_def is points allowed, so the best defence is the lowest number."""
        best = core_for(_store(self._sample()), 2026, 1, "Best")
        worst = core_for(_store(self._sample()), 2026, 1, "Worst")
        assert best["core_def"] < worst["core_def"]
        assert best["core_def_pct"] > worst["core_def_pct"]

    def test_a_missing_team_yields_nulls_rather_than_raising(self):
        got = core_for(_store(self._sample()), 2026, 1, "Nobody")
        assert all(v is None for v in got.values())

    def test_an_empty_table_yields_nulls(self):
        got = core_for(_store(_core([])), 2026, 1, "Best")
        assert all(v is None for v in got.values())

    def test_only_the_requested_week_is_ranked_against(self):
        core = _core([_row("A", 5.0, 0.0, week=1), _row("C", 1.0, 0.0, week=1),
                      _row("B", 50.0, 0.0, week=2)])
        got = core_for(_store(core), 2026, 1, "A")
        # B's week-2 blowout must not drag A's week-1 percentile down.
        assert got["core_off_pct"] == pytest.approx(100.0)

    def test_a_field_of_one_sits_at_the_midpoint(self):
        got = core_for(_store(_core([_row("Lonely", 5.0, 0.0)])), 2026, 1, "Lonely")
        assert got["core_off_pct"] == pytest.approx(50.0)
