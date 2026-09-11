import math
import polars as pl
import pytest
from workflows.cfb.matchup import (
    predict, build_radar, build_edges, suggest_teams, latest_through_week,
    CfbMatchupError, RADAR_AXES,
)


MODEL_ROW = {
    "win_intercept": 0.0, "win_b_elo_diff": 0.01, "win_b_adj_diff": 0.1,
    "win_b_eff_diff": 2.0, "win_b_home_field": 0.5,
    "margin_intercept": 0.0, "margin_b_elo_diff": 0.04, "margin_b_adj_diff": 0.6,
    "margin_b_eff_diff": 0.0, "margin_b_home_field": 2.5,
    "total_intercept": 50.0, "total_b_elo_diff": 0.0, "total_b_adj_diff": 0.0,
    "total_b_eff_diff": 0.0, "total_b_home_field": 0.0,
}

# A row persisted before eff_diff existed, to prove old models still load.
LEGACY_MODEL_ROW = {k: v for k, v in MODEL_ROW.items() if "eff_diff" not in k}


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


def test_predict_uses_the_efficiency_feature():
    base = predict(MODEL_ROW, elo_diff=0.0, adj_diff=0.0, home_field=0.0)
    better = predict(MODEL_ROW, elo_diff=0.0, adj_diff=0.0, home_field=0.0, eff_diff=0.25)
    assert better.home_win_prob > base.home_win_prob


def test_a_model_row_without_the_efficiency_column_still_predicts():
    # Coefficient columns absent from an older row are treated as zero.
    p = predict(LEGACY_MODEL_ROW, elo_diff=100.0, adj_diff=5.0, home_field=1.0, eff_diff=0.9)
    assert p.home_win_prob == pytest.approx(1 / (1 + math.exp(-2.0)))


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
    assert suggest_teams("georgia", names)[:3] == ["Georgia", "Georgia Southern", "Georgia Tech"]
    assert suggest_teams("Ohio St", names) == ["Ohio State"]
    assert suggest_teams("zzzz", names) == []


def test_latest_through_week_is_max_fetched_week_for_season():
    profile = pl.DataFrame({"season": [2026, 2026, 2026, 2025], "through_week": [0, 1, 2, 15],
                            "team": ["A"] * 4})
    assert latest_through_week(profile, 2026) == 2
    assert latest_through_week(profile, 2025) == 15
    with pytest.raises(CfbMatchupError):
        latest_through_week(profile, 2030)


# --- store + runners ---

import asyncio
import json
import time
from datetime import date
from unittest.mock import patch, MagicMock
import pyarrow as pa
import workflows.cfb.matchup as m
from workflows.cfb.matchup import Store, run_matchup, run_slate, list_teams, model_record
from schemas.models import CfbMatchupRequest


def _profile_df():
    rows = []
    for tw in (0, 1, 2):
        rows.append(_row("Georgia", season=2026, through_week=tw, games=tw,
                         elo=1800.0 + tw, adj_margin=10.0))
        rows.append(_row("Alabama", season=2026, through_week=tw, games=tw,
                         elo=1700.0 + tw, adj_margin=4.0))
        rows.append(_row("Georgia Tech", season=2026, through_week=tw, games=tw,
                         elo=1500.0, adj_margin=-2.0, conference="ACC"))
    rows.append(_row("Georgia", season=2025, through_week=15, games=13,
                     elo=1790.0, adj_margin=9.0))
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
    return {**MODEL_ROW, "shrink_k": 2, "trained_through_season": 2026,
            "accuracy": 0.7149, "brier": 0.1848, "log_loss": 0.5462,
            "elo_accuracy": 0.7035, "elo_brier": 0.1927, "elo_log_loss": 0.5682,
            "calibration_bins": json.dumps(
                [{"lo": 0.5, "hi": 0.6, "n": 10, "predicted": 0.55, "actual": 0.5}])}


def _store():
    return Store(profile=_profile_df(), games=_games_df(), model_row=_model_row(),
                 loaded_at=time.monotonic())


@pytest.fixture
def fake_store(monkeypatch):
    store = _store()
    monkeypatch.setattr(m, "_store", store)
    return store


def test_model_record_parses_bins_and_baseline():
    rec = model_record(_model_row())
    assert rec.accuracy == 0.7149 and rec.elo_baseline_accuracy == 0.7035 and rec.shrink_k == 2
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
    assert [g.season for g in resp.head_to_head] == [2026, 2024]   # completed only, newest first
    assert resp.model.trained_through_season == 2026


def test_run_matchup_neutral_and_explicit_week(fake_store):
    resp = asyncio.run(run_matchup(CfbMatchupRequest(
        home_team="Georgia", away_team="Alabama", neutral_site=True, through_week=0)))
    assert resp.through_week == 0 and resp.neutral_site is True
    expected = predict(_model_row(), elo_diff=100.0, adj_diff=6.0, home_field=0.0)
    assert resp.prediction.home_win_prob == pytest.approx(expected.home_win_prob)
    assert resp.data_through_date is None


def test_run_matchup_unknown_team_raises_with_suggestions(fake_store):
    with pytest.raises(CfbMatchupError) as ei:
        asyncio.run(run_matchup(CfbMatchupRequest(home_team="Georgi", away_team="Alabama")))
    assert "Georgia" in ei.value.suggestions and "Georgia Tech" in ei.value.suggestions


def test_run_slate_defaults_to_next_week_and_uses_game_elo(fake_store):
    resp = asyncio.run(run_slate(None))
    assert resp.season == 2026 and resp.through_week == 2 and resp.week == 3
    assert [g.home_team for g in resp.games] == ["Georgia", "Alabama"]
    g1 = resp.games[0]
    expected = predict(_model_row(), elo_diff=302.0, adj_diff=12.0, home_field=1.0)
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
    from config import settings
    monkeypatch.setattr(settings, "minio_bucket", "b")
    tables = {
        "cfb/team_profile": pa.table({"season": [2026], "through_week": [1], "team": ["Georgia"]}),
        "cfb/games": pa.table({"season": [2026]}),
        "cfb/model": pa.table({"fitted_at": [1, 2], "win_intercept": [0.0, 9.0]}),
    }

    def fake_dt(path, storage_options):
        key = next(k for k in tables if k in path)
        dt = MagicMock()
        dt.to_pyarrow_table.return_value = tables[key]
        return dt

    with patch("workflows.cfb.matchup.DeltaTable", side_effect=fake_dt):
        store = m._load_store()
    assert store.model_row["win_intercept"] == 9.0
    assert store.profile["team"].to_list() == ["Georgia"]
