import asyncio
import time
from datetime import date

import polars as pl
import pytest

import workflows.cfb.backtest as b
import workflows.cfb.matchup as m
from workflows.cfb.matchup import Store, predict
from schemas.models import CfbBacktestGame


MODEL_ROW = {
    "fitted_at": "2026-10-01T00:00:00",
    "trained_through_season": 2026, "shrink_k": 2,
    "win_intercept": 0.0, "win_b_elo_diff": 0.01, "win_b_adj_diff": 0.1,
    "win_b_eff_diff": 0.0, "win_b_home_field": 0.5,
    "margin_intercept": 0.0, "margin_b_elo_diff": 0.04, "margin_b_adj_diff": 0.6,
    "margin_b_eff_diff": 0.0, "margin_b_home_field": 2.5,
    "total_intercept": 50.0, "total_b_home_field": 0.0,
    "total_b_off_sum": 0.0, "total_b_def_sum": 0.0,
}


def _profile_row(team, season, through_week, elo, adj_margin, conference="SEC"):
    return {"team": team, "season": season, "through_week": through_week,
            "conference": conference, "games": through_week, "wins": 1, "losses": 0,
            "elo": elo, "adj_margin": adj_margin}


def _profile_df():
    rows = []
    for tw in (0, 1, 2):
        rows.append(_profile_row("Georgia", 2026, tw, 1800.0, 10.0))
        rows.append(_profile_row("Alabama", 2026, tw, 1700.0, 4.0))
    # No week-0 row for Vanderbilt, so its week-1 game cannot be replayed.
    rows.append(_profile_row("Vanderbilt", 2026, 1, 1400.0, -8.0))
    rows.append(_profile_row("Vanderbilt", 2026, 2, 1400.0, -8.0))
    return pl.DataFrame(rows)


def _games_df():
    return pl.DataFrame({
        "id": [10, 11, 12, 13],
        "season": [2026, 2026, 2026, 2026],
        "week": [1, 2, 1, 3],
        "season_type": ["regular"] * 4,
        "completed": [True, True, True, False],
        "neutral_site": [False, True, False, False],
        "start_date": [date(2026, 9, 5), date(2026, 9, 12), date(2026, 9, 5),
                       date(2026, 9, 19)],
        "home_team": ["Georgia", "Georgia", "Vanderbilt", "Georgia"],
        "away_team": ["Alabama", "Alabama", "Alabama", "Alabama"],
        "home_classification": ["fbs"] * 4,
        "away_classification": ["fbs"] * 4,
        "home_points": [27, 17, 10, None],
        "away_points": [20, 24, 35, None],
        "home_pregame_elo": [1800.0, 1810.0, 1400.0, 1820.0],
        "away_pregame_elo": [1700.0, 1690.0, 1700.0, 1680.0],
    })


@pytest.fixture
def fake_store(monkeypatch):
    store = Store(profile=_profile_df(), games=_games_df(), model_row=MODEL_ROW,
                  loaded_at=time.monotonic())
    monkeypatch.setattr(m, "_store", store)
    return store


@pytest.fixture(autouse=True)
def no_stored_predictions(monkeypatch):
    monkeypatch.setattr(b.predictions, "read", lambda: None)


def _scored(**kw) -> CfbBacktestGame:
    base = dict(game_id=1, week=1, date="2026-09-05", home_team="A", away_team="B",
                neutral_site=False, source="replayed", through_week=0, fitted_at="f",
                home_win_prob=0.7, projected_home_points=30.0, projected_away_points=20.0,
                expected_margin=10.0, actual_home_points=28, actual_away_points=21,
                actual_margin=7, actual_total=49, margin_error=3.0, total_error=1.0,
                straight_up_correct=True)
    return CfbBacktestGame(**{**base, **kw})


# --- scoring one game ---

def test_score_joins_prediction_to_final():
    game = _games_df().row(0, named=True)
    pred = {"through_week": 0, "fitted_at": "f", "home_win_prob": 0.8,
            "projected_home_points": 30.0, "projected_away_points": 20.0,
            "expected_margin": 10.0}
    g = b.score(game, pred, "stored")
    assert g.game_id == 10 and g.week == 1 and g.date == "2026-09-05"
    assert g.source == "stored" and g.through_week == 0
    assert g.actual_margin == 7 and g.actual_total == 47
    assert g.margin_error == pytest.approx(3.0)
    assert g.total_error == pytest.approx(3.0)
    assert g.straight_up_correct is True


def test_score_counts_a_wrong_side_as_a_miss():
    game = _games_df().row(1, named=True)       # Georgia 17, Alabama 24
    pred = {"through_week": 1, "home_win_prob": 0.8, "projected_home_points": 30.0,
            "projected_away_points": 20.0, "expected_margin": 10.0}
    assert b.score(game, pred, "replayed").straight_up_correct is False


def test_score_counts_a_coin_flip_as_a_miss():
    game = _games_df().row(0, named=True)
    pred = {"through_week": 0, "home_win_prob": 0.5, "projected_home_points": 25.0,
            "projected_away_points": 25.0, "expected_margin": 0.0}
    assert b.score(game, pred, "replayed").straight_up_correct is False


# --- metrics ---

def test_metrics_of_nothing_report_no_games():
    assert b.metrics([]) == {"games": 0}


def test_metrics_cover_the_whole_set():
    games = [
        _scored(home_win_prob=1.0, margin_error=2.0, total_error=4.0, straight_up_correct=True),
        _scored(home_win_prob=0.0, margin_error=-8.0, total_error=-10.0,
                actual_margin=-5, straight_up_correct=True),
        _scored(home_win_prob=0.4, margin_error=20.0, total_error=0.0,
                actual_margin=3, straight_up_correct=False),
    ]
    mt = b.metrics(games)
    assert mt["games"] == 3
    assert mt["straight_up_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert mt["brier"] == pytest.approx((0.0 + 0.0 + 0.36) / 3, abs=1e-4)
    assert mt["margin_mae"] == pytest.approx(10.0)
    assert mt["margin_median_abs_error"] == pytest.approx(8.0)
    assert mt["margin_bias"] == pytest.approx(14 / 3, abs=1e-2)
    assert mt["within_3"] == pytest.approx(1 / 3, abs=1e-4)
    assert mt["within_7"] == pytest.approx(1 / 3, abs=1e-4)
    assert mt["within_10"] == pytest.approx(2 / 3, abs=1e-4)
    assert mt["within_14"] == pytest.approx(2 / 3, abs=1e-4)
    assert mt["total_mae"] == pytest.approx(14 / 3, abs=1e-2)
    assert mt["total_bias"] == pytest.approx(-2.0)
    assert mt["home_score_mae"] == pytest.approx(2.0)
    assert mt["away_score_mae"] == pytest.approx(1.0)


def test_brier_scores_a_loss_against_zero():
    assert b.metrics([_scored(home_win_prob=1.0, actual_margin=-3,
                              straight_up_correct=False)])["brier"] == pytest.approx(1.0)


def test_by_week_splits_and_sorts():
    weeks = b.by_week([_scored(week=2), _scored(week=1), _scored(week=1)])
    assert [w.week for w in weeks] == [1, 2]
    assert [w.games for w in weeks] == [2, 1]


# --- choosing stored predictions ---

def _stored(game_id, week, through_week, prob=0.9, fitted="f"):
    return {"game_id": game_id, "season": 2026, "week": week, "through_week": through_week,
            "model_fitted_at": fitted, "home_win_prob": prob,
            "projected_home_points": 30.0, "projected_away_points": 20.0,
            "expected_margin": 10.0, "generated_at": "2026-09-01T00:00:00"}


def test_pregame_predictions_prefers_the_latest_pregame_look():
    stored = pl.DataFrame([_stored(10, 3, 0, prob=0.6), _stored(10, 3, 2, prob=0.8)])
    best = b.pregame_predictions(stored, 2026)
    assert best[10]["home_win_prob"] == 0.8


def test_pregame_predictions_ignore_rows_stamped_after_kickoff():
    stored = pl.DataFrame([_stored(10, 3, 3), _stored(10, 3, 5)])
    assert b.pregame_predictions(stored, 2026) == {}


def test_pregame_predictions_ignore_other_seasons():
    stored = pl.DataFrame([{**_stored(10, 3, 2), "season": 2025}])
    assert b.pregame_predictions(stored, 2026) == {}


def test_pregame_predictions_tolerate_no_table():
    assert b.pregame_predictions(None, 2026) == {}
    assert b.pregame_predictions(pl.DataFrame({"game_id": []}), 2026) == {}


# --- the endpoint's runner ---

def test_run_backtest_replays_when_nothing_is_stored(fake_store):
    resp = asyncio.run(b.run_backtest())
    assert resp.season == 2026 and resp.week is None
    assert resp.fitted_at == "2026-10-01T00:00:00"
    assert resp.trained_through_season == 2026
    # Week 1 Georgia-Alabama and week 2 Georgia-Alabama replay; Vanderbilt's
    # week-1 game has no week-0 profile row; the week-3 game is unplayed.
    assert [g.game_id for g in resp.games] == [10, 11]
    assert resp.stored_games == 0 and resp.replayed_games == 2
    assert resp.skipped_games == 1
    assert all(g.source == "replayed" for g in resp.games)
    assert [w.week for w in resp.by_week] == [1, 2]
    assert resp.overall.games == 2


def test_replayed_rows_match_the_slate_formula(fake_store):
    resp = asyncio.run(b.run_backtest(week=1))
    g = resp.games[0]
    expected = predict(MODEL_ROW, elo_diff=100.0, adj_diff=6.0, home_field=1.0)
    assert g.home_win_prob == pytest.approx(expected.home_win_prob)
    assert g.expected_margin == pytest.approx(expected.expected_margin)
    assert g.through_week == 0
    assert g.actual_home_points == 27 and g.actual_away_points == 20


def test_replay_uses_the_stored_pregame_elo_not_the_profile(fake_store):
    resp = asyncio.run(b.run_backtest(week=2))
    g = resp.games[0]
    # week-2 game row carries 1810 / 1690, the profile says 1800 / 1700
    expected = predict(MODEL_ROW, elo_diff=120.0, adj_diff=6.0, home_field=0.0)
    assert g.home_win_prob == pytest.approx(expected.home_win_prob)
    assert g.neutral_site is True


def test_stored_predictions_win_over_the_replay(fake_store, monkeypatch):
    monkeypatch.setattr(b.predictions, "read",
                        lambda: pl.DataFrame([_stored(10, 1, 0, prob=0.33,
                                                      fitted="2026-09-01T00:00:00")]))
    resp = asyncio.run(b.run_backtest(week=1))
    g = resp.games[0]
    assert g.source == "stored" and g.home_win_prob == 0.33
    assert g.fitted_at == "2026-09-01T00:00:00"
    assert g.straight_up_correct is False      # picked Alabama, Georgia won
    assert resp.stored_games == 1 and resp.replayed_games == 0


def test_a_week_filter_narrows_the_population(fake_store):
    assert [g.game_id for g in asyncio.run(b.run_backtest(week=2)).games] == [11]


def test_a_season_with_no_finished_games_reports_an_empty_set(fake_store):
    resp = asyncio.run(b.run_backtest(week=3))
    assert resp.games == [] and resp.by_week == []
    assert resp.overall.games == 0 and resp.overall.margin_mae is None
