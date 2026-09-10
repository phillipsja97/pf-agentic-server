import math
import polars as pl
import pytest
from workflows.cfb.matchup import (
    predict, build_radar, build_edges, suggest_teams, latest_through_week,
    CfbMatchupError, RADAR_AXES,
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
