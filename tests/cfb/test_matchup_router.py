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
    def rec(t):
        return CfbTeamRecord(team=t, conference="SEC", wins=2, losses=0, games_played=2, color="#ff8200")
    return CfbMatchupResponse(
        season=2026, through_week=2, data_through_date="2026-09-05", neutral_site=False,
        home=rec("Georgia"), away=rec("Alabama"),
        prediction=CfbPrediction(home_win_prob=0.7, away_win_prob=0.3,
                                 projected_home_points=31, projected_away_points=21,
                                 expected_margin=10),
        model=CfbModelRecord(accuracy=0.7149, brier=0.1848, log_loss=0.5462,
                             elo_baseline_accuracy=0.7035, elo_baseline_brier=0.1927,
                             calibration_bins=[], trained_through_season=2026, shrink_k=2),
        comparison={"title": "t", "axes": ["a"], "series": []}, edges=[], head_to_head=[],
    )


def test_matchup_returns_200(client):
    with patch("routers.workflows.run_cfb_matchup", new=AsyncMock(return_value=_matchup())):
        r = client.post("/workflows/cfb/matchup",
                        json={"home_team": "Georgia", "away_team": "Alabama"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["home_win_prob"] == 0.7
    assert body["home"]["team"] == "Georgia"
    assert body["model"]["elo_baseline_accuracy"] == 0.7035


def test_matchup_unknown_team_is_422_with_suggestions(client):
    err = CfbMatchupError("Unknown team 'Georgi'", suggestions=["Georgia", "Georgia Tech"])
    with patch("routers.workflows.run_cfb_matchup", new=AsyncMock(side_effect=err)):
        r = client.post("/workflows/cfb/matchup",
                        json={"home_team": "Georgi", "away_team": "Alabama"})
    assert r.status_code == 422
    assert r.json() == {"error": "Unknown team 'Georgi'", "suggestions": ["Georgia", "Georgia Tech"]}


def test_slate_is_get_and_not_swallowed_by_job_id_route(client):
    resp = CfbSlateResponse(season=2026, week=3, through_week=2,
                            data_through_date="2026-09-05", games=[])
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
    with patch("routers.workflows.list_cfb_teams",
               new=AsyncMock(return_value=[CfbTeam(team="Georgia", conference="SEC")])):
        r = client.get("/workflows/cfb/teams")
    assert r.status_code == 200
    assert r.json() == [{"team": "Georgia", "conference": "SEC"}]
