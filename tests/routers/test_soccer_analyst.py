import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app
from schemas.models import SoccerAnalystResponse


@pytest.fixture
def client():
    return TestClient(app)


def _mock_response():
    return SoccerAnalystResponse(
        question="who scored the most goals?",
        sql="SELECT player, goals FROM player_season ORDER BY goals DESC LIMIT 5",
        data=[{"player": "Haaland", "goals": 25}],
        chart={
            "data": [{"type": "bar", "x": ["Haaland"], "y": [25], "name": "goals"}],
            "layout": {"title": "Top Scorers", "xaxis": {"title": "player"}, "yaxis": {"title": "goals"}},
        },
        row_count=1,
    )


def test_soccer_analyst_returns_200(client):
    with patch(
        "routers.workflows.run_soccer_analyst",
        new=AsyncMock(return_value=_mock_response()),
        create=True,
    ):
        r = client.post(
            "/workflows/soccer-analyst",
            json={"question": "who scored the most goals?"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "who scored the most goals?"
    assert "sql" in body
    assert "chart" in body
    assert "data" in body
    assert body["row_count"] == 1


def test_soccer_analyst_requires_question(client):
    r = client.post("/workflows/soccer-analyst", json={})
    assert r.status_code == 422


def test_soccer_analyst_returns_422_on_workflow_error(client):
    with patch(
        "routers.workflows.run_soccer_analyst",
        new=AsyncMock(side_effect=Exception("DuckDB error: table not found")),
        create=True,
    ):
        r = client.post(
            "/workflows/soccer-analyst",
            json={"question": "bad question"},
        )
    assert r.status_code == 422
    assert "error" in r.json()
