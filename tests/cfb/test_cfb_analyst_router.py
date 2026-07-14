# tests/cfb/test_cfb_analyst_router.py
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
        question="who led the SEC in rushing?",
        sql="SELECT player, SUM(stat) FROM cfb_player_season WHERE stat_category = 'rushing' AND stat_type = 'YDS' GROUP BY player ORDER BY SUM(stat) DESC LIMIT 5",
        data=[{"player": "Ollie Gordon", "stat": 1700.0}],
        chart={
            "data": [{"type": "bar", "x": ["Ollie Gordon"], "y": [1700.0], "name": "stat"}],
            "layout": {"title": "SEC Rushing Leaders", "xaxis": {"title": "player"}, "yaxis": {"title": "stat"}},
        },
        row_count=1,
    )


def test_cfb_analyst_returns_200(client):
    with patch(
        "routers.workflows.run_cfb_analyst",
        new=AsyncMock(return_value=_mock_response()),
        create=True,
    ):
        r = client.post(
            "/workflows/cfb-analyst",
            json={"question": "who led the SEC in rushing?"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "who led the SEC in rushing?"
    assert "sql" in body
    assert "chart" in body
    assert "data" in body
    assert body["row_count"] == 1


def test_cfb_analyst_defaults_power4_true(client):
    with patch(
        "routers.workflows.run_cfb_analyst",
        new=AsyncMock(return_value=_mock_response()),
        create=True,
    ) as mock_fn:
        client.post("/workflows/cfb-analyst", json={"question": "test"})
    call_args = mock_fn.call_args[0][0]
    assert call_args.power4_only is True


def test_cfb_analyst_requires_question(client):
    r = client.post("/workflows/cfb-analyst", json={})
    assert r.status_code == 422


def test_cfb_analyst_returns_422_on_workflow_error(client):
    with patch(
        "routers.workflows.run_cfb_analyst",
        new=AsyncMock(side_effect=Exception("DuckDB error: table not found")),
        create=True,
    ):
        r = client.post("/workflows/cfb-analyst", json={"question": "bad question"})
    assert r.status_code == 422
    assert "error" in r.json()
