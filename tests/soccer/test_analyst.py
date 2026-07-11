import pytest
from schemas.models import SoccerQueryPlan, SoccerAnalystRequest, SoccerAnalystResponse


def test_soccer_query_plan_parses_json():
    raw = '{"sql": "SELECT player FROM player_season LIMIT 5", "chart_type": "bar", "x_column": "player", "y_column": "assists", "title": "Top Assisters"}'
    plan = SoccerQueryPlan.model_validate_json(raw)
    assert plan.sql == "SELECT player FROM player_season LIMIT 5"
    assert plan.chart_type == "bar"
    assert plan.x_column == "player"
    assert plan.y_column == "assists"
    assert plan.title == "Top Assisters"


def test_soccer_analyst_request_requires_question():
    req = SoccerAnalystRequest(question="who scored the most goals?")
    assert req.question == "who scored the most goals?"


def test_soccer_analyst_response_shape():
    resp = SoccerAnalystResponse(
        question="test",
        sql="SELECT 1",
        data=[{"player": "Salah", "goals": 20}],
        chart={"data": [], "layout": {}},
        row_count=1,
    )
    assert resp.row_count == 1
    assert resp.data[0]["player"] == "Salah"
