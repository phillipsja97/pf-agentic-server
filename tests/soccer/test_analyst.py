import pytest
import duckdb
import polars as pl
from unittest.mock import patch, MagicMock
import pyarrow as pa
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


def _make_arrow_table():
    return pa.table({"player": ["Salah", "Haaland"], "goals": [20, 25], "season": [2024, 2024]})


def test_build_connection_registers_tables(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "minio_endpoint", "http://fake:9000")
    monkeypatch.setattr(settings, "minio_access_key", "key")
    monkeypatch.setattr(settings, "minio_secret_key", "secret")
    monkeypatch.setattr(settings, "minio_bucket", "sports-data")

    mock_dt = MagicMock()
    mock_dt.to_pyarrow_table.return_value = _make_arrow_table()

    with patch("workflows.soccer.analyst.DeltaTable", return_value=mock_dt):
        from workflows.soccer.analyst import _build_connection
        con = _build_connection(league=None)

    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    assert "fixtures" in tables
    assert "player_match" in tables
    assert "player_season" in tables
    assert "team_stats" in tables


from workflows.soccer.analyst import _build_prompt, _call_llm_sync


def test_build_prompt_contains_question():
    prompt = _build_prompt("who scored the most goals?", league=None)
    assert "who scored the most goals?" in prompt


def test_build_prompt_contains_all_table_names():
    prompt = _build_prompt("test", league=None)
    assert "fixtures" in prompt
    assert "player_match" in prompt
    assert "player_season" in prompt
    assert "team_stats" in prompt


def test_build_prompt_mentions_season_type():
    prompt = _build_prompt("test", league=None)
    assert "2024" in prompt


def test_call_llm_sync_parses_clean_json(monkeypatch):
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        '{"sql": "SELECT player FROM player_season LIMIT 5", '
        '"chart_type": "bar", "x_column": "player", '
        '"y_column": "goals", "title": "Top Scorers"}'
    )
    result = _call_llm_sync(mock_client, "test prompt")
    assert result.sql == "SELECT player FROM player_season LIMIT 5"
    assert result.chart_type == "bar"


def test_call_llm_sync_strips_markdown_fences(monkeypatch):
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "```json\n"
        '{"sql": "SELECT 1", "chart_type": "bar", '
        '"x_column": "a", "y_column": "b", "title": "T"}\n'
        "```"
    )
    result = _call_llm_sync(mock_client, "test prompt")
    assert result.sql == "SELECT 1"


from workflows.soccer.analyst import _execute_sql, _build_plotly_figure


def _make_in_memory_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE player_season AS SELECT 'Salah' AS player, 20 AS goals, "
        "'FW' AS position, 2024 AS season"
    )
    return con


def test_execute_sql_returns_polars_df():
    con = _make_in_memory_con()
    df = _execute_sql(con, "SELECT player, goals FROM player_season")
    assert isinstance(df, pl.DataFrame)
    assert df["player"][0] == "Salah"
    assert df["goals"][0] == 20


def test_execute_sql_raises_on_bad_sql():
    con = _make_in_memory_con()
    with pytest.raises(Exception):
        _execute_sql(con, "SELECT * FROM nonexistent_table")


def test_build_plotly_figure_bar_chart():
    df = pl.DataFrame({"player": ["Salah", "Haaland"], "goals": [20, 25]})
    plan = SoccerQueryPlan(
        sql="SELECT player, goals FROM player_season LIMIT 2",
        chart_type="bar",
        x_column="player",
        y_column="goals",
        title="Top Scorers",
    )
    fig = _build_plotly_figure(df, plan)
    assert fig["data"][0]["type"] == "bar"
    assert fig["data"][0]["x"] == ["Salah", "Haaland"]
    assert fig["data"][0]["y"] == [20, 25]
    assert fig["layout"]["title"] == "Top Scorers"


def test_build_plotly_figure_missing_column_returns_nones():
    df = pl.DataFrame({"player": ["Salah"], "goals": [20]})
    plan = SoccerQueryPlan(
        sql="SELECT 1",
        chart_type="bar",
        x_column="player",
        y_column="nonexistent",
        title="T",
    )
    fig = _build_plotly_figure(df, plan)
    assert fig["data"][0]["y"] == [None]


def test_comparison_query_plan_parses():
    from schemas.models import ComparisonQueryPlan
    plan = ComparisonQueryPlan(
        is_comparison=True,
        player_a="N'Golo Kante",
        season_a=1617,
        player_b="Moises Caicedo",
        season_b=2425,
        title="Kante 2016-17 vs Caicedo 2024-25",
    )
    assert plan.player_a == "N'Golo Kante"
    assert plan.season_a == 1617
    assert plan.is_comparison is True


def test_soccer_analyst_response_accepts_comparison_field():
    from schemas.models import SoccerAnalystResponse
    resp_no_comparison = SoccerAnalystResponse(
        question="test", sql="SELECT 1", data=[], chart={}, row_count=0
    )
    assert resp_no_comparison.comparison is None

    resp_with_comparison = SoccerAnalystResponse(
        question="test", sql="SELECT 1", data=[], chart={}, row_count=0,
        comparison={"title": "A vs B", "axes": ["Goals/90"], "series": []},
    )
    assert resp_with_comparison.comparison["title"] == "A vs B"
