# tests/cfb/test_analyst.py
import pytest
import duckdb
import polars as pl
from unittest.mock import patch, MagicMock
import pyarrow as pa
from schemas.models import SoccerQueryPlan, CfbAnalystRequest


def _make_arrow_table():
    return pa.table({"school": ["Alabama", "Georgia"], "points": [42, 35], "year": [2024, 2024]})


def test_build_connection_registers_all_four_tables(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "minio_endpoint", "http://fake:9000")
    monkeypatch.setattr(settings, "minio_access_key", "key")
    monkeypatch.setattr(settings, "minio_secret_key", "secret")
    monkeypatch.setattr(settings, "minio_bucket", "sports-data")

    mock_dt = MagicMock()
    mock_dt.to_pyarrow_table.return_value = _make_arrow_table()

    with patch("workflows.cfb.analyst.DeltaTable", return_value=mock_dt):
        from workflows.cfb.analyst import _build_connection
        con = _build_connection(power4_only=True)

    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    assert "cfb_games" in tables
    assert "cfb_team_season_stats" in tables
    assert "cfb_player_season" in tables
    assert "cfb_team_game_stats" in tables


def test_build_connection_applies_partition_filter_for_power4(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "minio_endpoint", "http://fake:9000")
    monkeypatch.setattr(settings, "minio_access_key", "key")
    monkeypatch.setattr(settings, "minio_secret_key", "secret")
    monkeypatch.setattr(settings, "minio_bucket", "sports-data")

    mock_dt = MagicMock()
    mock_dt.to_pyarrow_table.return_value = _make_arrow_table()

    with patch("workflows.cfb.analyst.DeltaTable", return_value=mock_dt):
        from workflows.cfb.analyst import _build_connection
        _build_connection(power4_only=True)

    # to_pyarrow_table should be called 4 times total;
    # 3 conference-filtered tables get partitions=[("conference", "in", [...])]
    calls = mock_dt.to_pyarrow_table.call_args_list
    assert len(calls) == 4
    # cfb_games call has no partition filter (None)
    games_call = calls[0]
    assert games_call.kwargs.get("partitions") is None
    # the other three have partition filters
    for call in calls[1:]:
        partitions = call.kwargs.get("partitions")
        assert partitions is not None
        assert partitions[0][0] == "conference"


def test_build_prompt_contains_question():
    from workflows.cfb.analyst import _build_prompt
    prompt = _build_prompt("who led the SEC in rushing?", power4_only=True)
    assert "who led the SEC in rushing?" in prompt


def test_build_prompt_contains_all_table_names():
    from workflows.cfb.analyst import _build_prompt
    prompt = _build_prompt("test", power4_only=False)
    assert "cfb_games" in prompt
    assert "cfb_team_season_stats" in prompt
    assert "cfb_player_season" in prompt
    assert "cfb_team_game_stats" in prompt


def test_build_prompt_power4_mentions_conferences():
    from workflows.cfb.analyst import _build_prompt
    prompt = _build_prompt("test", power4_only=True)
    assert "SEC" in prompt
    assert "Big Ten" in prompt


def test_call_llm_sync_parses_clean_json():
    from workflows.cfb.analyst import _call_llm_sync
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        '{"sql": "SELECT school, points FROM cfb_team_game_stats LIMIT 5", '
        '"chart_type": "bar", "x_column": "school", '
        '"y_column": "points", "title": "Top Teams"}'
    )
    result = _call_llm_sync(mock_client, "test prompt")
    assert result.sql == "SELECT school, points FROM cfb_team_game_stats LIMIT 5"
    assert result.chart_type == "bar"


def test_execute_sql_returns_polars_df():
    from workflows.cfb.analyst import _execute_sql
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE cfb_games AS SELECT 1 AS id, 2024 AS season, 'Alabama' AS home_team, 'Georgia' AS away_team, 42 AS home_points, 35 AS away_points"
    )
    df = _execute_sql(con, "SELECT home_team, home_points FROM cfb_games")
    assert isinstance(df, pl.DataFrame)
    assert df["home_team"][0] == "Alabama"
    assert df["home_points"][0] == 42


def test_build_plotly_figure():
    from workflows.cfb.analyst import _build_plotly_figure
    df = pl.DataFrame({"school": ["Alabama", "Georgia"], "points": [42, 35]})
    plan = SoccerQueryPlan(
        sql="SELECT school, points FROM cfb_team_game_stats LIMIT 2",
        chart_type="bar",
        x_column="school",
        y_column="points",
        title="Top Scoring Teams",
    )
    fig = _build_plotly_figure(df, plan)
    assert fig["data"][0]["type"] == "bar"
    assert fig["data"][0]["x"] == ["Alabama", "Georgia"]
    assert fig["data"][0]["y"] == [42.0, 35.0]
    assert fig["layout"]["title"] == "Top Scoring Teams"
