# workflows/soccer/analyst.py
import asyncio
import json
from typing import Optional

import openai as openai_lib
import duckdb
import polars as pl
from deltalake import DeltaTable

from config import settings
from core.logging import logger
from schemas.models import SoccerAnalystRequest, SoccerAnalystResponse, SoccerQueryPlan, ComparisonQueryPlan


class SoccerAnalystError(Exception):
    def __init__(self, message: str, sql: str):
        super().__init__(message)
        self.sql = sql


_TABLE_NAMES = ["fixtures", "player_match", "player_season", "team_stats"]

_TABLE_SCHEMAS = {
    "fixtures": (
        "game_id (TEXT), date (DATE), home_team (TEXT), away_team (TEXT), "
        "home_goals (INT), away_goals (INT), home_xg (FLOAT), away_xg (FLOAT), "
        "season (INT), league (TEXT)"
    ),
    "player_match": (
        "player (TEXT), team (TEXT), position (TEXT), goals (INT), assists (INT), "
        "xg (FLOAT), xa (FLOAT), shots (INT), key_passes (INT), minutes (INT), "
        "yellow_cards (INT), red_cards (INT), season (INT), game_id (TEXT), league (TEXT)"
    ),
    "player_season": (
        "player (TEXT), team (TEXT), position (TEXT), matches (INT), minutes (INT), "
        "goals (INT), xg (FLOAT), np_goals (INT), np_xg (FLOAT), assists (INT), "
        "xa (FLOAT), shots (INT), key_passes (INT), yellow_cards (INT), "
        "red_cards (INT), season (INT), league (TEXT)"
    ),
    "team_stats": (
        "home_team (TEXT), away_team (TEXT), date (DATE), "
        "home_goals (INT), away_goals (INT), home_xg (FLOAT), away_xg (FLOAT), "
        "home_expected_points (FLOAT), away_expected_points (FLOAT), "
        "home_ppda (FLOAT), away_ppda (FLOAT), "
        "home_deep_completions (INT), away_deep_completions (INT), "
        "season (INT), game_id (TEXT), league (TEXT)"
    ),
}

_COMPARISON_SQL = """\
WITH base AS (
    SELECT player, season,
        goals       * 90.0 / NULLIF(minutes, 0) AS goals_p90,
        assists     * 90.0 / NULLIF(minutes, 0) AS assists_p90,
        xg          * 90.0 / NULLIF(minutes, 0) AS xg_p90,
        xa          * 90.0 / NULLIF(minutes, 0) AS xa_p90,
        key_passes  * 90.0 / NULLIF(minutes, 0) AS key_passes_p90,
        shots       * 90.0 / NULLIF(minutes, 0) AS shots_p90
    FROM player_season
    WHERE minutes > 0
),
ranked AS (
    SELECT player, season,
        ROUND(PERCENT_RANK() OVER (ORDER BY goals_p90)       * 100) AS goals_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY assists_p90)     * 100) AS assists_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY xg_p90)          * 100) AS xg_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY xa_p90)          * 100) AS xa_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY key_passes_p90)  * 100) AS key_passes_p90,
        ROUND(PERCENT_RANK() OVER (ORDER BY shots_p90)       * 100) AS shots_p90
    FROM base
)
SELECT * FROM ranked
WHERE (player ILIKE ? AND season = ?)
   OR (player ILIKE ? AND season = ?)
"""

_RADAR_AXES = ["Goals/90", "Assists/90", "xG/90", "xA/90", "Key Passes/90", "Shots/90"]
_RADAR_COLS = ["goals_p90", "assists_p90", "xg_p90", "xa_p90", "key_passes_p90", "shots_p90"]

# Per-league connection cache
_cons: dict[str, duckdb.DuckDBPyConnection] = {}
_con_locks: dict[str, asyncio.Lock] = {}
_con_locks_lock = asyncio.Lock()


def _build_connection(league: str | None) -> duckdb.DuckDBPyConnection:
    storage_options = {
        "endpoint_url": settings.minio_endpoint,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }

    con = duckdb.connect()
    for table_name in _TABLE_NAMES:
        path = f"s3://{settings.minio_bucket}/soccer/{table_name}/"
        dt = DeltaTable(path, storage_options=storage_options)
        partitions = [("league", "=", league)] if league else None
        arrow_table = dt.to_pyarrow_table(partitions=partitions)
        con.register(table_name, arrow_table)
        label = league or "all leagues"
        logger.info(f"soccer analyst: loaded {table_name} for {label!r} ({arrow_table.num_rows} rows)")

    return con


async def _get_con(league: str | None) -> duckdb.DuckDBPyConnection:
    key = league or "__all__"
    if key in _cons:
        return _cons[key]

    async with _con_locks_lock:
        if key not in _con_locks:
            _con_locks[key] = asyncio.Lock()

    async with _con_locks[key]:
        if key not in _cons:
            _cons[key] = await asyncio.to_thread(_build_connection, league)
    return _cons[key]


_SYSTEM_PROMPT = (
    "You are a soccer data analyst. You write DuckDB SQL queries. "
    "Tables are already loaded in memory — use plain table names, not delta_scan(). "
    "Return only valid JSON with no explanation or markdown."
)

_client: Optional[openai_lib.OpenAI] = None
_client_lock = asyncio.Lock()


def _parse_plan(raw: str) -> ComparisonQueryPlan | SoccerQueryPlan:
    data = json.loads(raw)
    if data.get("is_comparison"):
        return ComparisonQueryPlan.model_validate(data)
    return SoccerQueryPlan.model_validate(data)


async def _get_client() -> openai_lib.OpenAI:
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is not None:
            return _client
        _client = openai_lib.OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    return _client


def _build_prompt(question: str, league: str | None) -> str:
    schema_lines = [
        f"{table}: {columns}"
        for table, columns in _TABLE_SCHEMAS.items()
    ]
    schema_str = "\n".join(schema_lines)

    if league:
        scope = f"{league}"
        league_note = f"Data is pre-filtered to {league} — do not add a WHERE clause on the league column."
    else:
        scope = "all leagues"
        league_note = (
            "Data covers all leagues. You MUST include the league column in your SELECT "
            "so every result row is attributed to its league. For aggregation queries, "
            "include league in both SELECT and GROUP BY. "
            "Example — top scorers across all leagues:\n"
            "  SELECT player, league, SUM(goals) AS goals\n"
            "  FROM player_season\n"
            "  GROUP BY player, league\n"
            "  ORDER BY goals DESC\n"
            "  LIMIT 5"
        )

    return (
        f"Answer this question about {scope} soccer data: {question}\n\n"
        f"{league_note}\n\n"
        "Season encoding: the season column is an integer formed by concatenating the last two digits of the "
        "start and end calendar year (e.g. 2024-25 → 2425, 2016-17 → 1617, 2019-20 → 1920). "
        "For year ranges like '2016-2026' use BETWEEN (e.g. season BETWEEN 1617 AND 2526).\n\n"
        f"Available tables:\n{schema_str}\n\n"
        "Return JSON with exactly these fields:\n"
        '- sql: DuckDB SQL using plain table names (e.g. SELECT * FROM player_season)\n'
        '- chart_type: one of "bar", "line", "scatter"\n'
        "- x_column: column name for x axis\n"
        "- y_column: column name for y axis\n"
        '- series_column: column that groups or identifies series (e.g. "league") — null if not applicable\n'
        "- title: descriptive chart title\n\n"
        "If the question asks to compare two specific players across seasons, return this shape instead:\n"
        '  {"is_comparison": true, "player_a": "...", "season_a": YYZZ, "player_b": "...", "season_b": YYZZ, "title": "..."}\n'
        "Use the same YYZZ season encoding described above (e.g. 2016-17 → 1617).\n"
        "For all other questions, include \"is_comparison\": false alongside the existing fields."
    )


def _call_llm_sync(client: openai_lib.OpenAI, prompt: str) -> ComparisonQueryPlan | SoccerQueryPlan:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return _parse_plan(raw.strip())


def _execute_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame:
    return con.execute(sql).pl()


def _to_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _build_plotly_figure(df: pl.DataFrame, plan: SoccerQueryPlan) -> dict:
    rows = df.to_dicts()

    traces = [{
        "type": plan.chart_type,
        "x": [r.get(plan.x_column) for r in rows],
        "y": [_to_float(r.get(plan.y_column)) for r in rows],
        "name": plan.y_column,
    }]

    return {
        "data": traces,
        "layout": {
            "title": plan.title,
            "xaxis": {"title": plan.x_column},
            "yaxis": {"title": plan.y_column},
        },
    }


def _build_comparison(
    con: duckdb.DuckDBPyConnection,
    plan: ComparisonQueryPlan,
) -> tuple[str, list[dict], dict]:
    params = [plan.player_a, plan.season_a, plan.player_b, plan.season_b]
    try:
        df = con.execute(_COMPARISON_SQL, params).pl()
    except Exception as e:
        raise SoccerAnalystError(str(e), _COMPARISON_SQL)

    if len(df) == 0:
        raise SoccerAnalystError(
            "No data found for one or both players", _COMPARISON_SQL
        )
    if len(df) < 2:
        raise SoccerAnalystError(
            "Could not find data for both players in the specified seasons",
            _COMPARISON_SQL,
        )

    series = [
        {
            "name": row["player"],
            "values": [int(row.get(col) or 0) for col in _RADAR_COLS],
        }
        for row in df.to_dicts()
    ]
    radar_figure = {"title": plan.title, "axes": _RADAR_AXES, "series": series}
    return _COMPARISON_SQL, df.to_dicts(), radar_figure


async def run_soccer_analyst(request: SoccerAnalystRequest) -> SoccerAnalystResponse:
    client = await _get_client()
    con = await _get_con(request.league)
    prompt = _build_prompt(request.question, request.league)

    plan = await asyncio.to_thread(_call_llm_sync, client, prompt)

    if isinstance(plan, ComparisonQueryPlan):
        sql, data, comparison = await asyncio.to_thread(_build_comparison, con, plan)
        return SoccerAnalystResponse(
            question=request.question,
            sql=sql,
            data=data,
            chart={"data": [], "layout": {}},
            row_count=len(data),
            comparison=comparison,
        )

    try:
        df = await asyncio.to_thread(_execute_sql, con, plan.sql)
    except Exception as e:
        retry_prompt = (
            prompt
            + f"\n\nThe previous query failed with this error: {e}\n"
            "Fix the SQL and return corrected JSON."
        )
        plan = await asyncio.to_thread(_call_llm_sync, client, retry_prompt)
        try:
            df = await asyncio.to_thread(_execute_sql, con, plan.sql)
        except Exception as e:
            raise SoccerAnalystError(str(e), plan.sql)

    chart = _build_plotly_figure(df, plan)
    return SoccerAnalystResponse(
        question=request.question,
        sql=plan.sql,
        data=df.to_dicts(),
        chart=chart,
        row_count=len(df),
    )
