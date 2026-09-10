# workflows/cfb/analyst.py
import asyncio
from typing import Optional

import openai as openai_lib
import duckdb
import polars as pl
from deltalake import DeltaTable

from config import settings
from core.logging import logger
from schemas.models import CfbAnalystRequest, SoccerAnalystResponse, SoccerQueryPlan

POWER4 = ["SEC", "Big Ten", "ACC", "Big 12"]

_CONFERENCE_TABLES = {"cfb_team_season_stats", "cfb_player_season", "cfb_team_game_stats"}
_TABLE_NAMES = ["cfb_games", "cfb_team_season_stats", "cfb_player_season", "cfb_team_game_stats"]

_S3_PATHS = {
    "cfb_games":             "cfb/games",
    "cfb_team_season_stats": "cfb/team_season_stats",
    "cfb_player_season":     "cfb/player_season",
    "cfb_team_game_stats":   "cfb/team_game_stats",
}

_TABLE_SCHEMAS = {
    "cfb_games": (
        "id (INT), season (INT), week (INT), start_date (DATE), "
        "home_team (TEXT), away_team (TEXT), home_points (INT), away_points (INT), "
        "conference_game (BOOL), home_conference (TEXT), away_conference (TEXT), year (INT)"
    ),
    "cfb_team_season_stats": (
        "season (INT), team (TEXT), conference (TEXT), stat_name (TEXT), stat_value (FLOAT), year (INT) "
        "-- NARROW format: one row per stat per team. Use WHERE stat_name = '...' to isolate a stat."
    ),
    "cfb_player_season": (
        "season (INT), player (TEXT), position (TEXT), team (TEXT), conference (TEXT), "
        "category (TEXT), stat_type (TEXT), stat (FLOAT), year (INT) "
        "-- NARROW format: one row per stat per player. "
        "category values (lowercase): rushing, passing, receiving, defensive, kicking, punting, kickReturns, puntReturns, interceptions, fumbles. "
        "stat_type values are abbreviations, e.g. rushing: YDS, CAR, TD, YPC, LONG; passing: YDS, TD, INT, ATT, COMP; receiving: YDS, REC, TD, LONG. "
        "Use WHERE category = '...' AND stat_type = '...' to isolate a stat."
    ),
    "cfb_team_game_stats": (
        "game_id (INT), school (TEXT), conference (TEXT), home_away (TEXT), points (INT), year (INT)"
    ),
}

_cons: dict[str, duckdb.DuckDBPyConnection] = {}
_con_locks: dict[str, asyncio.Lock] = {}
_con_locks_lock = asyncio.Lock()


class CfbAnalystError(Exception):
    def __init__(self, message: str, sql: str):
        super().__init__(message)
        self.sql = sql


def _build_connection(power4_only: bool) -> duckdb.DuckDBPyConnection:
    storage_options = {
        "endpoint_url": settings.minio_endpoint,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }

    con = duckdb.connect()
    for table_name in _TABLE_NAMES:
        path = f"s3://{settings.minio_bucket}/{_S3_PATHS[table_name]}/"
        dt = DeltaTable(path, storage_options=storage_options)

        if power4_only and table_name in _CONFERENCE_TABLES:
            partitions = [("conference", "in", POWER4)]
        else:
            partitions = None

        arrow_table = dt.to_pyarrow_table(partitions=partitions)
        con.register(table_name, arrow_table)
        label = "Power 4" if power4_only else "all conferences"
        logger.info(f"cfb analyst: loaded {table_name} for {label!r} ({arrow_table.num_rows} rows)")

    return con


async def _get_con(power4_only: bool) -> duckdb.DuckDBPyConnection:
    key = "p4" if power4_only else "all"
    if key in _cons:
        return _cons[key]

    async with _con_locks_lock:
        if key not in _con_locks:
            _con_locks[key] = asyncio.Lock()

    async with _con_locks[key]:
        if key not in _cons:
            _cons[key] = await asyncio.to_thread(_build_connection, power4_only)
    return _cons[key]


_SYSTEM_PROMPT = (
    "You are a college football data analyst. You write DuckDB SQL queries. "
    "Tables are already loaded in memory — use plain table names, not delta_scan(). "
    "Stat tables (cfb_team_season_stats, cfb_player_season) are in narrow/tall format — "
    "use WHERE clauses on category or stat_type to isolate a specific stat. "
    "Season is a plain 4-digit year integer. Available seasons: 2015 through 2024. "
    "If the user asks about a season outside this range, use the closest available season. "
    "Return only valid JSON with no explanation or markdown."
)

_client: Optional[openai_lib.OpenAI] = None
_client_lock = asyncio.Lock()


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


def _build_prompt(question: str, power4_only: bool) -> str:
    schema_lines = [
        f"{table}: {columns}"
        for table, columns in _TABLE_SCHEMAS.items()
    ]
    schema_str = "\n".join(schema_lines)

    if power4_only:
        scope = "Power 4 conferences (SEC, Big Ten, ACC, Big 12)"
        scope_note = "Data is pre-filtered to Power 4 conferences (SEC, Big Ten, ACC, Big 12)."
    else:
        scope = "all FBS"
        scope_note = "Data covers all FBS conferences."

    return (
        f"Answer this question about {scope} college football data: {question}\n\n"
        f"{scope_note} Available seasons: 2015–2024 only. If the question names a season outside this range, use the closest one.\n\n"
        f"Available tables:\n{schema_str}\n\n"
        "Return JSON with exactly these fields:\n"
        '- sql: DuckDB SQL using plain table names\n'
        '- chart_type: one of "bar", "line", "scatter"\n'
        "- x_column: column name for x axis\n"
        "- y_column: column name for y axis\n"
        '- series_column: column that groups or identifies series — null if not applicable\n'
        "- title: descriptive chart title"
    )


def _call_llm_sync(client: openai_lib.OpenAI, prompt: str) -> SoccerQueryPlan:
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
    return SoccerQueryPlan.model_validate_json(raw.strip())


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


async def run_cfb_analyst(request: CfbAnalystRequest) -> SoccerAnalystResponse:
    client = await _get_client()
    con = await _get_con(request.power4_only)
    prompt = _build_prompt(request.question, request.power4_only)

    plan = await asyncio.to_thread(_call_llm_sync, client, prompt)

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
            raise CfbAnalystError(str(e), plan.sql)

    chart = _build_plotly_figure(df, plan)
    return SoccerAnalystResponse(
        question=request.question,
        sql=plan.sql,
        data=df.to_dicts(),
        chart=chart,
        row_count=len(df),
    )
