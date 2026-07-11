# workflows/soccer/analyst.py
import asyncio
from typing import Optional

import openai as openai_lib
import duckdb
import polars as pl
from deltalake import DeltaTable

from config import settings
from core.logging import logger
from schemas.models import SoccerAnalystRequest, SoccerAnalystResponse, SoccerQueryPlan

_TABLE_PATHS = {
    "fixtures": "soccer/fixtures/",
    "player_match": "soccer/player_match/",
    "player_season": "soccer/player_season/",
    "team_stats": "soccer/team_stats/",
}

_TABLE_SCHEMAS = {
    "fixtures": (
        "game_id (TEXT), date (DATE), home_team (TEXT), away_team (TEXT), "
        "home_goals (INT), away_goals (INT), home_xg (FLOAT), away_xg (FLOAT), "
        "season (INT)"
    ),
    "player_match": (
        "player (TEXT), team (TEXT), position (TEXT), goals (INT), assists (INT), "
        "xg (FLOAT), xa (FLOAT), shots (INT), key_passes (INT), minutes (INT), "
        "yellow_cards (INT), red_cards (INT), season (INT), game_id (TEXT)"
    ),
    "player_season": (
        "player (TEXT), team (TEXT), position (TEXT), matches (INT), minutes (INT), "
        "goals (INT), xg (FLOAT), np_goals (INT), np_xg (FLOAT), assists (INT), "
        "xa (FLOAT), shots (INT), key_passes (INT), yellow_cards (INT), "
        "red_cards (INT), season (INT)"
    ),
    "team_stats": (
        "home_team (TEXT), away_team (TEXT), date (DATE), "
        "home_goals (INT), away_goals (INT), home_xg (FLOAT), away_xg (FLOAT), "
        "home_expected_points (FLOAT), away_expected_points (FLOAT), "
        "home_ppda (FLOAT), away_ppda (FLOAT), "
        "home_deep_completions (INT), away_deep_completions (INT), "
        "season (INT), game_id (TEXT)"
    ),
}

_con: Optional[duckdb.DuckDBPyConnection] = None
_con_lock = asyncio.Lock()


def _build_connection() -> duckdb.DuckDBPyConnection:
    storage_options = {
        "endpoint_url": settings.minio_endpoint,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }

    con = duckdb.connect()
    for table_name, path_suffix in _TABLE_PATHS.items():
        path = f"s3://{settings.minio_bucket}/{path_suffix}"
        dt = DeltaTable(path, storage_options=storage_options)
        arrow_table = dt.to_pyarrow_table()
        con.register(table_name, arrow_table)
        logger.info(f"soccer analyst: loaded {table_name} ({arrow_table.num_rows} rows)")

    return con


async def _get_con() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is not None:
        return _con
    async with _con_lock:
        if _con is not None:
            return _con
        _con = await asyncio.to_thread(_build_connection)
    return _con


_SYSTEM_PROMPT = (
    "You are a soccer data analyst. You write DuckDB SQL queries. "
    "Tables are already loaded in memory — use plain table names, not delta_scan(). "
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


def _build_prompt(question: str) -> str:
    schema_lines = [
        f"{table}: {columns}"
        for table, columns in _TABLE_SCHEMAS.items()
    ]
    schema_str = "\n".join(schema_lines)

    return (
        f"Answer this question about Premier League 2024 data: {question}\n\n"
        f"Available tables (season column is integer 2024):\n{schema_str}\n\n"
        "Return JSON with exactly these fields:\n"
        '- sql: DuckDB SQL using plain table names (e.g. SELECT * FROM player_season)\n'
        '- chart_type: one of "bar", "line", "scatter"\n'
        "- x_column: column name for x axis\n"
        "- y_column: column name for y axis\n"
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
