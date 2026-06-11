import asyncio
from datetime import date
from typing import Optional

from effgen.tools.builtin import StockPriceTool, WeatherTool

from config import settings
from core.logging import logger
from core.storage.briefing_config import load_config
from core.storage.jobs import update_job
from core.storage.obsidian import write_brain_note
from core.tracing import observe
from schemas.models import BriefingConfig, BriefingOutput


def _load_model():
    from effgen import load_model
    return load_model(
        settings.llm_model,
        provider="openai",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


_model = None
_model_lock = asyncio.Lock()


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        logger.info(f"Connecting to LLM  base_url={settings.llm_base_url}  model={settings.llm_model}")
        model = await asyncio.to_thread(_load_model)
        _model = model
        logger.info("LLM connection ready")
    return _model


def _build_prompt(config: BriefingConfig) -> str:
    today = date.today().isoformat()
    parts = [
        f"Today is {today}. Produce a complete structured morning briefing.",
        "",
        "## Phase 1: Gather",
        f"1. Fetch weather for '{config.weather_location}':",
        f"   - Call WeatherTool with operation='current', location='{config.weather_location}', units='imperial'",
        f"   - Call WeatherTool with operation='forecast', location='{config.weather_location}', days=1, units='imperial'",
    ]

    if config.news_subjects:
        subjects_str = ", ".join(f"'{s}'" for s in config.news_subjects)
        parts += [
            "",
            f"2. For each of these subjects: {subjects_str}",
            "   - Call NewsTool with operation='search', query=<subject>, max_results=2",
            "   - Call HackerNewsTool with operation='top_stories', n=20, then pick up to 2 stories relevant to <subject> (skip if none match)",
        ]
        if config.rss_feeds:
            feeds_str = ", ".join(f"'{f}'" for f in config.rss_feeds)
            parts.append(
                f"   - For each RSS feed [{feeds_str}]: call RSSFeedTool with url=<feed_url>, query=<subject>, n=2"
            )

    if config.stocks:
        tickers_str = ", ".join(f"{s.ticker} ({s.name})" for s in config.stocks)
        parts += [
            "",
            f"3. Fetch stock data for: {tickers_str}",
            "   - Call StockPriceTool once per ticker using the exact symbol string.",
        ]

    parts += [
        "",
        "## Phase 2: Synthesize",
        "Weather: organize forecast data into morning (6am–12pm), afternoon (12pm–6pm),",
        "and evening (6pm–11pm) time buckets. ALL temperatures MUST be in Fahrenheit.",
    ]

    if config.news_subjects:
        parts += [
            "News: for each subject, merge results from all tools into a single story list.",
            "If two items describe the same event, keep the one with more detail (drop the duplicate).",
        ]

    if config.stocks:
        if config.stock_mode == "ticker-view":
            parts += [
                "Stocks (ticker-view): for each ticker record yesterday's closing price,",
                "percentage change from prior close, and a 1–2 sentence commentary.",
            ]
        else:
            parts += [
                "Stocks (portfolio-view): write one narrative paragraph summarizing",
                "how the full watchlist performed together, noting standouts and overall trend.",
            ]

    parts += [
        "",
        "## Phase 3: Format",
        "Return ONLY the BriefingOutput JSON.",
        "'weather' is always present (WeatherTool always runs).",
        "Omit the 'news' key entirely if no subjects were configured.",
        "Omit the 'stocks' key entirely if no stocks were configured.",
        "The forecast dict MUST use exactly these keys: 'morning', 'afternoon', 'evening'.",
        f"Set 'date' to '{today}'.",
    ]

    return "\n".join(parts)


def _run_briefing_sync(model, config: BriefingConfig) -> tuple[BriefingOutput, int]:
    from effgen import create_agent

    extra_tools = [WeatherTool()]
    if config.stocks:
        extra_tools.append(StockPriceTool())

    agent = create_agent(
        "research",
        model,
        extra_tools=extra_tools,
        tool_calling_mode="react",
    )

    task = _build_prompt(config)
    response = agent.run(task, output_model=BriefingOutput)

    if not response.success:
        reason = (response.metadata or {}).get("reason", "unknown")
        error = (response.metadata or {}).get("error", "")
        raise RuntimeError(
            f"Agent did not succeed after {response.iterations} iterations"
            f"  reason={reason}"
            + (f"  error={error}" if error else "")
        )

    parsed: Optional[BriefingOutput] = (response.metadata or {}).get("parsed")
    if parsed is None:
        raise RuntimeError("Agent succeeded but output could not be parsed as BriefingOutput")

    return parsed, response.tokens_used


def _format_briefing_note(output: BriefingOutput) -> str:
    lines = [f"# Morning Briefing — {output.date}", ""]
    w = output.weather
    lines += [
        "## Weather", "",
        f"**Location:** {w.location}",
        f"**Now:** {w.current.temp_f}°F, {w.current.condition}, {w.current.humidity} humidity, wind {w.current.wind}",
        "",
        "| Period | Temp | Condition |",
        "|--------|------|-----------|",
    ]
    for period, data in w.forecast.items():
        lines.append(f"| {period.capitalize()} | {data.temp_f}°F | {data.condition} |")
    lines.append("")

    if output.news:
        lines += ["## News", ""]
        for subject_block in output.news:
            lines.append(f"### {subject_block.subject}")
            for story in subject_block.stories:
                lines.append(f"- [{story.title}]({story.url}) — {story.source}")
                lines.append(f"  {story.summary}")
        lines.append("")

    if output.stocks:
        lines += ["## Stocks", ""]
        if output.stocks.tickers:
            for t in output.stocks.tickers:
                sign = "+" if t.change_pct >= 0 else ""
                lines.append(f"**{t.ticker}** ({t.name}): ${t.close:.2f} ({sign}{t.change_pct:.2f}%)")
                lines.append(f"  {t.summary}")
        elif output.stocks.narrative:
            lines.append(output.stocks.narrative)

    return "\n".join(lines)


@observe(name="briefing-workflow")
async def run_briefing(job_id: str) -> None:
    logger.info(f"job {job_id} → running  workflow=briefing")
    await update_job(job_id, "running")

    try:
        config = load_config(settings.briefing_config_path)
        model = await _get_model()

        try:
            parsed, tokens_used = await asyncio.to_thread(_run_briefing_sync, model, config)
        except Exception as e:
            if "connection" in str(e).lower():
                global _model
                _model = None
                logger.warning("LLM connection lost — cleared model cache")
            raise

        note_path = write_brain_note(
            "raw/briefings",
            f"briefing-{parsed.date}",
            _format_briefing_note(parsed),
            frontmatter={"date": f'"{parsed.date}"', "type": "briefing"},
        )

        result = parsed.model_dump(exclude_none=True)
        result["tokens_used"] = tokens_used
        result["obsidian_note"] = str(note_path)

        await update_job(job_id, "completed", result=result)
        logger.info(
            f"job {job_id} → completed  workflow=briefing  tokens={tokens_used}"
        )

    except Exception as e:
        logger.error(f"job {job_id} → failed  error={e}")
        await update_job(job_id, "failed", error=str(e))
