import asyncio
from datetime import date
from typing import Optional

from effgen.tools.builtin.finance import StockPriceTool
from effgen.tools.builtin.hackernews import HackerNewsTool
from effgen.tools.builtin.news import NewsTool
from effgen.tools.builtin.rss import RSSFeedTool
from effgen.tools.builtin.weather import WeatherTool
from pydantic import BaseModel

from config import settings
from core.logging import logger
from core.storage.briefing_config import load_config
from core.storage.jobs import update_job
from core.storage.obsidian import write_brain_note
from core.tracing import is_tracing_enabled, observe
from schemas.models import (
    BriefingConfig,
    BriefingOutput,
    NewsStory,
    NewsSubjectOutput,
    StocksOutput,
    TickerOutput,
    WeatherCurrent,
    WeatherForecast,
    WeatherOutput,
)


class _StorySelection(BaseModel):
    indices: list[int]  # 1-based positions of selected stories


class _StocksAgentOutput(BaseModel):
    tickers: Optional[list[TickerOutput]] = None
    narrative: Optional[str] = None


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


def _record_agent_span(prompt: str, output_dict: dict, response) -> None:
    """Push prompt, output, token usage, and full execution trace into the current Langfuse span."""
    if not is_tracing_enabled():
        return
    from langfuse import get_client
    get_client().update_current_span(
        input={"prompt": prompt},
        output=output_dict,
        metadata={
            "tokens_used": response.tokens_used,
            "iterations": response.iterations,
            "tool_calls": response.tool_calls,
            "execution_time_s": round(response.execution_time, 2),
            "success": response.success,
            "execution_trace": response.execution_trace,
        },
    )


_WIND_DIRS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


@observe(name="briefing-weather", as_type="span")
async def _fetch_weather(config: BriefingConfig) -> WeatherOutput:
    """Fetch weather directly — no LLM agent needed for structured tool data."""
    wt = WeatherTool()

    async def _call(loc: str):
        curr = await wt.execute(operation='current', location=loc, units='imperial')
        if not curr.output.get('success'):
            return None, None
        fc = await wt.execute(operation='forecast', location=loc, days=1, units='imperial')
        if not fc.output.get('success'):
            return None, None
        return curr.output['data'], fc.output['data']

    loc = config.weather_location
    curr_data, fc_data = await _call(loc)
    if curr_data is None:
        # WeatherTool geocoder only accepts bare city names — try stripping state/country
        city = loc.split(',')[0].strip()
        logger.info(f"WeatherTool rejected '{loc}' — retrying with '{city}'")
        curr_data, fc_data = await _call(city)
    if curr_data is None:
        raise RuntimeError(f"WeatherTool could not resolve location: {loc!r}")

    wind_dir = _WIND_DIRS[round(curr_data.get('wind_direction', 0) / 22.5) % 16]
    current = WeatherCurrent(
        temp_f=curr_data['temperature'],
        condition=curr_data['conditions'],
        humidity=f"{curr_data['humidity']}%",
        wind=f"{curr_data['wind_speed']} mph {wind_dir}",
    )

    day = fc_data['forecast'][0]
    precip = day.get('precipitation_prob')
    wind_max = day.get('wind_speed_max')
    summary = (
        f"High of {day['temp_max']:.0f}°F, low of {day['temp_min']:.0f}°F, "
        f"{day['conditions'].lower()}"
        + (f", {precip}% chance of precipitation" if precip and precip > 10 else "")
        + "."
    )
    forecast = WeatherForecast(
        temp_high_f=day['temp_max'],
        temp_low_f=day['temp_min'],
        condition=day['conditions'],
        precipitation_chance=precip,
        wind_speed_max=wind_max,
        summary=summary,
    )

    weather_out = WeatherOutput(
        location=curr_data.get('location', loc),
        current=current,
        forecast=forecast,
    )

    if is_tracing_enabled():
        from langfuse import get_client
        get_client().update_current_span(
            input={"location": loc},
            output=weather_out.model_dump(),
        )

    return weather_out


async def _fetch_news_candidates(subject: str, rss_feeds: list[str]) -> list[dict]:
    """Directly fetch story candidates — no LLM. Returns list of {title, summary, source, url}."""
    candidates = []

    ht = HackerNewsTool()
    r = await ht.execute(operation='top_stories', n=10)
    if r.success and r.output:
        for story in r.output.get('data', {}).get('stories', []):
            title = story.get('title', '')
            url = story.get('url') or story.get('hn_url', '')
            if title and url:
                candidates.append({
                    'title': title,
                    'summary': story.get('text') or title,
                    'source': 'Hacker News',
                    'url': url,
                })

    for feed_url in rss_feeds:
        try:
            rss = RSSFeedTool()
            r2 = await rss.execute(url=feed_url, query=subject, n=3)
            if r2.success and r2.output:
                for item in r2.output.get('data', {}).get('items', []):
                    title = item.get('title', '')
                    url = item.get('url') or item.get('link', '')
                    if title and url:
                        candidates.append({
                            'title': title,
                            'summary': item.get('summary') or item.get('description') or title,
                            'source': item.get('feed_title') or feed_url,
                            'url': url,
                        })
        except Exception as e:
            logger.warning(f"RSS feed failed  url={feed_url}  error={e}")

    return candidates


@observe(name="briefing-news-subject", as_type="agent")
def _select_stories_sync(model, subject: str, candidates: list[dict]) -> tuple[list[NewsStory], int]:
    """Ask the LLM to pick relevant story indices. Minimal task — just numbers, no fabrication possible."""
    from effgen import create_agent

    listing = "\n".join(
        f"{i + 1}. [{c['source']}] {c['title']}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"From the numbered story list below, select up to 2 that are most relevant to the topic '{subject}'. "
        f"Return ONLY a JSON object: {{\"indices\": [1-based numbers]}}. "
        f"Return {{\"indices\": []}} if none are relevant.\n\n{listing}"
    )

    agent = create_agent("minimal", model, extra_tools=[], tool_calling_mode="react", max_iterations=3,
                         max_context_length=settings.llm_max_tokens)
    run_kwargs: dict = {}
    if settings.llm_max_tokens is not None:
        run_kwargs["max_tokens"] = settings.llm_max_tokens
    response = agent.run(prompt, output_model=_StorySelection, **run_kwargs)

    parsed: Optional[_StorySelection] = (response.metadata or {}).get("parsed")
    if is_tracing_enabled():
        from langfuse import get_client
        get_client().update_current_span(
            input={"subject": subject, "candidates": len(candidates)},
            output={"selected_indices": parsed.indices if parsed else []},
            metadata={"iterations": response.iterations, "success": response.success,
                      "tokens_used": response.tokens_used},
        )

    if not parsed or not parsed.indices:
        return [], response.tokens_used

    stories = []
    for idx in parsed.indices[:2]:
        i = idx - 1
        if 0 <= i < len(candidates):
            c = candidates[i]
            stories.append(NewsStory(
                title=c['title'],
                summary=c['summary'],
                source=c['source'],
                url=c['url'],
            ))
    return stories, response.tokens_used


async def _run_one_subject(model, subject: str, rss_feeds: list[str]) -> tuple[Optional[NewsSubjectOutput], int]:
    candidates = await _fetch_news_candidates(subject, rss_feeds)
    if not candidates:
        logger.info(f"no news candidates  subject='{subject}'")
        return None, 0

    stories, tokens = await asyncio.to_thread(_select_stories_sync, model, subject, candidates)
    if not stories:
        logger.info(f"no relevant stories selected  subject='{subject}'")
        return None, tokens

    return NewsSubjectOutput(subject=subject, stories=stories), tokens


async def _run_news(model, config: BriefingConfig) -> tuple[list[NewsSubjectOutput], int]:
    subjects = []
    total_tokens = 0
    for subject in config.news_subjects:
        logger.info(f"news running  subject='{subject}'")
        try:
            result, tokens = await _run_one_subject(model, subject, config.rss_feeds)
            total_tokens += tokens
            if result:
                subjects.append(result)
                logger.info(f"news done  subject='{subject}'  stories={len(result.stories)}")
            else:
                logger.info(f"news done  subject='{subject}'  stories=0")
        except Exception as e:
            logger.warning(f"news failed  subject='{subject}'  error={e}")
    return subjects, total_tokens


async def _fetch_stocks(config: BriefingConfig) -> tuple[list[TickerOutput], list[dict]]:
    """Fetch real prices directly from StockPriceTool. Returns (ticker_outputs, raw_data)."""
    tool = StockPriceTool()
    ticker_outputs = []
    raw_data = []

    for stock in config.stocks:
        result = await tool.execute(symbol=stock.ticker)
        if not result.success or not result.output:
            logger.warning(f"StockPriceTool failed for {stock.ticker}  error={result.error}")
            continue
        d = result.output
        price = d.get('price', 0.0)
        prev = d.get('previous_close') or price
        change_pct = ((price - prev) / prev * 100) if prev else 0.0
        direction = "up" if change_pct >= 0 else "down"
        summary = (
            f"Closed at ${price:.2f}, {direction} {abs(change_pct):.2f}% "
            f"from prior close of ${prev:.2f}."
        )
        ticker_outputs.append(TickerOutput(
            ticker=stock.ticker,
            name=stock.name,
            close=price,
            change_pct=round(change_pct, 2),
            summary=summary,
        ))
        raw_data.append({
            "ticker": stock.ticker,
            "name": stock.name,
            "price": price,
            "previous_close": prev,
            "change_pct": round(change_pct, 2),
        })

    return ticker_outputs, raw_data


@observe(name="briefing-stocks", as_type="span")
async def _run_stocks(model, config: BriefingConfig) -> tuple[StocksOutput, int]:
    """Fetch prices directly, use LLM only for portfolio-view narrative."""
    ticker_outputs, raw_data = await _fetch_stocks(config)

    if not ticker_outputs:
        raise RuntimeError("StockPriceTool returned no valid data for any configured ticker")

    tokens_used = 0

    if config.stock_mode == "ticker-view":
        stocks_out = StocksOutput(mode="ticker-view", tickers=ticker_outputs)
    else:
        # portfolio-view: LLM writes one narrative paragraph with real prices embedded
        today = date.today().isoformat()
        data_str = "\n".join(
            f"- {d['ticker']} ({d['name']}): ${d['price']:.2f} "
            f"({'+'if d['change_pct']>=0 else ''}{d['change_pct']:.2f}% vs prior close ${d['previous_close']:.2f})"
            for d in raw_data
        )
        prompt = (
            f"Today is {today}. Write one paragraph summarizing how this watchlist performed, "
            f"noting standouts and overall trend. Use ONLY the data below — do not change any numbers.\n\n"
            f"{data_str}"
        )
        narrative, tokens_used = await asyncio.to_thread(
            _run_narrative_sync, model, prompt
        )
        stocks_out = StocksOutput(mode="portfolio-view", narrative=narrative)

    if is_tracing_enabled():
        from langfuse import get_client
        get_client().update_current_span(
            input={"tickers": [s.ticker for s in config.stocks], "mode": config.stock_mode},
            output=stocks_out.model_dump(exclude_none=True),
            metadata={"tokens_used": tokens_used},
        )

    return stocks_out, tokens_used


def _run_narrative_sync(model, prompt: str) -> tuple[str, int]:
    from effgen import create_agent
    agent = create_agent("minimal", model, extra_tools=[], tool_calling_mode="react", max_iterations=3,
                         max_context_length=settings.llm_max_tokens)
    run_kwargs: dict = {}
    if settings.llm_max_tokens is not None:
        run_kwargs["max_tokens"] = settings.llm_max_tokens
    response = agent.run(prompt, **run_kwargs)
    return response.output or "", response.tokens_used


def _format_briefing_note(output: BriefingOutput) -> str:
    lines = [f"# Morning Briefing — {output.date}", ""]
    w = output.weather
    f = w.forecast
    precip = f" | {f.precipitation_chance}% precip" if f.precipitation_chance is not None else ""
    wind = f" | wind max {f.wind_speed_max} mph" if f.wind_speed_max is not None else ""
    lines += [
        "## Weather", "",
        f"**Location:** {w.location}",
        f"**Now:** {w.current.temp_f}°F, {w.current.condition}, {w.current.humidity} humidity, wind {w.current.wind}",
        f"**Forecast:** High {f.temp_high_f}°F / Low {f.temp_low_f}°F, {f.condition}{precip}{wind}",
        f"_{f.summary}_",
        "",
    ]

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

    global _model

    try:
        config = load_config(settings.briefing_config_path)

        # Weather — direct tool call, no LLM needed
        logger.info(f"job {job_id} → fetching weather  location={config.weather_location!r}")
        weather_output = await _fetch_weather(config)
        logger.info(f"job {job_id} → weather done  location={weather_output.location!r}")

        # Only load the model if news or stocks are configured
        need_llm = bool(config.news_subjects or config.stocks)
        model = await _get_model() if need_llm else None

        # News — optional
        news_output: Optional[list[NewsSubjectOutput]] = None
        news_tokens = 0
        if config.news_subjects:
            logger.info(f"job {job_id} → running news  subjects={config.news_subjects}")
            try:
                news_output, news_tokens = await _run_news(model, config)
                logger.info(f"job {job_id} → news done  subjects_with_stories={len(news_output or [])}  tokens={news_tokens}")
            except Exception as e:
                if "connection" in str(e).lower():
                    _model = None
                    logger.warning("LLM connection lost — cleared model cache")
                logger.warning(f"job {job_id} → news failed, omitting  error={e}")

        # Stocks — optional, direct price fetch
        stocks_output: Optional[StocksOutput] = None
        stocks_tokens = 0
        if config.stocks:
            tickers = [s.ticker for s in config.stocks]
            logger.info(f"job {job_id} → fetching stocks  tickers={tickers}")
            try:
                stocks_output, stocks_tokens = await _run_stocks(model, config)
                logger.info(f"job {job_id} → stocks done  tokens={stocks_tokens}")
            except Exception as e:
                logger.warning(f"job {job_id} → stocks failed, omitting  error={e}")

        tokens_used = news_tokens + stocks_tokens

        parsed = BriefingOutput(
            date=date.today().isoformat(),
            weather=weather_output,
            news=news_output or None,
            stocks=stocks_output,
        )

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
