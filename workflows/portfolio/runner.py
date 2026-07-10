"""
Portfolio deep-dive workflow runner.

Fetches prices, news, and analyst data in parallel, then runs LLM analysis
per symbol to produce sentiment summaries and buy/hold/sell recommendations.
Writes a structured Markdown note to the Obsidian brain vault.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional

from config import settings
from core.logging import logger
from core.storage.jobs import update_job
from core.storage.obsidian import write_brain_note
from core.tracing import observe
from schemas.models import PortfolioJobResult, PortfolioRequest


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


def _format_portfolio_note(results: list[PortfolioJobResult], today: str, tickers: list[str]) -> str:
    lines = [f"# Portfolio Deep Dive — {today}", ""]
    lines += [
        "## Summary",
        "",
        "| Symbol | Price | Change | Recommendation |",
        "|--------|-------|--------|----------------|",
    ]
    for r in results:
        sign = "+" if r.day_change_pct >= 0 else ""
        lines.append(
            f"| {r.symbol} | ${r.current_price:.2f} | {sign}{r.day_change_pct:.2f}% | {r.recommendation.split('—')[0].strip()} |"
        )
    lines.append("")

    for r in results:
        sign = "+" if r.day_change_pct >= 0 else ""
        lines += [
            f"## {r.symbol}",
            "",
            f"**Price:** ${r.current_price:.2f} ({sign}{r.day_change_pct:.2f}%)",
            "",
            f"**Analyst Consensus:** {r.analyst_consensus}",
            "",
            f"**Sentiment:** {r.sentiment_summary}",
            "",
            f"**Recommendation:** {r.recommendation}",
            "",
        ]

    missing = [t for t in tickers if t not in {r.symbol for r in results}]
    if missing:
        lines += ["## Not Available", "", f"No data returned for: {', '.join(missing)}", ""]

    return "\n".join(lines)


@observe(name="portfolio-workflow")
async def run_portfolio(job_id: str, request: PortfolioRequest) -> None:
    global _model

    logger.info(f"job {job_id} → running  workflow=portfolio")
    await update_job(job_id, "running")

    try:
        tickers = (
            [t.strip().upper() for t in request.tickers if t.strip()]
            if request.tickers
            else [t.strip().upper() for t in settings.portfolio_tickers.split(",") if t.strip()]
        )
        logger.info(f"job {job_id} → tickers={tickers}")

        from workflows.portfolio.fetchers.prices import fetch_prices
        from workflows.portfolio.fetchers.news import fetch_news
        from workflows.portfolio.fetchers.analysts import fetch_analyst_data

        logger.info(f"job {job_id} → fetching prices, news, and analyst data in parallel")
        prices, news, analysts = await asyncio.gather(
            asyncio.to_thread(fetch_prices, tickers),
            fetch_news(tickers),
            asyncio.to_thread(fetch_analyst_data, tickers),
        )
        logger.info(
            f"job {job_id} → data fetched"
            f"  prices={len(prices)}  news_symbols={len(news)}  analysts={len(analysts)}"
        )

        if not prices:
            raise RuntimeError("No price data returned for any ticker")

        model = await _get_model()

        from workflows.portfolio.analysis import analyze_portfolio
        logger.info(f"job {job_id} → running LLM analysis  symbols={list(prices)}")
        analyses, tokens_used = await analyze_portfolio(model, prices, news, analysts)
        logger.info(f"job {job_id} → analysis done  tokens={tokens_used}")

        results: list[PortfolioJobResult] = []
        for symbol, price_data in prices.items():
            analysis = analyses.get(symbol)
            if analysis:
                results.append(PortfolioJobResult(
                    symbol=symbol,
                    current_price=price_data["price"],
                    day_change_pct=price_data["change_pct"],
                    sentiment_summary=analysis.sentiment_summary,
                    analyst_consensus=analysis.analyst_consensus,
                    recommendation=analysis.recommendation,
                ))

        today = date.today().isoformat()
        note_content = _format_portfolio_note(results, today, tickers)
        note_path = write_brain_note(
            "Portfolio",
            f"portfolio-{today}",
            note_content,
            frontmatter={"date": f'"{today}"', "type": "portfolio-analysis"},
        )

        result_data = {
            "results": [r.model_dump() for r in results],
            "tokens_used": tokens_used,
            "obsidian_note": str(note_path),
            "date": today,
        }
        await update_job(job_id, "completed", result=result_data)
        logger.info(f"job {job_id} → completed  workflow=portfolio  symbols={len(results)}  tokens={tokens_used}")

    except Exception as exc:
        logger.error(f"job {job_id} → failed  error={exc}")
        if "connection" in str(exc).lower():
            _model = None
            logger.warning("LLM connection lost — cleared model cache")
        await update_job(job_id, "failed", error=str(exc))
