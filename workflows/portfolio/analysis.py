"""
LLM-based sentiment analysis and hold/sell/buy recommendation for portfolio symbols.

Each symbol gets a one-paragraph sentiment summary (based on news headlines) and a
concise recommendation with brief rationale. Output maps directly to PortfolioJobResult.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from pydantic import BaseModel

from core.logging import logger
from core.tracing import is_tracing_enabled, observe


class StockAnalysis(BaseModel):
    sentiment_summary: str
    analyst_consensus: str
    recommendation: str


def _build_prompt(
    symbol: str,
    price_data: dict[str, Any],
    news_articles: list[dict[str, Any]],
    analyst_data: Optional[dict[str, Any]],
) -> str:
    price = price_data.get("price", 0.0)
    prev = price_data.get("previous_close", price)
    change = price_data.get("change_pct", 0.0)
    sign = "+" if change >= 0 else ""

    lines = [
        f"Symbol: {symbol}",
        f"Current Price: ${price:.2f} ({sign}{change:.2f}% from previous close ${prev:.2f})",
    ]

    if analyst_data:
        rec = analyst_data.get("recommendation") or "n/a"
        target = analyst_data.get("target_mean_price")
        n = analyst_data.get("num_analysts", 0)
        buys = analyst_data.get("buy_count", 0)
        holds = analyst_data.get("hold_count", 0)
        sells = analyst_data.get("sell_count", 0)
        target_str = f" | Target: ${target:.2f}" if target else ""
        lines.append(
            f"Analyst Data: {buys} buy, {holds} hold, {sells} sell"
            f"{target_str} | Consensus: {rec} ({n} analysts)"
        )
    else:
        lines.append("Analyst Data: none available")

    if news_articles:
        lines.append("Recent News:")
        for i, art in enumerate(news_articles[:5], 1):
            title = art.get("title", "")
            summary = art.get("summary", "")
            blurb = f" — {summary[:120]}" if summary else ""
            lines.append(f"{i}. {title}{blurb}")
    else:
        lines.append("Recent News: none available")

    lines += [
        "",
        "Based only on the data above, produce:",
        "1. sentiment_summary — one paragraph summarising market sentiment from the news and analyst stance.",
        "2. analyst_consensus — one sentence describing the analyst picture (ratings distribution and target).",
        '3. recommendation — "Buy", "Hold", or "Sell" followed by one sentence of rationale.',
        "",
        'Return JSON: {"sentiment_summary": "...", "analyst_consensus": "...", "recommendation": "..."}',
    ]
    return "\n".join(lines)


@observe(name="portfolio-analyze-symbol", as_type="span")
def _analyze_symbol_sync(
    model,
    symbol: str,
    price_data: dict[str, Any],
    news_articles: list[dict[str, Any]],
    analyst_data: Optional[dict[str, Any]],
) -> tuple[Optional[StockAnalysis], int]:
    from effgen import create_agent

    prompt = _build_prompt(symbol, price_data, news_articles, analyst_data)
    agent = create_agent("minimal", model, extra_tools=[], tool_calling_mode="react", max_iterations=5)
    response = agent.run(prompt, output_model=StockAnalysis)

    parsed: Optional[StockAnalysis] = (response.metadata or {}).get("parsed")

    if is_tracing_enabled():
        from langfuse import get_client
        get_client().update_current_span(
            input={"symbol": symbol, "has_news": bool(news_articles), "has_analysts": bool(analyst_data)},
            output=parsed.model_dump() if parsed else {},
            metadata={"tokens_used": response.tokens_used, "iterations": response.iterations, "success": response.success},
        )

    if not parsed:
        logger.warning(f"analysis: LLM returned no parsed output for {symbol}")
        parsed = StockAnalysis(
            sentiment_summary="Insufficient data for sentiment analysis.",
            analyst_consensus="No analyst consensus available.",
            recommendation="Hold — insufficient data to make a recommendation.",
        )

    return parsed, response.tokens_used


async def analyze_portfolio(
    model,
    prices: dict[str, dict[str, Any]],
    news: dict[str, list[dict[str, Any]]],
    analysts: dict[str, Optional[dict[str, Any]]],
) -> tuple[dict[str, StockAnalysis], int]:
    """Analyze each symbol sequentially via LLM. Returns analyses keyed by symbol and total tokens."""
    results: dict[str, StockAnalysis] = {}
    total_tokens = 0

    for symbol, price_data in prices.items():
        logger.info(f"analysis: running LLM for {symbol}")
        try:
            analysis, tokens = await asyncio.to_thread(
                _analyze_symbol_sync,
                model,
                symbol,
                price_data,
                news.get(symbol, []),
                analysts.get(symbol),
            )
            results[symbol] = analysis
            total_tokens += tokens
            logger.info(f"analysis: done  symbol={symbol}  tokens={tokens}")
        except Exception as exc:
            logger.warning(f"analysis: failed for {symbol}: {exc}")
            results[symbol] = StockAnalysis(
                sentiment_summary="Analysis failed.",
                analyst_consensus="N/A",
                recommendation="Hold — analysis could not be completed.",
            )

    return results, total_tokens
