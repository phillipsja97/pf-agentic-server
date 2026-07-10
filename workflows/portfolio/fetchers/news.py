"""Fetch recent news headlines for portfolio ticker symbols via RSS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from effgen.tools.builtin.rss import RSSFeedTool

from config import settings
from core.logging import logger

_YAHOO_FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
_ZERO_RESULT_THRESHOLD = 3
_STATS_PATH = settings.sqlite_path.parent / "portfolio_feed_stats.json"


class Article(TypedDict):
    title: str
    summary: str
    url: str
    published: str


def _load_stats() -> dict[str, int]:
    """Load consecutive zero-result counts keyed by symbol."""
    try:
        if _STATS_PATH.exists():
            return json.loads(_STATS_PATH.read_text())
    except Exception as e:
        logger.warning(f"portfolio feed stats load failed  error={e}")
    return {}


def _save_stats(stats: dict[str, int]) -> None:
    try:
        _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATS_PATH.write_text(json.dumps(stats))
    except Exception as e:
        logger.warning(f"portfolio feed stats save failed  error={e}")


def _update_zero_streak(stats: dict[str, int], symbol: str, got_results: bool) -> None:
    if got_results:
        stats.pop(symbol, None)
    else:
        stats[symbol] = stats.get(symbol, 0) + 1
        count = stats[symbol]
        if count >= _ZERO_RESULT_THRESHOLD:
            logger.warning(
                f"portfolio news feed returned 0 results {count} consecutive times  symbol={symbol}"
                f"  feed={_YAHOO_FINANCE_RSS.format(symbol=symbol)}"
            )


async def fetch_news(symbols: list[str], max_articles: int = 5) -> dict[str, list[Article]]:
    """Fetch news articles for each ticker symbol from Yahoo Finance RSS.

    Returns a dict keyed by symbol. Individual symbol failures are logged
    without crashing the batch. Symbols whose feed has returned zero results
    three or more consecutive times are flagged in the logs.
    """
    stats = _load_stats()
    results: dict[str, list[Article]] = {}
    rss = RSSFeedTool()

    for symbol in symbols:
        feed_url = _YAHOO_FINANCE_RSS.format(symbol=symbol)
        try:
            r = await rss.execute(operation="latest", url=feed_url, n=max_articles)
            if not r.success:
                logger.warning(f"portfolio news RSS failed  symbol={symbol}  error={r.error}")
                _update_zero_streak(stats, symbol, got_results=False)
                results[symbol] = []
                continue

            entries = r.output.get("entries", []) if r.output else []
            articles: list[Article] = []
            for entry in entries:
                title = entry.get("title", "").strip()
                url = entry.get("link") or entry.get("url", "")
                if not title or not url:
                    continue
                articles.append(Article(
                    title=title,
                    summary=(entry.get("summary") or entry.get("description") or title).strip(),
                    url=url,
                    published=entry.get("published", ""),
                ))

            got_results = len(articles) > 0
            _update_zero_streak(stats, symbol, got_results=got_results)
            if not got_results:
                logger.info(f"portfolio news: 0 articles returned  symbol={symbol}  feed={feed_url}")
            else:
                logger.info(f"portfolio news fetched  symbol={symbol}  articles={len(articles)}")
            results[symbol] = articles

        except Exception as e:
            logger.warning(f"portfolio news fetch error  symbol={symbol}  error={e}")
            _update_zero_streak(stats, symbol, got_results=False)
            results[symbol] = []

    _save_stats(stats)
    return results
