"""
Direct stock price fetcher for the portfolio workflow.

Fetches current price, previous close, and day-change % for a list of
ticker symbols. Uses yfinance as the primary source; falls back to the
Yahoo Finance public chart API if yfinance returns no data. No LLM or
API key required.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import yfinance as yf

from core.logging import logger


def _fetch_yahoo_api(symbol: str) -> dict[str, Any]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
        "?interval=1d&range=1d"
    )
    req = Request(url, headers={"User-Agent": "agentic-server/1.0", "Accept": "application/json"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    chart = (data or {}).get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo API error for '{symbol}': {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"No chart data returned for '{symbol}'")

    meta = results[0].get("meta", {})
    price = meta.get("regularMarketPrice")
    if price is None:
        raise ValueError(f"No price in chart response for '{symbol}'")
    prev = meta.get("chartPreviousClose") or price
    change_pct = round((price - prev) / prev * 100, 2) if prev else 0.0
    return {
        "symbol": symbol,
        "price": float(price),
        "previous_close": float(prev),
        "change_pct": change_pct,
    }


def fetch_prices(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Return price data for each symbol, keyed by uppercase ticker.

    Each value contains:
        price           – current/last market price (float)
        previous_close  – prior session's close (float)
        change_pct      – day change as a percentage (float, rounded to 2 dp)

    Symbols that are missing, delisted, or otherwise unreachable are omitted
    from the result (a warning is logged for each).
    """
    results: dict[str, dict[str, Any]] = {}

    for raw_symbol in symbols:
        symbol = raw_symbol.strip().upper()
        try:
            ticker = yf.Ticker(symbol)
            info = getattr(ticker, "fast_info", None) or {}

            price: float | None = None
            prev: float | None = None

            if info:
                price = info.get("last_price") or info.get("lastPrice")
                prev = info.get("previous_close") or info.get("previousClose")

            if price is None:
                hist = ticker.history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    if len(hist) >= 2:
                        prev = float(hist["Close"].iloc[-2])

            if price is None:
                # yfinance returned nothing — try the public chart API
                logger.debug(f"prices: yfinance returned no data for {symbol}, trying chart API")
                results[symbol] = _fetch_yahoo_api(symbol)
                continue

            if prev is None:
                prev = price
            change_pct = round((price - prev) / prev * 100, 2) if prev else 0.0
            results[symbol] = {
                "symbol": symbol,
                "price": float(price),
                "previous_close": float(prev),
                "change_pct": change_pct,
            }

        except Exception as exc:
            logger.warning(f"prices: failed to fetch data for {symbol}: {exc}")

    return results
