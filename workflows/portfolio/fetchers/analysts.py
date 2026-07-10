"""Fetch analyst consensus data for a list of ticker symbols via yfinance."""

from typing import Optional

import yfinance as yf

from core.logging import logger


def _safe_int(value) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def fetch_analyst_data(symbols: list[str]) -> dict[str, Optional[dict]]:
    """Return analyst consensus data keyed by symbol.

    For each symbol the returned dict contains:
        recommendation  – e.g. "buy", "hold", "sell" (from yfinance info)
        target_mean_price – analyst mean price target (float or None)
        num_analysts    – number of analyst opinions
        buy_count       – strongBuy + buy ratings (most-recent period)
        hold_count      – hold ratings
        sell_count      – underperform + sell ratings

    Symbols with no analyst coverage (e.g. mutual funds, ETFs) get None.
    """
    results: dict[str, Optional[dict]] = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            recommendation = info.get("recommendationKey") or info.get("recommendation")
            target_mean_price = info.get("targetMeanPrice")
            num_analysts = _safe_int(info.get("numberOfAnalystOpinions"))

            # Pull buy/hold/sell counts from recommendations_summary if available.
            buy_count = hold_count = sell_count = 0
            try:
                summary = ticker.recommendations_summary
                if summary is not None and not summary.empty:
                    # Use most recent period (row 0).
                    row = summary.iloc[0]
                    buy_count = _safe_int(row.get("strongBuy", 0)) + _safe_int(row.get("buy", 0))
                    hold_count = _safe_int(row.get("hold", 0))
                    sell_count = _safe_int(row.get("underperform", 0)) + _safe_int(row.get("sell", 0))
            except Exception:
                pass  # counts stay 0; still return the info fields below

            # Treat symbols where yfinance returns an empty info dict as no-data.
            if not recommendation and target_mean_price is None and num_analysts == 0:
                logger.debug(f"analysts: no coverage for {symbol}")
                results[symbol] = None
                continue

            results[symbol] = {
                "recommendation": recommendation,
                "target_mean_price": float(target_mean_price) if target_mean_price is not None else None,
                "num_analysts": num_analysts,
                "buy_count": buy_count,
                "hold_count": hold_count,
                "sell_count": sell_count,
            }

        except Exception as exc:
            logger.warning(f"analysts: failed to fetch data for {symbol}: {exc}")
            results[symbol] = None

    return results
