"""
External API Integration Layer
--------------------------------
Fetches live market data (NSE quotes by default) through a small, timeout-
bounded requests wrapper. Base URLs, timeouts, and any future credentials
are all read from environment variables (see .env.example) — nothing is
hardcoded, so the same code targets whatever endpoint the deployment
environment configures.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

logger = logging.getLogger("mo_insight.external")

NSE_BASE_URL = os.getenv("NSE_API_BASE_URL", "https://www.nseindia.com/api")
EXTERNAL_API_TIMEOUT = float(os.getenv("EXTERNAL_API_TIMEOUT", "5"))

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (MO-Insight local copilot)"})


def get_quote(symbol: str) -> dict:
    """Fetch a live quote for an NSE-listed equity symbol.

    Returns a dict that is always citation-ready, even on failure, so the
    router can still attach `[Source: API::NSE | Timestamp: ...]`.
    """
    symbol = symbol.upper().strip()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        # NSE's API requires a warm-up hit against the homepage to obtain
        # session cookies before the JSON endpoint will respond.
        _session.get("https://www.nseindia.com", timeout=EXTERNAL_API_TIMEOUT)
        resp = _session.get(
            f"{NSE_BASE_URL}/quote-equity",
            params={"symbol": symbol},
            timeout=EXTERNAL_API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        price_info = data.get("priceInfo", {})
        return {
            "symbol": symbol,
            "last_price": price_info.get("lastPrice"),
            "change": price_info.get("change"),
            "pct_change": price_info.get("pChange"),
            "timestamp_utc": timestamp,
            "source": "API::NSE",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("External quote fetch failed for %s: %s", symbol, exc)
        return {
            "symbol": symbol,
            "error": str(exc),
            "timestamp_utc": timestamp,
            "source": "API::NSE",
        }
