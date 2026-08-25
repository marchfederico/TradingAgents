"""Shared plumbing for the Massive (massive.com, formerly Polygon.io) vendor.

Massive's REST API keeps the Polygon.io endpoint shapes (/v2/aggs, /v2/reference/news,
/vX/reference/financials, /v1/indicators/...) under the rebranded host. Keys issued
under either brand keep working, so both MASSIVE_API_KEY and POLYGON_API_KEY are
honored, and the base URL can be pointed back at api.polygon.io via env override.
"""

import os

import requests

from .errors import VendorNotConfiguredError, VendorRateLimitError


def get_base_url() -> str:
    """Resolve the API host at call time so env overrides work after import."""
    return os.getenv("MASSIVE_API_BASE_URL", "https://api.massive.com").rstrip("/")


# Network timeout (seconds) so a stalled Massive request can't hang the
# CLI/agents indefinitely (same policy as the other HTTP vendors, #990).
REQUEST_TIMEOUT = 30


class MassiveNotConfiguredError(VendorNotConfiguredError):
    """Raised when Massive is selected but no API key is configured.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer's "vendor unavailable" handling and existing ValueError callers both
    keep working.
    """


class MassiveRateLimitError(VendorRateLimitError):
    """Raised when the Massive API rate limit is exceeded."""


def get_api_key() -> str:
    """Retrieve the API key for Massive from environment variables."""
    api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise MassiveNotConfiguredError(
            "MASSIVE_API_KEY environment variable is not set "
            "(POLYGON_API_KEY is accepted for keys issued before the rebrand)."
        )
    return api_key


def _classify_http_error(response: requests.Response) -> None:
    if response.status_code == 429:
        raise MassiveRateLimitError(
            f"Massive rate limit exceeded: {response.text[:200]}"
        )
    if response.status_code in (401, 403):
        raise MassiveNotConfiguredError(
            f"Massive API key invalid or not entitled for this endpoint "
            f"(HTTP {response.status_code}): {response.text[:200]}"
        )
    response.raise_for_status()


def _make_api_request(path: str, params: dict | None = None) -> dict:
    """GET a Massive REST endpoint and return the parsed JSON payload.

    Raises:
        MassiveRateLimitError: HTTP 429 (plan request-rate exceeded).
        MassiveNotConfiguredError: missing/invalid key or unentitled endpoint.
    """
    response = requests.get(
        f"{get_base_url()}{path}",
        params=params or {},
        headers={"Authorization": f"Bearer {get_api_key()}"},
        timeout=REQUEST_TIMEOUT,
    )
    _classify_http_error(response)
    payload = response.json()
    # Body-level errors come back HTTP 200 with status ERROR/NOT_AUTHORIZED.
    status = str(payload.get("status", "")).upper()
    if status == "NOT_AUTHORIZED":
        raise MassiveNotConfiguredError(
            f"Massive plan not entitled for {path}: {payload.get('message') or payload.get('error')}"
        )
    if status == "ERROR":
        raise ValueError(
            f"Massive API error for {path}: {payload.get('message') or payload.get('error')}"
        )
    return payload


def paginate_results(payload: dict, max_pages: int = 5) -> list:
    """Collect ``results`` across ``next_url`` pages (Massive's cursor pagination).

    ``max_pages`` bounds follow-up requests so one tool call can't burn a whole
    request-rate budget; partial results are returned rather than discarded.
    """
    results = list(payload.get("results") or [])
    next_url = payload.get("next_url")
    pages_followed = 0
    while next_url and pages_followed < max_pages:
        response = requests.get(
            next_url,
            headers={"Authorization": f"Bearer {get_api_key()}"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            break  # keep the pages already fetched instead of failing the call
        _classify_http_error(response)
        payload = response.json()
        results.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        pages_followed += 1
    return results


def normalize_massive_symbol(symbol: str) -> str:
    """Map user/Yahoo-style tickers to Massive's symbol conventions.

    Massive prefixes asset classes: crypto pairs are ``X:BTCUSD``, spot forex is
    ``C:EURUSD``; US equities/ETFs are the plain ticker. Yahoo-style crypto
    (``BTC-USD``) and forex (``EURUSD=X``) spellings therefore need rewriting or
    they return empty results, which the router would report as "no data".
    """
    sym = symbol.strip().upper()
    if sym.startswith(("X:", "C:", "O:", "I:")):
        return sym  # already in Massive notation
    if sym.endswith("=X") and len(sym) == 8:
        return f"C:{sym[:-2]}"
    if "-" in sym:
        base, _, quote = sym.partition("-")
        if base.isalpha() and quote.isalpha() and len(quote) == 3:
            return f"X:{base}{quote}"
    return sym
