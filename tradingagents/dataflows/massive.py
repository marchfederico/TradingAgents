"""Massive (massive.com, formerly Polygon.io) vendor implementations.

Covers core_stock_apis (aggregates), technical_indicators (native SMA/EMA/MACD/RSI
endpoints, locally computed Bollinger/ATR/VWMA), fundamental_data (ticker overview
and SEC-derived financial statements), and news_data (ticker + global news).
Massive has no insider-transactions endpoint; that method raises a typed
"no data" error so a multi-vendor chain (e.g. ``"massive,yfinance"``) can serve it.
"""

import json
from datetime import datetime, timedelta

import pandas as pd

from .config import get_config
from .massive_common import (
    _make_api_request,
    normalize_massive_symbol,
    paginate_results,
)
from .stockstats_utils import _assert_ohlcv_not_stale
from .symbol_utils import NoMarketDataError


def _aggs_to_dataframe(results: list) -> pd.DataFrame:
    """Convert aggregate bars to an OHLCV frame indexed by calendar date.

    Bar timestamps are Unix ms at the start of the aggregate window in Eastern
    time, so convert through America/New_York before taking the calendar date —
    a plain UTC date would be correct today but breaks if the window convention
    changes, and the exchange-local date is the unambiguous one.
    """
    df = pd.DataFrame(results)
    stamps = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["Date"] = stamps.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    df = df.rename(
        columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    )
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col].round(2)
    return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]


def _fetch_daily_aggs(canonical: str, start_date: str, end_date: str) -> list:
    payload = _make_api_request(
        f"/v2/aggs/ticker/{canonical}/range/1/day/{start_date}/{end_date}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    return paginate_results(payload)


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    """
    Returns split-adjusted daily OHLCV bars from Massive aggregates,
    filtered to the specified date range.

    Args:
        symbol: Ticker symbol, e.g. AAPL (crypto as BTC-USD or X:BTCUSD)
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily OHLCV data for the date range.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    canonical = normalize_massive_symbol(symbol)
    results = _fetch_daily_aggs(canonical, start_date, end_date)

    # Empty result means the symbol is unknown/delisted/uncovered. Raise the
    # typed error so the routing layer emits one unambiguous "no data" signal
    # instead of prose the agent could hallucinate a price around.
    if not results:
        raise NoMarketDataError(
            symbol, canonical, f"no rows between {start_date} and {end_date}"
        )

    data = _aggs_to_dataframe(results)

    # Reject a stale frame before it is formatted into the report; raises
    # NoMarketDataError, which the router turns into one clear signal (#1021).
    _assert_ohlcv_not_stale(data, end_date, symbol, canonical)

    csv_string = data.to_csv()
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Stock data for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# Indicators served by Massive's native /v1/indicators endpoints, with their
# endpoint kind and rolling window. MACD variants share one endpoint and differ
# only in which response field is read.
_NATIVE_INDICATORS = {
    "close_50_sma": ("sma", 50, "value"),
    "close_200_sma": ("sma", 200, "value"),
    "close_10_ema": ("ema", 10, "value"),
    "macd": ("macd", None, "value"),
    "macds": ("macd", None, "signal"),
    "macdh": ("macd", None, "histogram"),
    "rsi": ("rsi", 14, "value"),
}

# Indicators Massive has no endpoint for; computed locally from daily aggregates.
_COMPUTED_INDICATORS = {"boll", "boll_ub", "boll_lb", "atr", "vwma"}

_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.",
    "close_200_sma": "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.",
    "close_10_ema": "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.",
    "macd": "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.",
    "macds": "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.",
    "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.",
    "rsi": "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.",
    "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.",
    "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.",
    "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.",
    "atr": "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.",
    "vwma": "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.",
}


def _native_indicator_series(
    canonical: str, indicator: str, start_date: str, end_date: str
) -> list[tuple[datetime, float]]:
    kind, window, field = _NATIVE_INDICATORS[indicator]
    params = {
        "timespan": "day",
        "adjusted": "true",
        "series_type": "close",
        "order": "asc",
        "limit": 5000,
        "timestamp.gte": start_date,
        "timestamp.lte": end_date,
    }
    if window is not None:
        params["window"] = window
    payload = _make_api_request(f"/v1/indicators/{kind}/{canonical}", params)
    values = (payload.get("results") or {}).get("values") or []
    series = []
    for row in values:
        value = row.get(field)
        if value is None:
            continue
        stamp = pd.to_datetime(row["timestamp"], unit="ms", utc=True)
        date_dt = stamp.tz_convert("America/New_York").tz_localize(None).normalize()
        series.append((date_dt.to_pydatetime(), float(value)))
    return series


def _computed_indicator_series(
    canonical: str, symbol: str, indicator: str, start_date: str, end_date: str
) -> list[tuple[datetime, float]]:
    # Fetch extra history in front of the window so the rolling calculations
    # (20-bar Bollinger, 14-bar ATR/VWMA) are warmed up by the first reported day.
    warmup_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)
    ).strftime("%Y-%m-%d")
    results = _fetch_daily_aggs(canonical, warmup_start, end_date)
    if not results:
        raise NoMarketDataError(
            symbol, canonical, f"no rows between {warmup_start} and {end_date}"
        )
    df = _aggs_to_dataframe(results)

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    if indicator in ("boll", "boll_ub", "boll_lb"):
        mid = close.rolling(20).mean()
        band = 2 * close.rolling(20).std()
        series = {"boll": mid, "boll_ub": mid + band, "boll_lb": mid - band}[indicator]
    elif indicator == "atr":
        prev_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        # Wilder's smoothing, matching the stockstats ATR the yfinance path uses.
        series = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    elif indicator == "vwma":
        series = (close * volume).rolling(14).sum() / volume.rolling(14).sum()
    else:  # pragma: no cover - guarded by the caller's supported-indicator check
        raise ValueError(f"Indicator {indicator} is not supported.")

    series = series.dropna().loc[start_date:end_date]
    return [(stamp.to_pydatetime(), float(value)) for stamp, value in series.items()]


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    """
    Returns Massive technical indicator values over a time window.

    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to get the analysis and report of
        curr_date: The current trading date you are trading on, YYYY-mm-dd
        look_back_days: how many days to look back

    Returns:
        String containing indicator values and description
    """
    supported = set(_NATIVE_INDICATORS) | _COMPUTED_INDICATORS
    if indicator not in supported:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {sorted(supported)}"
        )

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - timedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    canonical = normalize_massive_symbol(symbol)
    if indicator in _NATIVE_INDICATORS:
        series = _native_indicator_series(canonical, indicator, start_date, curr_date)
    else:
        series = _computed_indicator_series(canonical, symbol, indicator, start_date, curr_date)

    ind_string = "".join(
        f"{date_dt.strftime('%Y-%m-%d')}: {value:.4f}\n" for date_dt, value in series
    )
    if not ind_string:
        ind_string = "No data available for the specified date range.\n"

    return (
        f"## {indicator.upper()} values from {start_date} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + _INDICATOR_DESCRIPTIONS.get(indicator, "No description available.")
    )


_OVERVIEW_FIELDS = (
    "name", "ticker", "market", "locale", "primary_exchange", "type",
    "currency_name", "market_cap", "share_class_shares_outstanding",
    "weighted_shares_outstanding", "total_employees", "list_date",
    "sic_code", "sic_description", "homepage_url", "description",
)


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve company overview fundamentals for a given ticker using Massive.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (used to
            date the reference-data request for historical consistency)

    Returns:
        str: Company overview data including identity, market cap, and shares
    """
    canonical = normalize_massive_symbol(ticker)
    params = {"date": curr_date} if curr_date else {}
    payload = _make_api_request(f"/v3/reference/tickers/{canonical}", params)
    details = payload.get("results")
    if not details:
        raise NoMarketDataError(ticker, canonical, "no ticker details on Massive")

    overview = {key: details[key] for key in _OVERVIEW_FIELDS if key in details}
    return json.dumps(overview, indent=2, default=str)


def _get_financial_statement(
    ticker: str, freq: str, curr_date: str, statement_key: str
) -> str:
    canonical = normalize_massive_symbol(ticker)
    timeframe = "quarterly" if str(freq).lower().startswith("q") else "annual"
    params = {
        "ticker": canonical,
        "timeframe": timeframe,
        "order": "desc",
        "sort": "period_of_report_date",
        "limit": 8,
    }
    if curr_date:
        params["period_of_report_date.lte"] = curr_date

    payload = _make_api_request("/vX/reference/financials", params)
    results = payload.get("results") or []

    reports = []
    for filing in results:
        # Look-ahead guard: a statement filed after curr_date was not public on
        # the simulated trading day even if its fiscal period had ended.
        filing_date = filing.get("filing_date")
        if curr_date and filing_date and filing_date > curr_date:
            continue
        statement = (filing.get("financials") or {}).get(statement_key) or {}
        if not statement:
            continue
        reports.append({
            "fiscal_year": filing.get("fiscal_year"),
            "fiscal_period": filing.get("fiscal_period"),
            "start_date": filing.get("start_date"),
            "end_date": filing.get("end_date"),
            "filing_date": filing_date,
            "items": {
                info.get("label", field): info.get("value")
                for field, info in statement.items()
            },
        })

    if not reports:
        raise NoMarketDataError(
            ticker, canonical, f"no {timeframe} {statement_key} filings on Massive"
        )

    return json.dumps(
        {
            "symbol": ticker,
            "statement": statement_key,
            "timeframe": timeframe,
            "reports": reports,
        },
        indent=2,
        default=str,
    )


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Massive."""
    return _get_financial_statement(ticker, freq, curr_date, "balance_sheet")


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Massive."""
    return _get_financial_statement(ticker, freq, curr_date, "cash_flow_statement")


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Massive."""
    return _get_financial_statement(ticker, freq, curr_date, "income_statement")


def _format_news_articles(articles: list, focus_ticker: str | None = None) -> str:
    news_str = ""
    for article in articles:
        publisher = (article.get("publisher") or {}).get("name", "Unknown")
        news_str += f"### {article.get('title', 'No title')} (source: {publisher})\n"
        if article.get("published_utc"):
            news_str += f"Published: {article['published_utc']}\n"
        if article.get("description"):
            news_str += f"{article['description']}\n"
        # Massive attaches per-ticker sentiment insights; surface the one for
        # the requested symbol so the analyst gets signal, not just headlines.
        if focus_ticker:
            for insight in article.get("insights") or []:
                if insight.get("ticker") == focus_ticker:
                    sentiment = insight.get("sentiment", "unknown")
                    reasoning = insight.get("sentiment_reasoning", "")
                    news_str += f"Sentiment ({focus_ticker}): {sentiment} — {reasoning}\n"
        if article.get("article_url"):
            news_str += f"Link: {article['article_url']}\n"
        news_str += "\n"
    return news_str


def _fetch_news(params: dict, limit: int) -> list:
    params = {
        **params,
        "order": "desc",
        "sort": "published_utc",
        "limit": min(limit, 1000),
    }
    # A single page covers the request: limit is capped at the API's page max.
    payload = _make_api_request("/v2/reference/news", params)
    return (payload.get("results") or [])[:limit]


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """
    Retrieve news for a specific stock ticker using Massive.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    article_limit = get_config()["news_article_limit"]
    canonical = normalize_massive_symbol(ticker)
    resolved = "" if canonical == ticker.upper() else f" (resolved to {canonical})"

    # published_utc.lte at a bare date means midnight, which would drop the end
    # day's articles; use an exclusive bound at the following midnight instead.
    end_exclusive = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    articles = _fetch_news(
        {
            "ticker": canonical,
            "published_utc.gte": start_date,
            "published_utc.lt": end_exclusive,
        },
        article_limit,
    )
    if not articles:
        return f"No news found for {ticker}{resolved} between {start_date} and {end_date}"

    news_str = _format_news_articles(articles, focus_ticker=canonical)
    return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{news_str}"


def get_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """
    Retrieve global market news using Massive (no ticker filter).

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back. ``None`` falls back to
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles to return. ``None`` falls back to
            ``global_news_article_limit`` from the active config.

    Returns:
        Formatted string containing global news articles
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    end_exclusive = (curr_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    articles = _fetch_news(
        {"published_utc.gte": start_date, "published_utc.lt": end_exclusive},
        limit,
    )
    if not articles:
        return f"No global news found between {start_date} and {curr_date}"

    news_str = _format_news_articles(articles)
    return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"


def get_insider_transactions(symbol: str) -> str:
    """Massive does not provide insider transaction data.

    Raises the typed "no data" error so a multi-vendor chain (e.g.
    ``news_data: "massive,yfinance"``) falls through to a vendor that has it,
    and a massive-only configuration degrades to the router's explicit
    NO_DATA sentinel instead of crashing the news analyst.
    """
    raise NoMarketDataError(
        symbol,
        detail=(
            "Massive (massive.com) has no insider-transactions endpoint; "
            "configure alpha_vantage or yfinance for this tool, e.g. "
            'news_data: "massive,yfinance" or tool_vendors: '
            '{"get_insider_transactions": "yfinance"}'
        ),
    )
