"""Massive (massive.com, formerly Polygon.io) vendor: HTTP handling, symbol
normalization, look-ahead guards, and router registration."""

import json
from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import interface, massive
from tradingagents.dataflows.massive_common import (
    MassiveNotConfiguredError,
    MassiveRateLimitError,
    normalize_massive_symbol,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _response(payload, status_code=200, text=""):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.text = text or json.dumps(payload)
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _et_epoch_ms(date_str: str) -> int:
    """Unix ms for midnight Eastern on the given date (aggregate bar convention)."""
    return int(pd.Timestamp(date_str, tz="America/New_York").value // 10**6)


def _aggs_payload(dates, close=100.0):
    return {
        "status": "OK",
        "results": [
            {
                "t": _et_epoch_ms(d),
                "o": close - 1,
                "h": close + 1,
                "l": close - 2,
                "c": close,
                "v": 1_000_000,
            }
            for d in dates
        ],
    }


@pytest.mark.unit
class TestMassiveCommon:
    def test_missing_api_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with pytest.raises(MassiveNotConfiguredError):
            massive.get_stock("AAPL", "2026-01-05", "2026-01-09")

    def test_polygon_key_accepted_for_continuity(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.setenv("POLYGON_API_KEY", "legacy-key")
        payload = _aggs_payload(["2026-01-05"])
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ) as get:
            massive.get_stock("AAPL", "2026-01-05", "2026-01-09")
        headers = get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer legacy-key"

    def test_http_429_raises_rate_limit(self):
        resp = _response({}, status_code=429, text="too many requests")
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get", return_value=resp
        ), pytest.raises(MassiveRateLimitError):
            massive.get_stock("AAPL", "2026-01-05", "2026-01-09")

    def test_http_403_raises_not_configured(self):
        resp = _response({}, status_code=403, text="unauthorized")
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get", return_value=resp
        ), pytest.raises(MassiveNotConfiguredError):
            massive.get_stock("AAPL", "2026-01-05", "2026-01-09")

    def test_body_level_not_authorized_raises_not_configured(self):
        resp = _response({"status": "NOT_AUTHORIZED", "message": "upgrade plan"})
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get", return_value=resp
        ), pytest.raises(MassiveNotConfiguredError):
            massive.get_stock("AAPL", "2026-01-05", "2026-01-09")

    def test_symbol_normalization(self):
        assert normalize_massive_symbol("aapl") == "AAPL"
        assert normalize_massive_symbol("BTC-USD") == "X:BTCUSD"
        assert normalize_massive_symbol("EURUSD=X") == "C:EURUSD"
        assert normalize_massive_symbol("X:ETHUSD") == "X:ETHUSD"
        assert normalize_massive_symbol("BRK-B") == "BRK-B"  # share class, not a pair


@pytest.mark.unit
class TestMassiveStock:
    def test_stock_data_formats_csv(self):
        payload = _aggs_payload(["2026-01-05", "2026-01-06", "2026-01-07"])
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ):
            result = massive.get_stock("AAPL", "2026-01-05", "2026-01-09")
        assert "# Stock data for AAPL from 2026-01-05 to 2026-01-09" in result
        assert "# Total records: 3" in result
        assert "2026-01-06" in result
        assert "Open,High,Low,Close,Volume" in result

    def test_empty_results_raise_no_market_data(self):
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response({"status": "OK", "results": []}),
        ), pytest.raises(NoMarketDataError):
            massive.get_stock("FAKESYM", "2026-01-05", "2026-01-09")

    def test_stale_frame_raises_no_market_data(self):
        # Latest bar ~11 months before the requested end date -> stale guard trips.
        payload = _aggs_payload(["2025-02-03", "2025-02-04"])
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ), pytest.raises(NoMarketDataError) as exc_info:
            massive.get_stock("AAPL", "2025-02-01", "2026-01-09")
        assert "stale" in str(exc_info.value)


@pytest.mark.unit
class TestMassiveIndicators:
    def test_native_sma_formats_window(self):
        values = [
            {"timestamp": _et_epoch_ms("2026-01-06"), "value": 101.2345},
            {"timestamp": _et_epoch_ms("2026-01-07"), "value": 102.5},
        ]
        payload = {"status": "OK", "results": {"values": values}}
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ) as get:
            result = massive.get_indicator("AAPL", "close_50_sma", "2026-01-09", 10)
        assert "## CLOSE_50_SMA values" in result
        assert "2026-01-06: 101.2345" in result
        assert "50 SMA" in result  # description appended
        assert "/v1/indicators/sma/AAPL" in get.call_args.args[0]
        assert get.call_args.kwargs["params"]["window"] == 50

    def test_computed_bollinger_from_aggregates(self):
        dates = pd.bdate_range("2025-11-01", "2026-01-09").strftime("%Y-%m-%d")
        payload = _aggs_payload(list(dates))
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ):
            result = massive.get_indicator("AAPL", "boll", "2026-01-09", 5)
        assert "## BOLL values" in result
        # Constant close of 100.0 -> 20-day SMA is exactly 100.
        assert "2026-01-09: 100.0000" in result

    def test_unsupported_indicator_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            massive.get_indicator("AAPL", "bogus_indicator", "2026-01-09", 10)


@pytest.mark.unit
class TestMassiveFundamentals:
    def test_income_statement_drops_filings_after_curr_date(self):
        payload = {
            "status": "OK",
            "results": [
                {  # filed AFTER curr_date -> must be excluded (look-ahead)
                    "fiscal_year": "2026", "fiscal_period": "Q1",
                    "start_date": "2025-10-01", "end_date": "2025-12-31",
                    "filing_date": "2026-02-01",
                    "financials": {"income_statement": {
                        "revenues": {"label": "Revenues", "value": 999.0},
                    }},
                },
                {
                    "fiscal_year": "2025", "fiscal_period": "Q3",
                    "start_date": "2025-07-01", "end_date": "2025-09-30",
                    "filing_date": "2025-11-01",
                    "financials": {"income_statement": {
                        "revenues": {"label": "Revenues", "value": 123.0},
                    }},
                },
            ],
        }
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ):
            result = massive.get_income_statement("AAPL", "quarterly", "2026-01-09")
        parsed = json.loads(result)
        assert len(parsed["reports"]) == 1
        assert parsed["reports"][0]["filing_date"] == "2025-11-01"
        assert parsed["reports"][0]["items"]["Revenues"] == 123.0

    def test_no_filings_raises_no_market_data(self):
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response({"status": "OK", "results": []}),
        ), pytest.raises(NoMarketDataError):
            massive.get_balance_sheet("FAKESYM", "quarterly", "2026-01-09")

    def test_fundamentals_overview_fields(self):
        payload = {
            "status": "OK",
            "results": {
                "ticker": "AAPL", "name": "Apple Inc.", "market_cap": 3.5e12,
                "primary_exchange": "XNAS", "total_employees": 160000,
                "cik": "0000320193",  # not in the overview whitelist
            },
        }
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ):
            result = massive.get_fundamentals("AAPL", "2026-01-09")
        parsed = json.loads(result)
        assert parsed["name"] == "Apple Inc."
        assert "cik" not in parsed


@pytest.mark.unit
class TestMassiveNews:
    def test_news_formats_articles_with_sentiment(self):
        payload = {
            "status": "OK",
            "results": [{
                "title": "Apple beats estimates",
                "published_utc": "2026-01-06T14:00:00Z",
                "description": "Strong quarter.",
                "article_url": "https://example.com/a",
                "publisher": {"name": "Newswire"},
                "insights": [{
                    "ticker": "AAPL", "sentiment": "positive",
                    "sentiment_reasoning": "Earnings beat.",
                }],
            }],
        }
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response(payload),
        ) as get:
            result = massive.get_news("AAPL", "2026-01-05", "2026-01-09")
        assert "### Apple beats estimates (source: Newswire)" in result
        assert "Sentiment (AAPL): positive" in result
        # End bound is exclusive at the following midnight so the end day's
        # articles are included.
        params = get.call_args.kwargs["params"]
        assert params["published_utc.lt"] == "2026-01-10"

    def test_news_empty_returns_no_news_message(self):
        with mock.patch(
            "tradingagents.dataflows.massive_common.requests.get",
            return_value=_response({"status": "OK", "results": []}),
        ):
            result = massive.get_news("AAPL", "2026-01-05", "2026-01-09")
        assert "No news found for AAPL" in result

    def test_insider_transactions_raise_no_market_data(self):
        with pytest.raises(NoMarketDataError, match="insider"):
            massive.get_insider_transactions("AAPL")


@pytest.mark.unit
class TestMassiveRouting:
    def test_massive_registered_for_expected_methods(self):
        assert "massive" in interface.VENDOR_LIST
        for method in (
            "get_stock_data", "get_indicators", "get_fundamentals",
            "get_balance_sheet", "get_cashflow", "get_income_statement",
            "get_news", "get_global_news", "get_insider_transactions",
        ):
            assert "massive" in interface.VENDOR_METHODS[method], method

    def test_insider_chain_falls_through_to_yfinance(self):
        # news_data "massive,yfinance": massive has no insider data, so the
        # router must fall through to yfinance instead of crashing or lying.
        import copy

        import tradingagents.dataflows.config as config_module
        import tradingagents.default_config as default_config

        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
        try:
            config_module.set_config(
                {"data_vendors": {"news_data": "massive,yfinance"}}
            )
            with mock.patch.dict(
                interface.VENDOR_METHODS,
                {"get_insider_transactions": {
                    "massive": massive.get_insider_transactions,
                    "yfinance": lambda ticker: "YF_INSIDER_DATA",
                }},
                clear=False,
            ):
                result = interface.route_to_vendor("get_insider_transactions", "AAPL")
            assert result == "YF_INSIDER_DATA"
        finally:
            config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
