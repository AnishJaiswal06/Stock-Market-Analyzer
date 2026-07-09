"""
data_pipeline.py — Data acquisition and alignment layer for the Stock Analyzer.

Responsibilities:
  • Fetch OHLCV data via yfinance
  • Fetch fundamental data (Revenue, Net Income, EPS) via SEC EDGAR XBRL API
  • Align and merge daily price data with quarterly fundamentals

All public functions carry PEP 484 type hints and return Optional types where
appropriate so callers can distinguish "no data" from errors.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import date
from functools import wraps
import requests
import time
import logging
import yfinance

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CIK cache  —  avoids repeated lookups against the SEC ticker-to-CIK file
# ---------------------------------------------------------------------------
_CIK_CACHE: Dict[str, str] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Retry decorator
# ═══════════════════════════════════════════════════════════════════════════

def retry(max_attempts: int = 3, backoff_factor: float = 0.5):
    """
    Decorator that retries a function on ``requests.exceptions.RequestException``
    with exponential backoff.

    Args:
        max_attempts:   Maximum number of invocations before giving up.
        backoff_factor: Multiplier applied to ``2 ** attempt`` to compute the
                        sleep duration between retries.

    Returns:
        The decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt + 1 == max_attempts:
                        logger.error(
                            "API call failed permanently after %d attempts: %s",
                            max_attempts,
                            e,
                        )
                        raise
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        "Attempt %d failed with %s. Retrying in %.2f seconds...",
                        attempt + 1,
                        type(e).__name__,
                        wait_time,
                    )
                    time.sleep(wait_time)
            return None  # pragma: no cover — unreachable
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# fetch_ohlcv
# ═══════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(
    ticker: str,
    start_date: date,
    end_date: date,
    max_retries: int = 5,
    base_wait: float = 5.0,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical Open / High / Low / Close / Volume data via *yfinance*.

    Includes built-in retry logic with exponential back-off specifically
    designed to handle Yahoo Finance's aggressive rate-limiting (HTTP 429)
    which yfinance surfaces as timeout errors.

    Args:
        ticker:     Stock ticker symbol (e.g. ``'AAPL'``).
        start_date: First calendar date (inclusive) of the window.
        end_date:   Last calendar date (exclusive — yfinance convention).
        max_retries: Maximum number of download attempts.
        base_wait:  Initial wait time (seconds) between retries; doubles
                    each attempt.

    Returns:
        A ``pd.DataFrame`` with a ``DatetimeIndex`` named ``'Date'`` and
        columns ``[Open, High, Low, Close, Volume]``, or ``None`` if no
        data could be retrieved.
    """
    logger.info(
        "Fetching OHLCV data for %s from %s to %s.", ticker, start_date, end_date
    )

    for attempt in range(1, max_retries + 1):
        try:
            # Small pre-download pause to reduce chance of rate-limiting
            if attempt > 1:
                wait = base_wait * (2 ** (attempt - 2))
                logger.info(
                    "Rate-limit cooldown: waiting %.1f s before retry %d/%d...",
                    wait, attempt, max_retries,
                )
                time.sleep(wait)

            data: pd.DataFrame = yfinance.download(
                ticker,
                start=str(start_date),
                end=str(end_date),
                progress=False,
                timeout=30,
            )

            if data.empty:
                logger.warning(
                    "No data returned for %s (attempt %d/%d).",
                    ticker, attempt, max_retries,
                )
                if attempt < max_retries:
                    continue   # retry — may be a transient 429
                return None

            # yfinance ≥ 0.2.31 may return multi-level columns when a single
            # ticker is requested.  Flatten them if necessary.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Keep only the canonical OHLCV columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                logger.error("Missing expected columns after download: %s", missing)
                return None

            data = data[required_cols].copy()

            # Ensure the index is a proper DatetimeIndex named 'Date'
            data.index = pd.to_datetime(data.index)
            data.index.name = "Date"

            logger.info(
                "Successfully retrieved OHLCV data — %d rows, %s → %s.",
                len(data),
                data.index.min().date(),
                data.index.max().date(),
            )
            return data

        except Exception as e:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt, max_retries, ticker, e,
            )
            if attempt == max_retries:
                logger.error(
                    "All %d download attempts failed for %s. "
                    "This is usually caused by Yahoo Finance rate-limiting (HTTP 429). "
                    "Try again in a few minutes.",
                    max_retries, ticker,
                )
                return None

    return None  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════════
# fetch_fundamentals  (SEC EDGAR XBRL — free, no API key)
# ═══════════════════════════════════════════════════════════════════════════

def _build_headers(user_agent_email: str) -> Dict[str, str]:
    """Return the ``User-Agent`` header required by the SEC EDGAR API."""
    return {"User-Agent": f"StockAnalyzer {user_agent_email}"}


def _resolve_cik(ticker: str, headers: Dict[str, str]) -> Optional[str]:
    """
    Map *ticker* → CIK (Central Index Key) using the SEC company-tickers
    JSON file.  Results are cached in ``_CIK_CACHE``.

    Returns:
        10-digit zero-padded CIK string, or ``None`` on failure.
    """
    ticker_upper = ticker.upper()

    if ticker_upper in _CIK_CACHE:
        return _CIK_CACHE[ticker_upper]

    url = "https://www.sec.gov/files/company_tickers.json"
    logger.info("Resolving CIK for %s via %s", ticker_upper, url)

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tickers_json: Dict[str, Dict[str, Any]] = resp.json()

        for entry in tickers_json.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                cik_padded = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker_upper] = cik_padded
                logger.info("Resolved %s → CIK %s", ticker_upper, cik_padded)
                return cik_padded

        logger.warning(
            "Ticker %s not found in SEC company_tickers.json — "
            "it may be an ETF or delisted security.",
            ticker_upper,
        )
        return None

    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch SEC company tickers: %s", e)
        return None


def _extract_metric(
    facts_us_gaap: Dict[str, Any],
    candidate_keys: List[str],
    unit_key: str,
) -> Optional[pd.DataFrame]:
    """
    Pull a single metric from the ``us-gaap`` facts dictionary.

    Args:
        facts_us_gaap:  The ``facts['us-gaap']`` dictionary from the EDGAR
                        companyfacts JSON.
        candidate_keys: Ordered list of XBRL tag names to try.
        unit_key:       ``'USD'`` for monetary values, ``'USD/shares'`` for
                        per-share values.

    Returns:
        A two-column DataFrame ``[end_date, value]`` filtered to 10-K / 10-Q
        filings, or ``None`` if no data was found.
    """
    for key in candidate_keys:
        concept = facts_us_gaap.get(key)
        if concept is None:
            continue

        units = concept.get("units", {})
        entries = units.get(unit_key, [])
        if not entries:
            continue

        rows: List[Dict[str, Any]] = []
        for entry in entries:
            form = entry.get("form", "")
            if form not in ("10-K", "10-Q"):
                continue
            rows.append({
                "end_date": entry.get("end"),
                "value": entry.get("val"),
            })

        if rows:
            result_df = pd.DataFrame(rows)
            result_df["end_date"] = pd.to_datetime(result_df["end_date"])
            result_df = (
                result_df.dropna(subset=["end_date", "value"])
                .drop_duplicates(subset=["end_date"], keep="last")
                .sort_values("end_date")
                .reset_index(drop=True)
            )
            logger.info(
                "Extracted %d records for metric '%s'.", len(result_df), key
            )
            return result_df

    logger.warning(
        "None of the candidate keys %s found in us-gaap facts.", candidate_keys
    )
    return None


@retry()
def fetch_fundamentals(
    ticker: str,
    user_agent_email: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly/annual fundamental data from SEC EDGAR XBRL API.

    The function resolves the company's CIK, then pulls Revenue, Net Income,
    and EPS from the ``companyfacts`` endpoint.  ETFs and other entities
    without SEC filings return ``None``.

    Args:
        ticker:           Stock ticker symbol (e.g. ``'AAPL'``).
        user_agent_email: Email address included in the ``User-Agent`` header
                          as required by SEC EDGAR fair-access policy.

    Returns:
        A ``pd.DataFrame`` with columns ``[end_date, Revenue, Net_Income,
        EPS]`` where ``end_date`` is ``datetime64``, or ``None`` if the
        ticker is an ETF, not found, or the API call fails.
    """
    headers = _build_headers(user_agent_email)

    # Step 1 — resolve ticker → CIK
    cik_padded = _resolve_cik(ticker, headers)
    if cik_padded is None:
        return None

    # Rate-limit per SEC fair-access policy
    time.sleep(0.1)

    # Step 2 — fetch company facts
    facts_url = (
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    )
    logger.info("Fetching company facts from %s", facts_url)

    try:
        resp = requests.get(facts_url, headers=headers, timeout=15)
        resp.raise_for_status()
        facts: Dict[str, Any] = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch company facts for CIK %s: %s", cik_padded, e)
        return None

    us_gaap = facts.get("facts", {}).get("us-gaap")
    if us_gaap is None:
        logger.warning(
            "No 'us-gaap' facts found for %s (CIK %s) — likely an ETF.",
            ticker,
            cik_padded,
        )
        return None

    # Step 3 — extract individual metrics
    revenue_df = _extract_metric(
        us_gaap,
        candidate_keys=[
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ],
        unit_key="USD",
    )

    time.sleep(0.1)  # courteous rate-limiting between logical operations

    net_income_df = _extract_metric(
        us_gaap,
        candidate_keys=["NetIncomeLoss"],
        unit_key="USD",
    )

    eps_df = _extract_metric(
        us_gaap,
        candidate_keys=["EarningsPerShareDiluted", "EarningsPerShareBasic"],
        unit_key="USD/shares",
    )

    # Step 4 — merge the individual metric DataFrames
    merged: Optional[pd.DataFrame] = None

    for label, metric_df in [
        ("Revenue", revenue_df),
        ("Net_Income", net_income_df),
        ("EPS", eps_df),
    ]:
        if metric_df is None:
            continue

        metric_df = metric_df.rename(columns={"value": label})

        if merged is None:
            merged = metric_df
        else:
            merged = pd.merge(merged, metric_df, on="end_date", how="outer")

    if merged is None or merged.empty:
        logger.warning(
            "Could not extract any fundamental metrics for %s.", ticker
        )
        return None

    merged = merged.sort_values("end_date").reset_index(drop=True)

    # Ensure all expected columns exist (fill missing metrics with NaN)
    for col in ("Revenue", "Net_Income", "EPS"):
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[["end_date", "Revenue", "Net_Income", "EPS"]]

    logger.info(
        "Fundamentals retrieved for %s — %d records, %s → %s.",
        ticker,
        len(merged),
        merged["end_date"].min().date(),
        merged["end_date"].max().date(),
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# align_and_merge
# ═══════════════════════════════════════════════════════════════════════════

def align_and_merge(
    ohlcv_df: pd.DataFrame,
    fundamentals_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge daily OHLCV prices with quarterly fundamentals using a
    forward-fill (``pd.merge_asof``) strategy so that each trading day
    carries the most-recently-reported fundamental values.

    Args:
        ohlcv_df:        DataFrame with ``DatetimeIndex`` named ``'Date'``
                         and OHLCV columns.
        fundamentals_df: DataFrame with columns ``[end_date, Revenue,
                         Net_Income, EPS]``, or ``None`` (e.g. for ETFs).

    Returns:
        A unified ``pd.DataFrame`` with a ``DatetimeIndex`` named
        ``'Date'``.  If *fundamentals_df* is ``None`` the original
        *ohlcv_df* is returned unchanged.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        raise ValueError("OHLCV DataFrame cannot be empty or None.")

    if fundamentals_df is None or fundamentals_df.empty:
        logger.info(
            "No fundamentals provided — returning OHLCV data as-is."
        )
        return ohlcv_df

    logger.info("Aligning fundamentals with OHLCV data via merge_asof.")

    # Prepare OHLCV side: ensure sorted DatetimeIndex and a 'Date' column
    ohlcv = ohlcv_df.copy()
    ohlcv.index = pd.to_datetime(ohlcv.index)
    ohlcv = ohlcv.sort_index()
    ohlcv["_merge_date"] = ohlcv.index

    # Prepare fundamentals side: sorted datetime column
    fund = fundamentals_df.copy()
    fund["end_date"] = pd.to_datetime(fund["end_date"])
    fund = fund.sort_values("end_date").reset_index(drop=True)

    # Asof merge: for each trading day, pick the most recent fundamental row
    merged = pd.merge_asof(
        ohlcv,
        fund,
        left_on="_merge_date",
        right_on="end_date",
        direction="backward",
    )

    # Restore the DatetimeIndex
    merged.index = ohlcv.index
    merged.index.name = "Date"

    # Clean up helper columns
    merged.drop(columns=["_merge_date", "end_date"], inplace=True, errors="ignore")

    logger.info(
        "Merge complete. Final DataFrame shape: %s. Columns: %s",
        merged.shape,
        list(merged.columns),
    )
    return merged