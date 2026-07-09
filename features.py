"""
features.py — Feature engineering and scaling for the Stock Analyzer.

Responsibilities:
  • Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands,
    volatility, momentum, skewness, daily returns)
  • Create binary target labels for supervised learning
  • Scale feature columns with train-only fitted MinMaxScaler

All three public functions are TOP-LEVEL exports so they can be imported
directly:

    from features import calculate_technical_indicators, create_target_labels, scale_features
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import logging

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# calculate_technical_indicators
# ═══════════════════════════════════════════════════════════════════════════

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment a price DataFrame with standard technical indicators.

    The function operates **in-place on a copy** — the original DataFrame
    is not mutated.

    Required input columns: ``Close``, ``High``, ``Low``, ``Volume``.

    Added columns:
        SMA_20, SMA_50, EMA_20, EMA_50, RSI_14,
        MACD, Signal_Line, MACD_Hist,
        BB_Upper, BB_Middle, BB_Lower,
        Vol_20, Momentum_10, Skew_20, Daily_Returns

    Args:
        df: DataFrame with a ``DatetimeIndex`` and at least the four
            required price/volume columns.

    Returns:
        The same DataFrame with all indicator columns appended.

    Raises:
        ValueError: If the index is not datetime or required columns are
                    missing.
    """
    logger.info("Starting technical indicator calculation...")

    # ── Validation ──────────────────────────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError(
            "DataFrame index must be a DatetimeIndex for time-series features."
        )

    required = {"Close", "High", "Low", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    out = df.copy()
    close: pd.Series = out["Close"]

    # ── Simple Moving Averages ──────────────────────────────────────────
    out["SMA_20"] = close.rolling(window=20).mean()
    out["SMA_50"] = close.rolling(window=50).mean()
    logger.info("SMA calculations complete.")

    # ── Exponential Moving Averages ─────────────────────────────────────
    out["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    logger.info("EMA calculations complete.")

    # ── RSI (14-period, EWMA smoothing) ─────────────────────────────────
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()

    rs = avg_gain / avg_loss
    out["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
    logger.info("RSI calculation complete.")

    # ── MACD (12, 26, 9) ───────────────────────────────────────────────
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["Signal_Line"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["Signal_Line"]
    logger.info("MACD calculation complete.")

    # ── Bollinger Bands (20-day, 2 std) ─────────────────────────────────
    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    out["BB_Upper"] = rolling_mean + 2.0 * rolling_std
    out["BB_Middle"] = rolling_mean
    out["BB_Lower"] = rolling_mean - 2.0 * rolling_std
    logger.info("Bollinger Bands calculation complete.")

    # ── Volatility (20-day rolling std of daily returns) ────────────────
    daily_ret = close.pct_change()
    out["Vol_20"] = daily_ret.rolling(window=20).std()
    logger.info("Volatility calculation complete.")

    # ── Momentum (10-day) ──────────────────────────────────────────────
    out["Momentum_10"] = close / close.shift(10) - 1.0

    # ── Skewness (20-day rolling, using pandas — no scipy needed) ──────
    out["Skew_20"] = daily_ret.rolling(window=20).skew()
    logger.info("Skewness calculation complete.")

    # ── Daily Returns ──────────────────────────────────────────────────
    out["Daily_Returns"] = daily_ret

    logger.info(
        "Technical indicator calculation finished. "
        "DataFrame shape: %s, new columns: %d.",
        out.shape,
        len(out.columns) - len(df.columns),
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# create_target_labels
# ═══════════════════════════════════════════════════════════════════════════

def create_target_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Create binary classification labels based on forward-looking returns.

    Label logic::

        future_return = (Close[t + horizon] − Close[t]) / Close[t]
        label = 1  if future_return > threshold
                0  otherwise

    The last *horizon* entries are set to ``np.nan`` because the future
    return cannot be computed.

    Args:
        df:        DataFrame containing a ``Close`` column.
        horizon:   Number of trading days to look ahead (default 5).
        threshold: Minimum fractional return to trigger a positive label
                   (default 0.01, i.e. 1 %).

    Returns:
        A 1-D ``np.ndarray`` of length ``len(df)`` with values in
        ``{0.0, 1.0, np.nan}``.
    """
    logger.info(
        "Creating target labels — horizon=%d, threshold=%.4f.", horizon, threshold
    )

    close: pd.Series = df["Close"]
    future_close = close.shift(-horizon)
    future_return = (future_close - close) / close

    labels = np.where(future_return > threshold, 1.0, 0.0)

    # Mark the trailing positions where forward return is undefined
    labels[-horizon:] = np.nan

    logger.info(
        "Label distribution (excl. NaN): 1=%.1f%%, 0=%.1f%%.",
        np.nanmean(labels) * 100,
        (1 - np.nanmean(labels)) * 100,
    )
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# scale_features  (TOP-LEVEL — importable directly)
# ═══════════════════════════════════════════════════════════════════════════

def scale_features(
    feature_df: pd.DataFrame,
    train_end_idx: int,
    feature_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit a ``MinMaxScaler`` on the **training split only** and transform
    both training and test splits to prevent data leakage.

    Args:
        feature_df:      Full DataFrame containing at least *feature_columns*.
        train_end_idx:   Integer index that separates training from test data.
                         Training data = ``feature_df.iloc[:train_end_idx]``.
        feature_columns: List of column names to scale.

    Returns:
        A 3-tuple ``(scaled_train, scaled_test, scaler)`` where:

        * **scaled_train** — ``np.ndarray`` of shape
          ``(train_end_idx, len(feature_columns))``
        * **scaled_test**  — ``np.ndarray`` of shape
          ``(len(feature_df) − train_end_idx, len(feature_columns))``
        * **scaler**       — the fitted ``MinMaxScaler`` instance (useful
          for inverse-transforming predictions later)
    """
    logger.info(
        "Scaling %d features. Train rows: %d, Test rows: %d.",
        len(feature_columns),
        train_end_idx,
        len(feature_df) - train_end_idx,
    )

    train_data = feature_df[feature_columns].iloc[:train_end_idx]
    test_data = feature_df[feature_columns].iloc[train_end_idx:]

    scaler = MinMaxScaler()
    scaled_train: np.ndarray = scaler.fit_transform(train_data)
    scaled_test: np.ndarray = scaler.transform(test_data)

    logger.info(
        "Scaling complete. Train array shape: %s, Test array shape: %s.",
        scaled_train.shape,
        scaled_test.shape,
    )
    return scaled_train, scaled_test, scaler