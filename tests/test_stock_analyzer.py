"""
Unit tests for the Stock Analyzer core modules (no network access needed).

Run with:  python -m pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from features import calculate_technical_indicators, create_target_labels, scale_features
from models import (StockDataset, train_random_forest,
                    compute_probability_bounds, normalize_probabilities)
from backtest import run_backtest, calculate_performance_metrics


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV frame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

def test_indicators_add_expected_columns_without_mutating_input():
    df = make_ohlcv()
    original_cols = list(df.columns)
    out = calculate_technical_indicators(df)

    expected = ["SMA_20", "SMA_50", "EMA_20", "EMA_50", "RSI_14", "MACD",
                "Signal_Line", "MACD_Hist", "BB_Upper", "BB_Middle", "BB_Lower",
                "Vol_20", "Momentum_10", "Daily_Returns", "ATR_14", "OBV",
                "ROC_10", "Stoch_K", "Stoch_D", "VWAP"]
    for col in expected:
        assert col in out.columns, f"missing indicator column {col}"
    assert list(df.columns) == original_cols, "input DataFrame was mutated"


def test_indicators_reject_non_datetime_index():
    df = make_ohlcv().reset_index(drop=True)
    with pytest.raises(ValueError):
        calculate_technical_indicators(df)


def test_target_labels_horizon_and_threshold():
    # Deterministic price ramp: +1% per day
    n = 50
    close = 100 * 1.01 ** np.arange(n)
    df = pd.DataFrame({"Close": close},
                      index=pd.bdate_range("2024-01-01", periods=n))
    horizon = 5
    labels = create_target_labels(df, horizon=horizon, threshold=0.02)

    # 5 days of +1% ≈ +5.1% > 2% → every computable label is 1
    assert np.all(labels[:-horizon] == 1.0)
    # Last `horizon` labels are undefined
    assert np.all(np.isnan(labels[-horizon:]))

    # Threshold above the 5-day return → all labels 0
    labels_hi = create_target_labels(df, horizon=horizon, threshold=0.10)
    assert np.all(labels_hi[:-horizon] == 0.0)


def test_scale_features_fits_on_train_only():
    df = make_ohlcv()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    train_end = 200
    scaled_train, scaled_test, scaler = scale_features(df, train_end, cols)

    assert scaled_train.shape == (train_end, len(cols))
    assert scaled_test.shape == (len(df) - train_end, len(cols))
    # Train data must be within [0, 1] by construction of MinMaxScaler
    assert scaled_train.min() >= -1e-9 and scaled_train.max() <= 1 + 1e-9
    # Scaler bounds must equal the train slice's bounds (proof it never saw test)
    np.testing.assert_allclose(scaler.data_max_, df[cols].iloc[:train_end].max().values)


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

def test_stock_dataset_window_and_target_alignment():
    n, f, seq = 40, 3, 10
    features = np.arange(n * f, dtype=np.float32).reshape(n, f)
    labels = np.arange(n, dtype=np.float32)
    ds = StockDataset(features, labels, sequence_length=seq)

    assert len(ds) == n - seq
    X, y = ds[0]
    np.testing.assert_allclose(X.numpy(), features[0:seq])
    assert y.item() == labels[seq]
    # Window-end row is features[idx + seq - 1]
    X5, y5 = ds[5]
    np.testing.assert_allclose(X5.numpy()[-1], features[5 + seq - 1])
    assert y5.item() == labels[5 + seq]


def test_tree_training_aligns_with_window_end_row():
    # Label at t is a deterministic function of the feature row at t-1
    # (the window-end row the ensemble sees at eval time). A correctly
    # aligned tree model should learn this perfectly.
    rng = np.random.default_rng(1)
    n, seq = 400, 10
    features = rng.random((n, 4)).astype(np.float32)
    labels = np.zeros(n, dtype=np.float32)
    labels[1:] = (features[:-1, 0] > 0.5).astype(np.float32)

    ds = StockDataset(features, labels, sequence_length=seq)
    rf = train_random_forest(ds, n_estimators=50)

    window_end_X = features[seq - 1: seq - 1 + len(ds)]
    window_targets = labels[seq: seq + len(ds)]
    acc = (rf.predict(window_end_X) == window_targets).mean()
    assert acc > 0.95, f"tree/eval alignment broken — accuracy {acc:.2f}"


def test_probability_normalization_bounds_and_clipping():
    probs = np.linspace(0.2, 0.4, 100)
    bounds = compute_probability_bounds(probs)
    assert bounds[0] < bounds[1]

    normalized = normalize_probabilities(np.array([0.0, 0.3, 1.0]), bounds)
    assert normalized[0] == 0.0          # below range → clipped
    assert normalized[2] == 1.0          # above range → clipped
    assert 0.0 < normalized[1] < 1.0

    # Degenerate (constant) probabilities must not divide by zero
    flat_bounds = compute_probability_bounds(np.full(50, 0.5))
    assert flat_bounds == (0.0, 1.0)


# ---------------------------------------------------------------------------
# backtest.py
# ---------------------------------------------------------------------------

def _simple_backtest(prices, predictions, **kwargs):
    dates = pd.bdate_range("2024-01-01", periods=len(prices))
    return run_backtest(
        close_prices=pd.Series(prices, index=dates),
        predictions=np.asarray(predictions, dtype=float),
        dates=dates,
        initial_capital=10_000.0,
        signal_threshold=0.5,
        **kwargs,
    )


def test_backtest_buy_take_profit_sell_cycle():
    # Buy at 100, +5% the next day hits the +3% take profit
    pf = _simple_backtest([100.0, 105.0, 105.0], [0.9, 0.9, 0.1])
    assert list(pf["Action"]) == ["BUY", "SELL", "HOLD"]
    assert pf["Portfolio_Value"].iloc[-1] > 10_000


def test_backtest_commission_and_slippage_reduce_returns():
    prices = [100.0, 105.0, 105.0]
    preds = [0.9, 0.9, 0.1]
    free = _simple_backtest(prices, preds)
    costly = _simple_backtest(prices, preds, commission_pct=0.001, slippage_pct=0.001)

    assert costly["Cum_Fees"].iloc[-1] > 0
    assert costly["Portfolio_Value"].iloc[-1] < free["Portfolio_Value"].iloc[-1]
    # Buy fill must be worse (higher) than close, sell fill lower than close
    buy_row = costly[costly["Action"] == "BUY"].iloc[0]
    sell_row = costly[costly["Action"] == "SELL"].iloc[0]
    assert buy_row["Fill_Price"] > buy_row["Price"]
    assert sell_row["Fill_Price"] < sell_row["Price"]


def test_backtest_capital_conservation_with_costs():
    prices = [100.0, 105.0, 105.0]
    pf = _simple_backtest(prices, [0.9, 0.9, 0.1],
                          commission_pct=0.001, slippage_pct=0.001)
    buy = pf[pf["Action"] == "BUY"].iloc[0]
    sell = pf[pf["Action"] == "SELL"].iloc[0]
    shares = buy["Shares_Held"]
    expected_final = (10_000.0
                      + shares * (sell["Fill_Price"] - buy["Fill_Price"])
                      - pf["Cum_Fees"].iloc[-1])
    assert pf["Portfolio_Value"].iloc[-1] == pytest.approx(expected_final, abs=0.05)


def test_backtest_atr_risk_sizing_caps_position():
    n = 60
    prices = np.full(n, 100.0)
    dates = pd.bdate_range("2024-01-01", periods=n)
    trend = pd.DataFrame({"ATR_14": np.full(n, 5.0)}, index=dates)

    preds = np.zeros(n)
    preds[0] = 0.9
    pf = run_backtest(
        close_prices=pd.Series(prices, index=dates), predictions=preds,
        dates=dates, initial_capital=10_000.0, signal_threshold=0.5,
        trend_data=trend, use_atr_risk=True,
        atr_stop_mult=2.0, risk_per_trade=0.02,
    )
    buy_rows = pf[pf["Action"] == "BUY"]
    assert len(buy_rows) == 1
    # risk budget 200 / stop distance 10 → 20 shares (vs 90 with fixed sizing)
    assert buy_rows["Shares_Held"].iloc[0] == 20


def test_metrics_win_rate_uses_fill_prices():
    pf = _simple_backtest([100.0, 105.0, 105.0], [0.9, 0.9, 0.1],
                          commission_pct=0.001, slippage_pct=0.001)
    metrics = calculate_performance_metrics(pf, initial_capital=10_000.0)
    assert metrics["Total_Trades"] == 1
    assert metrics["Win_Rate_Pct"] == 100.0
    assert metrics["Total_Fees"] > 0
    assert metrics["Buy_Hold_Return_Pct"] == pytest.approx(5.0)


def test_metrics_empty_portfolio():
    metrics = calculate_performance_metrics(pd.DataFrame(), initial_capital=10_000.0)
    assert metrics["Total_Trades"] == 0
    assert metrics["Total_Return_Pct"] == 0.0
