#!/usr/bin/env python3
"""
walkforward.py — Walk-forward validation for the Stock Analyzer.

Instead of a single 80/20 split, the model is retrained on a rolling window
and tested on the segment that immediately follows, sliding forward until
the data is exhausted:

    fold 1:  [train ------ 500 rows ------][test 63]
    fold 2:        [train ------ 500 rows ------][test 63]
    ...

This produces many out-of-sample estimates of accuracy and backtest
performance, giving a far more trustworthy picture than one split.

Usage:
    python walkforward.py            # uses .env config (TARGET_TICKER etc.)

Extra env vars:
    WF_TRAIN_ROWS   rolling training window size in trading days (default 500)
    WF_TEST_ROWS    test segment size in trading days (default 63 ≈ one quarter)
    WF_EPOCHS       LSTM epochs per fold (default 15 — folds retrain many times)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
import torch
from torch.utils.data import DataLoader

from data_pipeline import fetch_ohlcv, fetch_fundamentals, align_and_merge
from features import calculate_technical_indicators, create_target_labels, scale_features
from models import (StockDataset, LSTMModel, train_model, evaluate_model,
                    train_random_forest, train_gradient_boosting,
                    compute_probability_bounds, normalize_probabilities)
from backtest import run_backtest, calculate_performance_metrics
from main import _load_config, FEATURE_COLUMNS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("walkforward")


def run_fold(df: pd.DataFrame, labels: np.ndarray, cfg: dict,
             fold_start: int, train_rows: int, test_rows: int,
             epochs: int, device: str) -> dict | None:
    """Train on df[fold_start : fold_start+train_rows], test on the next
    test_rows rows. Returns fold metrics or None if the fold is unusable."""
    seq_len = cfg["sequence_length"]
    fold_end = fold_start + train_rows + test_rows
    fold_df = df.iloc[fold_start:fold_end]
    fold_labels = labels[fold_start:fold_end]

    feat_cols = [c for c in FEATURE_COLUMNS if c in fold_df.columns]
    try:
        scaled_train, scaled_test, scaler = scale_features(fold_df, train_rows, feat_cols)
        train_ds = StockDataset(scaled_train, fold_labels[:train_rows], sequence_length=seq_len)
        test_ds = StockDataset(scaled_test, fold_labels[train_rows:], sequence_length=seq_len)
    except ValueError:
        return None
    if len(train_ds) == 0 or len(test_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = LSTMModel(
        input_size=scaled_train.shape[1],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    train_model(model, train_loader, epochs=epochs, lr=cfg["learning_rate"],
                device=device, weight_decay=1e-4)

    rf_model = train_random_forest(
        train_ds, n_estimators=cfg["rf_n_estimators"],
        max_depth=cfg["rf_max_depth"], min_samples_leaf=cfg["rf_min_samples_leaf"])
    gb_model = train_gradient_boosting(
        train_ds, n_estimators=cfg["gb_n_estimators"], max_depth=cfg["gb_max_depth"],
        learning_rate=cfg["gb_lr"], subsample=cfg["gb_subsample"])

    ens_kwargs = dict(rf_model=rf_model, gb_model=gb_model,
                      lstm_weight=cfg["lstm_weight"], rf_weight=cfg["rf_weight"],
                      gb_weight=cfg["gb_weight"])

    probs, actuals = evaluate_model(model, test_loader, device=device, **ens_kwargs)

    # Leak-free normalization: bounds come from training-set predictions
    train_eval_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False)
    train_probs, _ = evaluate_model(model, train_eval_loader, device=device, **ens_kwargs)
    probs = normalize_probabilities(probs, compute_probability_bounds(train_probs))

    preds = (probs >= cfg["signal_threshold"]).astype(float)
    accuracy = (preds == actuals).mean() * 100

    test_df = fold_df.iloc[train_rows:]
    bt_close = test_df["Close"].iloc[seq_len: seq_len + len(probs)]
    bt_dates = test_df.index[seq_len: seq_len + len(probs)]

    portfolio_df = run_backtest(
        close_prices=bt_close, predictions=probs, dates=bt_dates,
        initial_capital=cfg["initial_capital"],
        signal_threshold=cfg["signal_threshold"], trend_data=df,
        commission_pct=cfg["commission_pct"], slippage_pct=cfg["slippage_pct"],
        use_atr_risk=cfg["use_atr_risk"], atr_stop_mult=cfg["atr_stop_mult"],
        atr_tp_mult=cfg["atr_tp_mult"], risk_per_trade=cfg["risk_per_trade"],
    )
    metrics = calculate_performance_metrics(
        portfolio_df, initial_capital=cfg["initial_capital"],
        risk_free_rate=cfg["risk_free_rate"])

    wins = metrics["Win_Rate_Pct"] / 100 * metrics["Total_Trades"]
    return {
        "test_start": str(bt_dates[0].date()) if len(bt_dates) else "n/a",
        "test_end": str(bt_dates[-1].date()) if len(bt_dates) else "n/a",
        "accuracy": accuracy,
        "return_pct": metrics["Total_Return_Pct"],
        "bh_return_pct": metrics["Buy_Hold_Return_Pct"],
        "sharpe": metrics["Sharpe_Ratio"],
        "max_dd_pct": metrics["Max_Drawdown_Pct"],
        "trades": metrics["Total_Trades"],
        "wins": wins,
    }


def main() -> None:
    cfg = _load_config()
    ticker = cfg["ticker"]
    train_rows = int(os.getenv("WF_TRAIN_ROWS", 500))
    test_rows = int(os.getenv("WF_TEST_ROWS", 63))
    epochs = int(os.getenv("WF_EPOCHS", 15))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fetch a wide window so several folds fit
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=2500)

    print("=" * 70)
    print(f"  WALK-FORWARD VALIDATION — {ticker}")
    print(f"  Train window: {train_rows} rows  |  Test window: {test_rows} rows  "
          f"|  Epochs/fold: {epochs}")
    print("=" * 70)

    ohlcv_df = fetch_ohlcv(ticker, start_date=start_date, end_date=end_date)
    if ohlcv_df is None or ohlcv_df.empty:
        print("Failed to fetch OHLCV data. Aborting.")
        return
    try:
        fund_df = fetch_fundamentals(ticker, user_agent_email=cfg["sec_email"])
    except Exception:
        fund_df = None

    df = calculate_technical_indicators(align_and_merge(ohlcv_df, fund_df))
    labels = create_target_labels(df, horizon=cfg["prediction_horizon"],
                                  threshold=cfg["classification_threshold"])
    df = df.copy()
    df["_target"] = labels
    valid_cols = [c for c in FEATURE_COLUMNS + ["_target"] if c in df.columns]
    df.dropna(subset=valid_cols, inplace=True)
    labels_clean = df["_target"].values.astype(np.float32)

    n = len(df)
    n_folds = max(0, (n - train_rows) // test_rows)
    print(f"  Clean rows: {n}  →  {n_folds} folds\n")
    if n_folds == 0:
        print("Not enough data for a single fold. Reduce WF_TRAIN_ROWS/WF_TEST_ROWS.")
        return

    folds = []
    for k in range(n_folds):
        fold_start = k * test_rows
        r = run_fold(df, labels_clean, cfg, fold_start, train_rows, test_rows,
                     epochs, device)
        if r is None:
            print(f"  Fold {k + 1}/{n_folds}: SKIPPED (insufficient data)")
            continue
        folds.append(r)
        print(f"  Fold {k + 1}/{n_folds} [{r['test_start']} → {r['test_end']}]  "
              f"acc={r['accuracy']:.1f}%  ret={r['return_pct']:+.2f}%  "
              f"b&h={r['bh_return_pct']:+.2f}%  trades={r['trades']}")

    if not folds:
        print("\nNo usable folds.")
        return

    total_trades = sum(f["trades"] for f in folds)
    total_wins = sum(f["wins"] for f in folds)
    compounded = (np.prod([1 + f["return_pct"] / 100 for f in folds]) - 1) * 100
    compounded_bh = (np.prod([1 + f["bh_return_pct"] / 100 for f in folds]) - 1) * 100

    print("\n" + "=" * 70)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 70)
    print(f"  Folds completed:        {len(folds)}")
    print(f"  Mean test accuracy:     {np.mean([f['accuracy'] for f in folds]):.2f}%")
    print(f"  Mean fold return:       {np.mean([f['return_pct'] for f in folds]):+.2f}%")
    print(f"  Compounded return:      {compounded:+.2f}%")
    print(f"  Compounded buy & hold:  {compounded_bh:+.2f}%")
    print(f"  Mean Sharpe:            {np.mean([f['sharpe'] for f in folds]):.2f}")
    print(f"  Worst fold drawdown:    {max(f['max_dd_pct'] for f in folds):.2f}%")
    print(f"  Total round-trips:      {total_trades}")
    if total_trades > 0:
        print(f"  AGGREGATE WIN RATE:     {total_wins / total_trades * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
