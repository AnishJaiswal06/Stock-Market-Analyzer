#!/usr/bin/env python3
"""
Multi-Stock Win Rate Test
=========================
Runs the full pipeline (data → features → ensemble → backtest) across
multiple S&P 500 stocks and reports win rates for each.
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
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader

from data_pipeline import fetch_ohlcv, fetch_fundamentals, align_and_merge
from features import calculate_technical_indicators, create_target_labels, scale_features
from models import (StockDataset, LSTMModel, train_model, evaluate_model,
                    train_random_forest, train_gradient_boosting,
                    compute_probability_bounds, normalize_probabilities)
from backtest import run_backtest, calculate_performance_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "Signal_Line", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "Vol_20", "Momentum_10", "Daily_Returns", "Sector_Return",
    "ATR_14", "OBV", "ROC_10", "Stoch_K", "Stoch_D", "VWAP",
]

# Diverse S&P 500 stocks across sectors
TEST_TICKERS = [
    "AAPL",   # Technology
    "MSFT",   # Technology
    "AMZN",   # Consumer Discretionary
    "GOOG",   # Communication Services
    "META",   # Communication Services
    "NVDA",   # Technology (Semiconductors)
    "JPM",    # Financials
    "JNJ",    # Healthcare
    "PG",     # Consumer Staples
    "XOM",    # Energy
]


def run_single_stock(ticker: str, params: dict) -> dict:
    """Run the full pipeline for a single stock and return metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=params["date_range_days"])

    result = {"ticker": ticker, "status": "FAILED", "win_rate": None,
              "total_trades": 0, "total_return": None, "buy_hold_return": None,
              "accuracy": None, "auc": None}

    # 1. Fetch data
    try:
        ohlcv_df = fetch_ohlcv(ticker, start_date=start_date, end_date=end_date)
        if ohlcv_df is None or ohlcv_df.empty:
            result["status"] = "NO_DATA"
            return result
    except Exception:
        return result

    # 2. Fundamentals
    try:
        fund_df = fetch_fundamentals(ticker, user_agent_email=params.get("sec_email", ""))
    except Exception:
        fund_df = None

    # 3. Merge & features
    try:
        merged_df = align_and_merge(ohlcv_df, fund_df)
        df = calculate_technical_indicators(merged_df)
    except Exception:
        return result

    # 4. Target labels
    try:
        labels = create_target_labels(
            df, horizon=params["prediction_horizon"],
            threshold=params["classification_threshold"],
        )
    except Exception:
        return result

    # 5. Clean
    df = df.copy()
    df["_target"] = labels
    valid_cols = [c for c in FEATURE_COLUMNS + ["_target"] if c in df.columns]
    df.dropna(subset=valid_cols, inplace=True)

    seq_len = params["sequence_length"]
    if len(df) < seq_len + 20:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    labels_clean = df["_target"].values.astype(np.float32)

    # 6. Train/test split
    train_end = int(len(df) * 0.8)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    try:
        scaled_train, scaled_test, scaler = scale_features(df, train_end, feature_cols)
    except Exception:
        return result

    train_labels = labels_clean[:train_end]
    test_labels = labels_clean[train_end:]

    # 7. Datasets
    try:
        train_ds = StockDataset(scaled_train, train_labels, sequence_length=seq_len)
        test_ds = StockDataset(scaled_test, test_labels, sequence_length=seq_len)
        if len(train_ds) == 0 or len(test_ds) == 0:
            result["status"] = "EMPTY_DATASET"
            return result
    except Exception:
        return result

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)

    # Val loader
    val_loader = None
    val_split = int(len(scaled_train) * 0.9)
    if len(scaled_train) - val_split > seq_len + 1:
        try:
            val_ds = StockDataset(scaled_train[val_split:], train_labels[val_split:], sequence_length=seq_len)
            if len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
        except ValueError:
            pass

    # 8. Train LSTM
    input_size = scaled_train.shape[1]
    model = LSTMModel(
        input_size=input_size, hidden_size=params["hidden_size"],
        num_layers=params["num_layers"], dropout=params["dropout"],
    ).to(device)

    try:
        train_model(model, train_loader, val_loader=val_loader,
                    epochs=params["epochs"], lr=params["learning_rate"],
                    device=device, weight_decay=1e-4)
    except Exception:
        return result

    # 9. Train RF + GB
    try:
        rf_model = train_random_forest(train_ds,
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_leaf=params["rf_min_samples_leaf"])
    except Exception:
        return result

    try:
        gb_model = train_gradient_boosting(train_ds,
            n_estimators=params["gb_n_estimators"],
            max_depth=params["gb_max_depth"],
            learning_rate=params["gb_lr"],
            subsample=params["gb_subsample"])
    except Exception:
        return result

    # 10. Evaluate ensemble
    try:
        probabilities, actuals = evaluate_model(
            model, test_loader, device=device,
            rf_model=rf_model, gb_model=gb_model,
            lstm_weight=params["lstm_weight"],
            rf_weight=params["rf_weight"],
            gb_weight=params["gb_weight"],
        )
    except Exception:
        return result

    # Normalize probabilities with bounds fitted on TRAINING-set predictions
    # (leak-free — no look-ahead into the test window)
    try:
        train_eval_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=False)
        train_probs, _ = evaluate_model(
            model, train_eval_loader, device=device,
            rf_model=rf_model, gb_model=gb_model,
            lstm_weight=params["lstm_weight"],
            rf_weight=params["rf_weight"],
            gb_weight=params["gb_weight"],
        )
        probabilities = normalize_probabilities(
            probabilities, compute_probability_bounds(train_probs))
    except Exception:
        return result

    # Prediction-quality metrics
    preds_binary = (probabilities >= params["signal_threshold"]).astype(float)
    result["accuracy"] = (preds_binary == actuals).mean() * 100
    if len(np.unique(actuals)) > 1:
        from sklearn.metrics import roc_auc_score
        result["auc"] = roc_auc_score(actuals, probabilities)

    # 11. Backtest
    test_df = df.iloc[train_end:]
    bt_start = seq_len
    bt_close = test_df["Close"].iloc[bt_start: bt_start + len(probabilities)]
    bt_dates = test_df.index[bt_start: bt_start + len(probabilities)]

    try:
        portfolio_df = run_backtest(
            close_prices=bt_close, predictions=probabilities,
            dates=bt_dates, initial_capital=params["initial_capital"],
            signal_threshold=params["signal_threshold"],
            trend_data=df,
            commission_pct=params.get("commission_pct", 0.0),
            slippage_pct=params.get("slippage_pct", 0.0),
            use_atr_risk=params.get("use_atr_risk", False),
        )
    except Exception:
        return result

    metrics = calculate_performance_metrics(
        portfolio_df, initial_capital=params["initial_capital"],
        risk_free_rate=params["risk_free_rate"],
    )

    result["status"] = "OK"
    result["win_rate"] = metrics["Win_Rate_Pct"]
    result["total_trades"] = metrics["Total_Trades"]
    result["total_return"] = metrics["Total_Return_Pct"]
    result["buy_hold_return"] = metrics["Buy_Hold_Return_Pct"]
    result["sharpe"] = metrics["Sharpe_Ratio"]
    result["max_drawdown"] = metrics["Max_Drawdown_Pct"]

    return result


if __name__ == "__main__":
    load_dotenv()

    params = {
        "date_range_days": 1500,
        "sequence_length": 30,
        "epochs": 30,
        "batch_size": 32,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.15,
        "learning_rate": 0.001,
        "prediction_horizon": 10,
        "classification_threshold": 0.02,
        "initial_capital": 10000,
        "risk_free_rate": 0.02,
        "signal_threshold": 0.40,
        "lstm_weight": 0.1,
        "rf_weight": 0.3,
        "gb_weight": 0.6,
        "rf_n_estimators": 500,
        "rf_max_depth": 10,
        "rf_min_samples_leaf": 15,
        "gb_n_estimators": 400,
        "gb_max_depth": 3,
        "gb_lr": 0.03,
        "gb_subsample": 0.8,
        "commission_pct": 0.0005,
        "slippage_pct": 0.0005,
        "use_atr_risk": False,
        "sec_email": os.getenv("SEC_USER_AGENT_EMAIL", ""),
    }

    print("=" * 75)
    print("  MULTI-STOCK WIN RATE TEST")
    print(f"  Testing {len(TEST_TICKERS)} S&P 500 stocks")
    print(f"  Signal threshold: {params['signal_threshold']}")
    print("=" * 75)
    print()

    results = []
    for i, ticker in enumerate(TEST_TICKERS, 1):
        print(f"  [{i}/{len(TEST_TICKERS)}] Running {ticker}...", end=" ", flush=True)
        r = run_single_stock(ticker, params)
        results.append(r)

        if r["status"] == "OK" and r["total_trades"] > 0:
            print(f"Win Rate: {r['win_rate']:.1f}% | "
                  f"Trades: {r['total_trades']} | "
                  f"Return: {r['total_return']:.2f}% | "
                  f"B&H: {r['buy_hold_return']:.2f}%")
        elif r["status"] == "OK":
            print(f"No trades executed (0 round-trips)")
        else:
            print(f"Status: {r['status']}")

    # Summary
    print()
    print("=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    ok_results = [r for r in results if r["status"] == "OK"]
    traded = [r for r in ok_results if r["total_trades"] > 0]
    no_trades = [r for r in ok_results if r["total_trades"] == 0]

    total_wins = sum(r["total_trades"] * r["win_rate"] / 100 for r in traded)
    total_trades = sum(r["total_trades"] for r in traded)

    print(f"  Stocks tested:        {len(TEST_TICKERS)}")
    print(f"  Stocks with trades:   {len(traded)}")
    print(f"  Stocks (no trades):   {len(no_trades)}")
    print(f"  Total round-trips:    {total_trades}")

    if total_trades > 0:
        aggregate_wr = (total_wins / total_trades) * 100
        print(f"  AGGREGATE WIN RATE:   {aggregate_wr:.1f}%")
    else:
        print(f"  AGGREGATE WIN RATE:   N/A (no trades)")

    if traded:
        avg_return = np.mean([r["total_return"] for r in traded])
        avg_bh = np.mean([r["buy_hold_return"] for r in traded])
        print(f"  Avg strategy return:  {avg_return:.2f}%")
        print(f"  Avg buy-and-hold:     {avg_bh:.2f}%")

    accs = [r["accuracy"] for r in ok_results if r["accuracy"] is not None]
    aucs = [r["auc"] for r in ok_results if r["auc"] is not None]
    if accs:
        print(f"  Mean test accuracy:   {np.mean(accs):.2f}%")
    if aucs:
        print(f"  Mean ROC-AUC:         {np.mean(aucs):.4f}")

    print()
    print("  Per-Stock Breakdown:")
    print(f"  {'Ticker':<8} {'Status':<8} {'Win Rate':>10} {'Trades':>8} {'Return':>10} "
          f"{'B&H':>10} {'Accuracy':>10} {'AUC':>7}")
    print("  " + "-" * 76)
    for r in results:
        wr = f"{r['win_rate']:.1f}%" if r['win_rate'] is not None else "N/A"
        ret = f"{r['total_return']:.2f}%" if r['total_return'] is not None else "N/A"
        bh = f"{r['buy_hold_return']:.2f}%" if r['buy_hold_return'] is not None else "N/A"
        acc = f"{r['accuracy']:.1f}%" if r['accuracy'] is not None else "N/A"
        auc = f"{r['auc']:.3f}" if r['auc'] is not None else "N/A"
        print(f"  {r['ticker']:<8} {r['status']:<8} {wr:>10} {r['total_trades']:>8} "
              f"{ret:>10} {bh:>10} {acc:>10} {auc:>7}")

    print("=" * 75)
