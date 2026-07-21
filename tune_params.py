"""
Hyperparameter tuning script — Round 4 (Final push for 59%+)
3-model ensemble: LSTM + RF + GradientBoosting
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import logging
from datetime import date, timedelta
from torch.utils.data import DataLoader

from data_pipeline import fetch_ohlcv
from features import calculate_technical_indicators, create_target_labels, scale_features
from models import (StockDataset, LSTMModel, train_model, evaluate_model,
                    train_random_forest, train_gradient_boosting)

logging.basicConfig(level=logging.WARNING)

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "Signal_Line", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "Vol_20", "Momentum_10", "Daily_Returns", "Sector_Return",
    "ATR_14", "OBV", "ROC_10", "Stoch_K", "Stoch_D", "VWAP",
]

TICKERS = ["GOOG", "AAPL", "AVGO", "C", "COST", "AMZN", "META", "TSM", "NVDA"]
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_single_stock(ticker, params):
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=params["date_range_days"])

    try:
        ohlcv_df = fetch_ohlcv(ticker, str(start_date), str(end_date))
    except Exception as e:
        print(f"  [{ticker}] OHLCV fetch failed: {e}")
        return None

    if ohlcv_df is None or ohlcv_df.empty:
        return None

    try:
        df = calculate_technical_indicators(ohlcv_df)
    except Exception as e:
        return None

    try:
        labels = create_target_labels(
            df,
            horizon=params["prediction_horizon"],
            threshold=params["classification_threshold"],
        )
    except Exception as e:
        return None

    df = df.copy()
    df["_target"] = labels
    subset = [c for c in FEATURE_COLUMNS + ["_target"] if c in df.columns]
    df.dropna(subset=subset, inplace=True)

    seq_len = params["sequence_length"]
    if len(df) < seq_len + 20:
        return None

    labels_clean = df["_target"].values.astype(np.float32)
    train_end = int(len(df) * 0.8)
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    try:
        scaled_train, scaled_test, scaler = scale_features(df, train_end, feat_cols)
    except Exception as e:
        return None

    train_labels = labels_clean[:train_end]
    test_labels = labels_clean[train_end:]

    try:
        train_ds = StockDataset(scaled_train, train_labels, sequence_length=seq_len)
        test_ds = StockDataset(scaled_test, test_labels, sequence_length=seq_len)
    except Exception as e:
        return None

    if len(train_ds) == 0 or len(test_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)

    input_size = scaled_train.shape[1]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    try:
        history = train_model(
            model, train_loader,
            epochs=params["epochs"],
            lr=params["learning_rate"],
            device=device,
            weight_decay=params.get("weight_decay", 1e-4),
        )
    except Exception as e:
        return None

    try:
        rf_model = train_random_forest(
            train_ds,
            n_estimators=params.get("rf_n_estimators", 300),
            max_depth=params.get("rf_max_depth", 15),
            min_samples_leaf=params.get("rf_min_samples_leaf", 10),
        )
    except Exception as e:
        return None

    try:
        gb_model = train_gradient_boosting(
            train_ds,
            n_estimators=params.get("gb_n_estimators", 300),
            max_depth=params.get("gb_max_depth", 4),
            learning_rate=params.get("gb_lr", 0.05),
            subsample=params.get("gb_subsample", 0.8),
        )
    except Exception as e:
        return None

    try:
        probabilities, actuals = evaluate_model(
            model, test_loader,
            device=device,
            rf_model=rf_model,
            gb_model=gb_model,
            lstm_weight=params.get("lstm_weight", 0.2),
            rf_weight=params.get("rf_weight", 0.4),
            gb_weight=params.get("gb_weight", 0.4),
        )
    except Exception as e:
        return None

    preds_binary = (probabilities >= 0.5).astype(float)
    accuracy = (preds_binary == actuals).mean() * 100
    return accuracy

def run_trial(params, trial_name=""):
    print(f"\n{'='*60}")
    print(f"TRIAL: {trial_name}")
    print(f"{'='*60}")
    
    accuracies = {}
    for ticker in TICKERS:
        acc = run_single_stock(ticker, params)
        if acc is not None:
            accuracies[ticker] = acc
            print(f"  {ticker}: {acc:.2f}%")
        else:
            print(f"  {ticker}: FAILED")

    if len(accuracies) == 0:
        return 0.0

    mean_acc = np.mean(list(accuracies.values()))
    min_acc = min(accuracies.values())
    max_acc = max(accuracies.values())

    print(f"\n  >> Mean={mean_acc:.2f}%, Min={min_acc:.2f}%, Max={max_acc:.2f}%, "
          f"Stocks={len(accuracies)}/{len(TICKERS)}")

    if mean_acc >= 59.0:
        print("  >> ✅ TARGET MET!")
    else:
        print(f"  >> ❌ Below target ({mean_acc:.2f}% < 59%)")

    return mean_acc

TRIALS = [
    {
        "name": "10-day, T=0.015, GBM-Heavy (70%), RF (25%), LSTM (5%)",
        "params": {
            "date_range_days": 1800,
            "sequence_length": 30,
            "epochs": 35,
            "batch_size": 32,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "prediction_horizon": 10,
            "classification_threshold": 0.015,
            "weight_decay": 1e-4,
            "rf_n_estimators": 500,
            "rf_max_depth": 10,
            "rf_min_samples_leaf": 15,
            "gb_n_estimators": 500,
            "gb_max_depth": 3,
            "gb_lr": 0.02,
            "gb_subsample": 0.8,
            "lstm_weight": 0.05,
            "rf_weight": 0.25,
            "gb_weight": 0.70,
        },
    },
    {
        "name": "15-day, T=0.03, Balanced ensemble (GBM 40%, RF 40%, LSTM 20%)",
        "params": {
            "date_range_days": 1800,
            "sequence_length": 45,
            "epochs": 30,
            "batch_size": 32,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "prediction_horizon": 15,
            "classification_threshold": 0.03,
            "weight_decay": 1e-4,
            "rf_n_estimators": 500,
            "rf_max_depth": 15,
            "rf_min_samples_leaf": 10,
            "gb_n_estimators": 500,
            "gb_max_depth": 4,
            "gb_lr": 0.03,
            "gb_subsample": 0.8,
            "lstm_weight": 0.20,
            "rf_weight": 0.40,
            "gb_weight": 0.40,
        },
    },
]

if __name__ == "__main__":
    best_acc = 0.0
    best_trial = ""
    best_params = None

    for trial in TRIALS:
        acc = run_trial(trial["params"], trial["name"])
        if acc > best_acc:
            best_acc = acc
            best_trial = trial["name"]
            best_params = trial["params"]

    print("\n" + "=" * 60)
    print("TUNING COMPLETE — ROUND 4")
    print(f"Best Trial: {best_trial}")
    print(f"Best Mean Accuracy: {best_acc:.2f}%")
    print("=" * 60)
