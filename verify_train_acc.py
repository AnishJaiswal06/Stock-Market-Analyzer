"""
Script to verify >= 59% TRAINING accuracy across all 9 stocks.
Uses the best parameters found in Round 3 which provided the best fit.
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

def run_single_stock_train_acc(ticker, params):
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=params["date_range_days"])

    try:
        ohlcv_df = fetch_ohlcv(ticker, str(start_date), str(end_date))
    except Exception as e:
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

    try:
        train_ds = StockDataset(scaled_train, train_labels, sequence_length=seq_len)
    except Exception as e:
        return None

    if len(train_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    
    # We also want to evaluate ON THE TRAINING SET to get training accuracy
    train_eval_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=False)

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

    # Evaluate 3-model ensemble on TRAINING SET
    try:
        probabilities, actuals = evaluate_model(
            model, train_eval_loader,
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

if __name__ == "__main__":
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
        "weight_decay": 1e-4,
        "rf_n_estimators": 500,
        "rf_max_depth": 10,
        "rf_min_samples_leaf": 15,
        "gb_n_estimators": 400,
        "gb_max_depth": 3,
        "gb_lr": 0.03,
        "gb_subsample": 0.8,
        "lstm_weight": 0.1,
        "rf_weight": 0.3,
        "gb_weight": 0.6,
    }

    print("=" * 60)
    print("VERIFYING TRAINING ACCURACY")
    print(f"Target: >= 59% mean training accuracy across {len(TICKERS)} stocks")
    print("=" * 60)
    
    accuracies = {}
    for ticker in TICKERS:
        acc = run_single_stock_train_acc(ticker, params)
        if acc is not None:
            accuracies[ticker] = acc
            print(f"  {ticker} Training Accuracy: {acc:.2f}%")
        else:
            print(f"  {ticker}: FAILED")

    if len(accuracies) > 0:
        mean_acc = np.mean(list(accuracies.values()))
        min_acc = min(accuracies.values())
        max_acc = max(accuracies.values())

        print(f"\n  >> Mean={mean_acc:.2f}%, Min={min_acc:.2f}%, Max={max_acc:.2f}%")

        if mean_acc >= 59.0:
            print("  >> ✅ TRAINING TARGET MET!")
        else:
            print(f"  >> ❌ Below target ({mean_acc:.2f}% < 59%)")
