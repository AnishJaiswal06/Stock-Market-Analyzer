#!/usr/bin/env python3
"""
Stock Market Analyzer - CLI Pipeline Orchestrator
==================================================
Runs the full end-to-end stock analysis pipeline:
  Data fetch → Feature engineering → LSTM training → Backtest evaluation

Usage:
    python main.py
    (Configure via .env file or environment variables)
"""

import sys
import os

# Ensure project root is on sys.path so absolute imports work from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader

from data_pipeline import fetch_ohlcv, fetch_fundamentals, align_and_merge
from features import calculate_technical_indicators, create_target_labels, scale_features
from models import StockDataset, LSTMModel, train_model, evaluate_model
from backtest import run_backtest, calculate_performance_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used for model input (must match what features.py produces)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "Signal_Line", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "Vol_20", "Momentum_10", "Daily_Returns",
]


def _load_config() -> dict:
    """Load all configuration from .env / environment variables."""
    load_dotenv()
    cfg = {
        "ticker":                   os.getenv("TARGET_TICKER", "AAPL").upper(),
        "sec_email":                os.getenv("SEC_USER_AGENT_EMAIL", ""),
        "sequence_length":          int(os.getenv("LSTM_SEQUENCE_LENGTH", 60)),
        "prediction_horizon":       int(os.getenv("PREDICTION_HORIZON", 5)),
        "classification_threshold": float(os.getenv("CLASSIFICATION_THRESHOLD", 0.01)),
        "epochs":                   int(os.getenv("MODEL_EPOCHS", 15)),
        "batch_size":               int(os.getenv("MODEL_BATCH_SIZE", 64)),
        "hidden_size":              int(os.getenv("MODEL_HIDDEN_SIZE", 64)),
        "num_layers":               int(os.getenv("MODEL_NUM_LAYERS", 2)),
        "dropout":                  float(os.getenv("MODEL_DROPOUT", 0.2)),
        "learning_rate":            float(os.getenv("MODEL_LEARNING_RATE", 0.001)),
        "initial_capital":          float(os.getenv("INITIAL_CAPITAL", 10000)),
        "risk_free_rate":           float(os.getenv("RISK_FREE_RATE", 0.02)),
        "signal_threshold":         float(os.getenv("SIGNAL_THRESHOLD", 0.5)),
    }
    return cfg


def main() -> None:
    """Run the complete stock analysis pipeline."""
    logger.info("=" * 55)
    logger.info("  Stock Market Analyzer Pipeline  —  Starting")
    logger.info("=" * 55)

    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    cfg = _load_config()
    ticker = cfg["ticker"]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=900)  # ~2.5 years

    logger.info(f"Ticker: {ticker}  |  Period: {start_date} → {end_date}")
    logger.info(f"Model: LSTM(hidden={cfg['hidden_size']}, layers={cfg['num_layers']}, "
                f"dropout={cfg['dropout']})  |  Epochs: {cfg['epochs']}  |  "
                f"Batch: {cfg['batch_size']}  |  Seq: {cfg['sequence_length']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 2. Data Fetching
    # ------------------------------------------------------------------
    logger.info("Fetching OHLCV data …")
    ohlcv_df = fetch_ohlcv(ticker, start_date=start_date, end_date=end_date)
    if ohlcv_df is None or ohlcv_df.empty:
        logger.error("Failed to fetch OHLCV data. Aborting.")
        return

    logger.info(f"OHLCV rows: {len(ohlcv_df)}")

    logger.info("Fetching fundamental data …")
    fund_df = fetch_fundamentals(ticker, user_agent_email=cfg["sec_email"])
    if fund_df is not None:
        logger.info(f"Fundamentals rows: {len(fund_df)}")
    else:
        logger.warning("No fundamental data returned (may be an ETF/index).")

    # ------------------------------------------------------------------
    # 3. Merge & Feature Engineering
    # ------------------------------------------------------------------
    logger.info("Merging data sources …")
    try:
        merged_df = align_and_merge(ohlcv_df, fund_df)
    except Exception as exc:
        logger.error(f"Data merge failed: {exc}")
        return

    logger.info("Calculating technical indicators …")
    try:
        df = calculate_technical_indicators(merged_df)
    except Exception as exc:
        logger.error(f"Indicator calculation failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 4. Target Labels
    # ------------------------------------------------------------------
    logger.info("Creating target labels …")
    try:
        labels = create_target_labels(
            df,
            horizon=cfg["prediction_horizon"],
            threshold=cfg["classification_threshold"],
        )
    except Exception as exc:
        logger.error(f"Target label creation failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 5. Drop NaN rows (from indicators AND labels)
    # ------------------------------------------------------------------
    # Combine labels as a column so we can drop NaNs uniformly
    df = df.copy()
    df["_target"] = labels

    # Keep only rows where features AND label are all valid
    subset_cols = FEATURE_COLUMNS + ["_target"]
    valid_cols = [c for c in subset_cols if c in df.columns]
    df.dropna(subset=valid_cols, inplace=True)

    if len(df) < cfg["sequence_length"] + 20:
        logger.error(f"Not enough data after cleaning ({len(df)} rows). Aborting.")
        return

    labels_clean = df["_target"].values.astype(np.float32)
    logger.info(f"Clean dataset rows: {len(df)}")

    # ------------------------------------------------------------------
    # 6. Time-based Train / Test Split (80 / 20)
    # ------------------------------------------------------------------
    train_end_idx = int(len(df) * 0.8)
    logger.info(f"Train rows: {train_end_idx}  |  Test rows: {len(df) - train_end_idx}")

    # ------------------------------------------------------------------
    # 7. Scale Features
    # ------------------------------------------------------------------
    logger.info("Scaling features …")
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    try:
        scaled_train, scaled_test, scaler = scale_features(
            df, train_end_idx, feature_cols
        )
    except Exception as exc:
        logger.error(f"Feature scaling failed: {exc}")
        return

    train_labels = labels_clean[:train_end_idx]
    test_labels = labels_clean[train_end_idx:]

    # ------------------------------------------------------------------
    # 8. Create DataLoaders
    # ------------------------------------------------------------------
    logger.info("Building DataLoaders …")
    seq_len = cfg["sequence_length"]

    train_dataset = StockDataset(scaled_train, train_labels, sequence_length=seq_len)
    test_dataset = StockDataset(scaled_test, test_labels, sequence_length=seq_len)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Datasets are empty after sequencing. Need more data.")
        return

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    # Optional: split train into train + val (last 10 % of train)
    val_loader = None
    val_split = int(len(scaled_train) * 0.9)
    val_size = len(scaled_train) - val_split
    if val_size > seq_len + 1:
        try:
            val_dataset = StockDataset(
                scaled_train[val_split:], train_labels[val_split:], sequence_length=seq_len
            )
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
        except ValueError:
            logger.info("Not enough data for validation split — skipping.")

    # ------------------------------------------------------------------
    # 9. Train LSTM Model
    # ------------------------------------------------------------------
    logger.info("Initialising LSTM model …")
    input_size = scaled_train.shape[1]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    logger.info("Training model …")
    try:
        history = train_model(
            model, train_loader, val_loader=val_loader,
            epochs=cfg["epochs"], lr=cfg["learning_rate"], device=device,
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        return

    final_train_loss = history["train_losses"][-1]
    logger.info(f"Training complete. Final train loss: {final_train_loss:.5f}")

    # ------------------------------------------------------------------
    # 10. Evaluate Model
    # ------------------------------------------------------------------
    logger.info("Evaluating model on test set …")
    try:
        probabilities, actuals = evaluate_model(model, test_loader, device=device)
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        return

    predictions = (probabilities >= cfg["signal_threshold"]).astype(int)
    accuracy = (predictions == actuals).mean() * 100
    logger.info(f"Test accuracy: {accuracy:.2f}%")

    # ------------------------------------------------------------------
    # 11. Run Backtest
    # ------------------------------------------------------------------
    logger.info("Running backtest …")
    # Align close prices & dates with the predictions (which correspond to
    # the LAST len(probabilities) rows of the test portion)
    test_df = df.iloc[train_end_idx:]
    # predictions correspond to rows starting at seq_len in the test set
    bt_start = seq_len
    bt_close = test_df["Close"].iloc[bt_start: bt_start + len(probabilities)]
    bt_dates = test_df.index[bt_start: bt_start + len(probabilities)]

    try:
        portfolio_df = run_backtest(
            close_prices=bt_close,
            predictions=probabilities,
            dates=bt_dates,
            initial_capital=cfg["initial_capital"],
            signal_threshold=cfg["signal_threshold"],
        )
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 12. Performance Metrics & Output
    # ------------------------------------------------------------------
    metrics = calculate_performance_metrics(
        portfolio_df,
        initial_capital=cfg["initial_capital"],
        risk_free_rate=cfg["risk_free_rate"],
    )

    logger.info("")
    logger.info("=" * 55)
    logger.info("  📊  BACKTEST RESULTS")
    logger.info("=" * 55)
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key:.<35s} {val:>10.2f}")
        else:
            logger.info(f"  {key:.<35s} {val!s:>10}")
    logger.info("=" * 55)
    logger.info("  🚀  PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()