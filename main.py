#!/usr/bin/env python3
"""
Stock Market Analyzer - CLI Pipeline Orchestrator
==================================================
Runs the full end-to-end stock analysis pipeline:
  Data fetch → Feature engineering → 3-Model Ensemble training → Backtest evaluation

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
from models import (StockDataset, LSTMModel, train_model, evaluate_model,
                    train_random_forest, train_gradient_boosting,
                    compute_probability_bounds, normalize_probabilities)
from backtest import run_backtest, calculate_performance_metrics
from persistence import save_artifacts

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
# Feature columns used for model input — full 26-feature set
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "Signal_Line", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "Vol_20", "Momentum_10", "Daily_Returns", "Sector_Return",
    "ATR_14", "OBV", "ROC_10", "Stoch_K", "Stoch_D", "VWAP",
]


def _load_config() -> dict:
    """Load all configuration from .env / environment variables."""
    load_dotenv()
    cfg = {
        "ticker":                   os.getenv("TARGET_TICKER", "AAPL").upper(),
        "sec_email":                os.getenv("SEC_USER_AGENT_EMAIL", ""),
        "sequence_length":          int(os.getenv("LSTM_SEQUENCE_LENGTH", 30)),
        "prediction_horizon":       int(os.getenv("PREDICTION_HORIZON", 10)),
        "classification_threshold": float(os.getenv("CLASSIFICATION_THRESHOLD", 0.02)),
        "epochs":                   int(os.getenv("MODEL_EPOCHS", 30)),
        "batch_size":               int(os.getenv("MODEL_BATCH_SIZE", 32)),
        "hidden_size":              int(os.getenv("MODEL_HIDDEN_SIZE", 128)),
        "num_layers":               int(os.getenv("MODEL_NUM_LAYERS", 2)),
        "dropout":                  float(os.getenv("MODEL_DROPOUT", 0.15)),
        "learning_rate":            float(os.getenv("MODEL_LEARNING_RATE", 0.001)),
        "initial_capital":          float(os.getenv("INITIAL_CAPITAL", 10000)),
        "risk_free_rate":           float(os.getenv("RISK_FREE_RATE", 0.02)),
        "signal_threshold":         float(os.getenv("SIGNAL_THRESHOLD", 0.62)),
        # Ensemble weights
        "lstm_weight":              float(os.getenv("LSTM_WEIGHT", 0.1)),
        "rf_weight":                float(os.getenv("RF_WEIGHT", 0.3)),
        "gb_weight":                float(os.getenv("GB_WEIGHT", 0.6)),
        # Random Forest params
        "rf_n_estimators":          int(os.getenv("RF_N_ESTIMATORS", 500)),
        "rf_max_depth":             int(os.getenv("RF_MAX_DEPTH", 10)),
        "rf_min_samples_leaf":      int(os.getenv("RF_MIN_SAMPLES_LEAF", 15)),
        # Gradient Boosting params
        "gb_n_estimators":          int(os.getenv("GB_N_ESTIMATORS", 400)),
        "gb_max_depth":             int(os.getenv("GB_MAX_DEPTH", 3)),
        "gb_lr":                    float(os.getenv("GB_LR", 0.03)),
        "gb_subsample":             float(os.getenv("GB_SUBSAMPLE", 0.8)),
        # Transaction costs
        "commission_pct":           float(os.getenv("COMMISSION_PCT", 0.0005)),
        "slippage_pct":             float(os.getenv("SLIPPAGE_PCT", 0.0005)),
        # ATR risk management
        "use_atr_risk":             os.getenv("USE_ATR_RISK", "false").lower() in ("1", "true", "yes"),
        "atr_stop_mult":            float(os.getenv("ATR_STOP_MULT", 2.0)),
        "atr_tp_mult":              float(os.getenv("ATR_TP_MULT", 3.0)),
        "risk_per_trade":           float(os.getenv("RISK_PER_TRADE", 0.02)),
        # Model persistence
        "save_model":               os.getenv("SAVE_MODEL", "true").lower() in ("1", "true", "yes"),
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
    start_date = end_date - timedelta(days=1500)  # ~4 years of data

    logger.info(f"Ticker: {ticker}  |  Period: {start_date} → {end_date}")
    logger.info(f"Model: LSTM(hidden={cfg['hidden_size']}, layers={cfg['num_layers']}, "
                f"dropout={cfg['dropout']})  |  Epochs: {cfg['epochs']}  |  "
                f"Batch: {cfg['batch_size']}  |  Seq: {cfg['sequence_length']}")
    logger.info(f"Ensemble: LSTM={cfg['lstm_weight']}, RF={cfg['rf_weight']}, "
                f"GB={cfg['gb_weight']}  |  Signal threshold: {cfg['signal_threshold']}")

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

    if len(scaled_train) <= seq_len or len(scaled_test) <= seq_len:
        logger.error(
            f"Not enough data for sequence length {seq_len} "
            f"(train rows: {len(scaled_train)}, test rows: {len(scaled_test)}). "
            "Use a wider date range or a shorter LSTM_SEQUENCE_LENGTH."
        )
        return

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

    logger.info("Training LSTM model …")
    try:
        history = train_model(
            model, train_loader, val_loader=val_loader,
            epochs=cfg["epochs"], lr=cfg["learning_rate"], device=device,
            weight_decay=1e-4,
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        return

    final_train_loss = history["train_losses"][-1]
    logger.info(f"LSTM training complete. Final train loss: {final_train_loss:.5f}")

    # ------------------------------------------------------------------
    # 9.5. Train Random Forest Ensemble
    # ------------------------------------------------------------------
    logger.info("Training Random Forest ensemble …")
    try:
        rf_model = train_random_forest(
            train_dataset,
            n_estimators=cfg["rf_n_estimators"],
            max_depth=cfg["rf_max_depth"],
            min_samples_leaf=cfg["rf_min_samples_leaf"],
        )
    except Exception as exc:
        logger.error(f"Random Forest training failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 9.6. Train Gradient Boosting
    # ------------------------------------------------------------------
    logger.info("Training Gradient Boosting model …")
    try:
        gb_model = train_gradient_boosting(
            train_dataset,
            n_estimators=cfg["gb_n_estimators"],
            max_depth=cfg["gb_max_depth"],
            learning_rate=cfg["gb_lr"],
            subsample=cfg["gb_subsample"],
        )
    except Exception as exc:
        logger.error(f"Gradient Boosting training failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 10. Evaluate 3-Model Ensemble
    # ------------------------------------------------------------------
    logger.info("Evaluating 3-model ensemble on test set …")
    try:
        probabilities, actuals = evaluate_model(
            model, test_loader, device=device,
            rf_model=rf_model, gb_model=gb_model,
            lstm_weight=cfg["lstm_weight"],
            rf_weight=cfg["rf_weight"],
            gb_weight=cfg["gb_weight"],
        )
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        return

    # Normalize probabilities to [0, 1] using bounds fitted on TRAINING-set
    # predictions only — the ensemble's raw outputs cluster in a narrow range,
    # but deriving the bounds from the test set would be look-ahead bias.
    logger.info("Computing normalization bounds from training-set predictions …")
    train_eval_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=False)
    train_probs, _ = evaluate_model(
        model, train_eval_loader, device=device,
        rf_model=rf_model, gb_model=gb_model,
        lstm_weight=cfg["lstm_weight"],
        rf_weight=cfg["rf_weight"],
        gb_weight=cfg["gb_weight"],
    )
    prob_bounds = compute_probability_bounds(train_probs)
    logger.info(f"Raw test probs: min={probabilities.min():.4f}, "
                f"max={probabilities.max():.4f}, mean={probabilities.mean():.4f}  |  "
                f"train-based bounds: [{prob_bounds[0]:.4f}, {prob_bounds[1]:.4f}]")
    probabilities = normalize_probabilities(probabilities, prob_bounds)
    logger.info(f"Normalized probs: min={probabilities.min():.4f}, "
                f"max={probabilities.max():.4f}, mean={probabilities.mean():.4f}")

    predictions = (probabilities >= cfg["signal_threshold"]).astype(int)
    accuracy = (predictions == actuals).mean() * 100
    logger.info(f"Test accuracy: {accuracy:.2f}%")
    logger.info(f"Signals above threshold: {predictions.sum()} / {len(predictions)}")

    # Classification diagnostics — accuracy alone can hide a model that just
    # predicts the majority class.
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    precision = precision_score(actuals, predictions, zero_division=0) * 100
    recall = recall_score(actuals, predictions, zero_division=0) * 100
    f1 = f1_score(actuals, predictions, zero_division=0) * 100
    if len(np.unique(actuals)) > 1:
        auc = roc_auc_score(actuals, probabilities)
        logger.info(f"Precision: {precision:.2f}%  |  Recall: {recall:.2f}%  |  "
                    f"F1: {f1:.2f}%  |  ROC-AUC: {auc:.4f}")
    else:
        logger.info(f"Precision: {precision:.2f}%  |  Recall: {recall:.2f}%  |  "
                    f"F1: {f1:.2f}%  |  ROC-AUC: N/A (single-class test set)")

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
            trend_data=df,
            commission_pct=cfg["commission_pct"],
            slippage_pct=cfg["slippage_pct"],
            use_atr_risk=cfg["use_atr_risk"],
            atr_stop_mult=cfg["atr_stop_mult"],
            atr_tp_mult=cfg["atr_tp_mult"],
            risk_per_trade=cfg["risk_per_trade"],
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

    # ------------------------------------------------------------------
    # 13. Persist trained artifacts
    # ------------------------------------------------------------------
    if cfg["save_model"]:
        try:
            save_path = save_artifacts(
                ticker, model, rf_model, gb_model, scaler,
                params=cfg, history=history, prob_bounds=prob_bounds,
                feature_columns=feature_cols,
            )
            logger.info(f"Model artifacts saved to: {save_path}")
        except Exception as exc:
            logger.warning(f"Failed to save model artifacts: {exc}")

    logger.info("  🚀  PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()