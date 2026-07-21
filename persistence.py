"""
persistence.py — Save/load trained model artifacts for the Stock Analyzer.

Stores, per ticker, everything needed to run inference without retraining:
    saved_models/<TICKER>/
        lstm.pt        — LSTM state_dict + architecture config
        rf.joblib      — Random Forest model
        gb.joblib      — Gradient Boosting model
        scaler.joblib  — fitted MinMaxScaler
        meta.json      — params, training history, probability bounds,
                         feature columns, timestamp
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import joblib
import torch

from models import LSTMModel

logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


def save_artifacts(
    ticker: str,
    model: LSTMModel,
    rf_model,
    gb_model,
    scaler,
    params: dict,
    history: dict,
    prob_bounds: Tuple[float, float],
    feature_columns: list,
    base_dir: str = ARTIFACT_DIR,
) -> str:
    """Persist all trained artifacts for *ticker*. Returns the save path."""
    path = os.path.join(base_dir, ticker.upper())
    os.makedirs(path, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": model.lstm.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "dropout": model.dropout.p,
        },
        os.path.join(path, "lstm.pt"),
    )
    joblib.dump(rf_model, os.path.join(path, "rf.joblib"))
    joblib.dump(gb_model, os.path.join(path, "gb.joblib"))
    joblib.dump(scaler, os.path.join(path, "scaler.joblib"))

    meta = {
        "ticker": ticker.upper(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "params": {k: v for k, v in params.items()
                   if isinstance(v, (int, float, str, bool))},
        "history": history,
        "prob_bounds": list(prob_bounds),
        "feature_columns": list(feature_columns),
    }
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("Saved model artifacts for %s to %s", ticker.upper(), path)
    return path


def load_artifacts(ticker: str, base_dir: str = ARTIFACT_DIR) -> Optional[dict]:
    """Load saved artifacts for *ticker*.

    Returns a dict with keys ``model``, ``rf_model``, ``gb_model``,
    ``scaler``, ``meta`` — or ``None`` if no complete artifact set exists.
    """
    path = os.path.join(base_dir, ticker.upper())
    required = ["lstm.pt", "rf.joblib", "gb.joblib", "scaler.joblib", "meta.json"]
    if not all(os.path.exists(os.path.join(path, f)) for f in required):
        return None

    try:
        ckpt = torch.load(os.path.join(path, "lstm.pt"),
                          map_location="cpu", weights_only=False)
        model = LSTMModel(
            input_size=ckpt["input_size"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        rf_model = joblib.load(os.path.join(path, "rf.joblib"))
        gb_model = joblib.load(os.path.join(path, "gb.joblib"))
        scaler = joblib.load(os.path.join(path, "scaler.joblib"))
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        logger.error("Failed to load artifacts for %s: %s", ticker.upper(), exc)
        return None

    logger.info("Loaded saved artifacts for %s (saved_at=%s)",
                ticker.upper(), meta.get("saved_at"))
    return {
        "model": model,
        "rf_model": rf_model,
        "gb_model": gb_model,
        "scaler": scaler,
        "meta": meta,
    }
