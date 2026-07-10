"""
models.py — ML/Trading Engine: PyTorch LSTM model for stock price movement prediction.

Provides:
    - StockDataset: Sliding-window time-series dataset for LSTM consumption.
    - LSTMModel: Multi-layer LSTM with dropout for binary classification.
    - train_model: Full training loop with validation, BCELoss + Adam.
    - evaluate_model: Inference routine returning probability and ground-truth arrays.

All interfaces are designed to integrate with the broader stock_analyzer_project pipeline,
consuming scaled feature arrays from features.py and producing predictions consumed by backtest.py.
"""

import logging
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

# --- STREAMLIT + PYTORCH BUG FIX ---
# Streamlit's file watcher incorrectly triggers __getattr__ on torch._classes
# when looking for __path__._path. Setting it to an empty list bypasses this.
try:
    torch._classes.__path__ = []
except Exception:
    pass
# -----------------------------------
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StockDataset(Dataset):
    """Sliding-window dataset that converts flat feature/label arrays into
    sequences suitable for recurrent models.

    For each valid index *i*, the dataset yields:
        X — a tensor of shape ``(sequence_length, num_features)`` representing
            the feature window ``features[i : i + sequence_length]``.
        y — a scalar tensor holding ``labels[i + sequence_length]``, i.e. the
            target that immediately follows the look-back window.

    This means there are ``len(labels) - sequence_length`` valid samples.

    Args:
        features: 2-D array of shape ``(num_samples, num_features)`` with the
            scaled indicator values produced by ``features.scale_features``.
        labels: 1-D array of shape ``(num_samples,)`` with binary targets
            (1 = price went up, 0 = price went down).
        sequence_length: Number of historical time-steps in each look-back
            window fed to the LSTM.  Defaults to ``60``.

    Raises:
        ValueError: If *features* and *labels* have incompatible first
            dimensions, or if *sequence_length* is not positive.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 60,
    ) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features and labels must have the same number of samples, "
                f"got {features.shape[0]} and {labels.shape[0]}"
            )
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if features.shape[0] <= sequence_length:
            raise ValueError(
                f"Not enough samples ({features.shape[0]}) for sequence_length "
                f"({sequence_length}). Need at least {sequence_length + 1} samples."
            )

        self.features: np.ndarray = features.astype(np.float32)
        self.labels: np.ndarray = labels.astype(np.float32)
        self.sequence_length: int = sequence_length

        logger.debug(
            "StockDataset created — samples=%d, features=%d, seq_len=%d, usable=%d",
            features.shape[0],
            features.shape[1],
            sequence_length,
            len(self),
        )

    def __len__(self) -> int:
        """Return the number of usable (window, target) pairs."""
        return max(0, len(self.labels) - self.sequence_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the *idx*-th sample.

        Args:
            idx: Index into the usable range ``[0, len(self))``.

        Returns:
            A tuple ``(X, y)`` where *X* has shape
            ``(sequence_length, num_features)`` and *y* is a scalar tensor.
        """
        X = torch.tensor(
            self.features[idx : idx + self.sequence_length],
            dtype=torch.float32,
        )  # (seq_len, num_features)
        y = torch.tensor(
            self.labels[idx + self.sequence_length],
            dtype=torch.float32,
        )  # scalar
        return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Multi-layer LSTM classifier for binary stock-movement prediction.

    Architecture
    ------------
    1. ``nn.LSTM`` with *num_layers* stacked layers (``batch_first=True``).
       Inter-layer dropout is applied when ``num_layers > 1``.
    2. ``nn.Dropout`` on the last time-step hidden state.
    3. ``nn.Linear`` projection to *output_size* logits.
    4. ``torch.sigmoid`` activation (output in ``[0, 1]``).

    Args:
        input_size: Number of features per time-step.
        hidden_size: Dimensionality of the LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between LSTM layers and before
            the fully-connected head.
        output_size: Number of output neurons (``1`` for binary classification).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        # LSTM inter-layer dropout is only valid when num_layers > 1.
        lstm_dropout: float = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        logger.info(
            "LSTMModel initialised — input=%d, hidden=%d, layers=%d, dropout=%.2f",
            input_size,
            hidden_size,
            num_layers,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, sequence_length, input_size)``.

        Returns:
            Sigmoid probabilities of shape ``(batch, 1)``.
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the output of the *last* time-step.
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        out = self.dropout(last_hidden)
        out = self.fc(out)  # (batch, output_size)
        out = torch.sigmoid(out)  # (batch, output_size)
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 15,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """Train the model using Binary Cross-Entropy loss and the Adam optimiser.

    Args:
        model: An ``nn.Module`` (typically ``LSTMModel``) to train.
        train_loader: ``DataLoader`` yielding ``(X, y)`` training batches.
        val_loader: Optional ``DataLoader`` for validation.  When provided,
            validation loss is computed at the end of every epoch.
        epochs: Number of full passes through *train_loader*.
        lr: Learning rate for the Adam optimiser.
        weight_decay: L2 regularization penalty to prevent overfitting.
        early_stopping_patience: Epochs to wait for val_loss improvement before stopping.
        device: PyTorch device string (``'cpu'`` or ``'cuda'``).

    Returns:
        A dict with keys ``'train_losses'`` and ``'val_losses'``, each
        containing a list of per-epoch mean losses.  ``'val_losses'`` is
        empty if *val_loader* is ``None``.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: List[float] = []
    val_losses: List[float] = []

    logger.info(
        "Training started — epochs=%d, lr=%.5f, wd=%s, device=%s, "
        "train_batches=%d, val_batches=%s",
        epochs,
        lr,
        weight_decay,
        device,
        len(train_loader),
        len(val_loader) if val_loader is not None else "N/A",
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # ----- Training phase -----
        model.train()
        epoch_loss: float = 0.0
        num_batches: int = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)  # (batch, seq_len, features)
            batch_y = batch_y.to(device)  # (batch,)

            optimiser.zero_grad()
            predictions = model(batch_X).squeeze(-1)  # (batch,)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            num_batches += 1

        mean_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(mean_train_loss)

        # ----- Validation phase -----
        mean_val_loss: Optional[float] = None
        if val_loader is not None:
            model.eval()
            val_epoch_loss: float = 0.0
            val_batches: int = 0

            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X = val_X.to(device)
                    val_y = val_y.to(device)

                    val_preds = model(val_X).squeeze(-1)
                    v_loss = criterion(val_preds, val_y)
                    val_epoch_loss += v_loss.item()
                    val_batches += 1

            mean_val_loss = val_epoch_loss / max(val_batches, 1)
            val_losses.append(mean_val_loss)

        # ----- Logging -----
        if mean_val_loss is not None:
            logger.info(
                "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                epoch,
                epochs,
                mean_train_loss,
                mean_val_loss,
            )
        else:
            logger.info(
                "Epoch %d/%d — train_loss=%.6f",
                epoch,
                epochs,
                mean_train_loss,
            )

        # ----- Early Stopping -----
        if mean_val_loss is not None:
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d. Best val_loss: %.6f", epoch, best_val_loss)
                    break

    # Restore best weights if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model weights with val_loss=%.6f", best_val_loss)

    logger.info("Training complete.")
    return {"train_losses": train_losses, "val_losses": val_losses}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    rf_model: Optional[RandomForestClassifier] = None,
    rf_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a test set and collect predictions vs ground truth.

    Args:
        model: A trained ``nn.Module``.
        test_loader: ``DataLoader`` yielding ``(X, y)`` test batches.
        device: PyTorch device string.
        rf_model: Optional trained RandomForestClassifier for ensembling.
        rf_weight: Weight applied to the RF model's probabilities (0.0 to 1.0).

    Returns:
        A tuple ``(probabilities, actuals)`` where both are 1-D NumPy arrays.
        *probabilities* contains the raw sigmoid outputs and *actuals*
        contains the binary ground-truth labels.
    """
    model = model.to(device)
    model.eval()

    all_probs: List[np.ndarray] = []
    all_actuals: List[np.ndarray] = []

    logger.info("Evaluation started — test_batches=%d", len(test_loader))

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            lstm_preds = model(batch_X).squeeze(-1).cpu().numpy()  # (batch,)
            
            # Incorporate Random Forest ensemble if provided
            if rf_model is not None:
                # Extract the last timestep feature vector for RF
                X_rf = batch_X[:, -1, :].cpu().numpy()
                rf_preds = rf_model.predict_proba(X_rf)[:, 1]
                preds = (lstm_preds * (1.0 - rf_weight)) + (rf_preds * rf_weight)
            else:
                preds = lstm_preds

            all_probs.append(preds)
            all_actuals.append(batch_y.numpy())

    probabilities = np.concatenate(all_probs)    # 1-D
    actuals = np.concatenate(all_actuals)        # 1-D

    logger.info("Evaluation complete — samples=%d, mean_prob=%.4f", len(probabilities), probabilities.mean())
    return probabilities, actuals


def train_random_forest(
    dataset: StockDataset,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """Train a Random Forest classifier using the dataset's tabular features.
    
    Extracts the features corresponding to the end of each LSTM sequence window.
    """
    valid_len = len(dataset)
    X = dataset.features[dataset.sequence_length : dataset.sequence_length + valid_len]
    y = dataset.labels[dataset.sequence_length : dataset.sequence_length + valid_len]
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    logger.info("Training Random Forest ensemble model (n_estimators=%d)...", n_estimators)
    rf.fit(X, y)
    logger.info("Random Forest training complete.")
    return rf