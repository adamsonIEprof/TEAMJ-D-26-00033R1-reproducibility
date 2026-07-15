from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from repro_common import set_global_seed


class SequenceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dense_size: int, num_layers: int, dropout: float):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, endpoints, sequence_length: int):
    xs, ys, kept = [], [], []
    for endpoint in np.asarray(endpoints, dtype=int):
        if endpoint < sequence_length - 1:
            continue
        xs.append(X[endpoint - sequence_length + 1 : endpoint + 1])
        ys.append(y[endpoint])
        kept.append(endpoint)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(kept, dtype=int),
    )


def fit_predict_lstm(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_endpoints,
    predict_endpoints,
    cfg: dict,
):
    preprocessing_start = time.perf_counter()
    seed = int(cfg["random_seed"])
    set_global_seed(seed)
    train_endpoints = np.asarray(train_endpoints, dtype=int)
    predict_endpoints = np.asarray(predict_endpoints, dtype=int)
    max_train_endpoint = int(train_endpoints.max())

    x_scaler = StandardScaler().fit(X_all[: max_train_endpoint + 1])
    y_scaler = StandardScaler().fit(y_all[train_endpoints].reshape(-1, 1))
    X_scaled = x_scaler.transform(X_all)
    y_scaled = y_scaler.transform(y_all.reshape(-1, 1)).ravel()

    X_train, y_train, _ = make_sequences(
        X_scaled, y_scaled, train_endpoints, int(cfg["sequence_length"])
    )
    X_pred, _, kept_pred = make_sequences(
        X_scaled, y_scaled, predict_endpoints, int(cfg["sequence_length"])
    )
    if len(X_train) < 5:
        raise ValueError("Insufficient LSTM sequence samples in the training window.")

    internal_fraction = float(cfg["internal_validation_fraction"])
    n_internal = max(1, int(round(len(X_train) * internal_fraction)))
    n_internal = min(n_internal, max(1, len(X_train) - 2))
    X_fit, X_val = X_train[:-n_internal], X_train[-n_internal:]
    y_fit, y_val = y_train[:-n_internal], y_train[-n_internal:]

    model = SequenceLSTM(
        input_size=X_all.shape[1],
        hidden_size=int(cfg["hidden_size"]),
        dense_size=int(cfg["dense_size"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.L1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit)),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    best_state = None
    best_loss = np.inf
    bad_epochs = 0
    epochs_run = 0
    training_start = time.perf_counter()
    for epoch in range(int(cfg["max_epochs"])):
        epochs_run = epoch + 1
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(torch.from_numpy(X_val)), torch.from_numpy(y_val)).item())
        if val_loss < best_loss - float(cfg["early_stopping_min_delta"]):
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["early_stopping_patience"]):
                break
    training_seconds = time.perf_counter() - training_start

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_pred)).cpu().numpy()
    inference_seconds = time.perf_counter() - inference_start
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return np.clip(pred, 0, None), kept_pred, {
        "training_sequence_samples": int(len(X_train)),
        "prediction_sequence_samples": int(len(X_pred)),
        "epochs_run": int(epochs_run),
        "best_internal_validation_mae_scaled": float(best_loss),
        "preprocessing_seconds": float(preprocessing_seconds),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
    }
