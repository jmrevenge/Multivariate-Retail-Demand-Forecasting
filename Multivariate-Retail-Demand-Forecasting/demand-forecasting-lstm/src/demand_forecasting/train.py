"""Training pipeline: deterministic, leakage-safe scaling, early stopping.

Feature scaling statistics are computed on the training region of the timeline only,
then applied to val and test, so no future information leaks into the scaler. Demand is
predicted in standardized space and inverse-transformed to raw units before any metric
is computed, so every model (LSTM and baselines) is scored on the same raw scale.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import settings
from .data import TARGET_INDEX, Panel, generate_panel, make_windows, temporal_split_indices
from .models import LSTMForecaster


def set_seed(seed: int = settings.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, panel: Panel, train_end: int) -> "Scaler":
        region = panel.array[:, :train_end, :].reshape(-1, panel.array.shape[-1])
        mean = region.mean(axis=0)
        std = region.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform_panel(self, panel: Panel) -> Panel:
        scaled = (panel.array - self.mean) / self.std
        return Panel(scaled.astype(np.float32), panel.series_ids, panel.feature_names)

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.std[TARGET_INDEX] + self.mean[TARGET_INDEX]


@dataclass
class Datasets:
    panel_raw: Panel
    scaler: Scaler
    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    # raw (unscaled) windows for baselines that work in raw units
    train_raw: tuple[np.ndarray, np.ndarray]
    test_raw: tuple[np.ndarray, np.ndarray]


def build_datasets(cfg=settings) -> Datasets:
    panel = generate_panel(cfg)
    train_end, val_end = temporal_split_indices(panel.n_days, cfg)
    scaler = Scaler.fit(panel, train_end)
    scaled = scaler.transform_panel(panel)

    train = make_windows(scaled, 0, train_end, cfg)
    val = make_windows(scaled, train_end, val_end, cfg)
    test = make_windows(scaled, val_end, panel.n_days, cfg)
    train_raw = make_windows(panel, 0, train_end, cfg)
    test_raw = make_windows(panel, val_end, panel.n_days, cfg)
    return Datasets(panel, scaler, train, val, test, train_raw, test_raw)


def train_lstm(ds: Datasets, cfg=settings, verbose: bool = False) -> LSTMForecaster:
    set_seed(cfg.seed)
    n_features = ds.train[0].shape[-1]
    model = LSTMForecaster(n_features=n_features)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(ds.train[0])
    ytr = torch.from_numpy(ds.train[1])
    Xva = torch.from_numpy(ds.val[0])
    yva = torch.from_numpy(ds.val[1])
    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )

    best_val = float("inf")
    best_state = None
    stale = 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if verbose:
            print(f"epoch {epoch:02d}  val_mse={val_loss:.5f}  best={best_val:.5f}")
        if stale >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def lstm_predict(model: LSTMForecaster, X_scaled: np.ndarray, scaler: Scaler) -> np.ndarray:
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_scaled)).numpy()
    return scaler.inverse_target(pred_scaled)
