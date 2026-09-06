"""Baselines the LSTM must beat to justify its complexity.

* SeasonalNaiveForecaster: predict each future day as the demand from 7 days earlier
  (the same weekday). A strong, honest baseline for daily retail demand.
* LinearForecaster: ridge regression mapping the flattened lookback window to the
  horizon. Same information as the LSTM, but no capacity to model interactions or
  nonlinear responses across time. This is the fair "did the LSTM earn its keep" test.
"""
from __future__ import annotations

import numpy as np

from .config import settings
from .data import TARGET_INDEX


class SeasonalNaiveForecaster:
    def __init__(self, period: int = 7, horizon: int = settings.horizon):
        self.period = period
        self.horizon = horizon

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (n, lookback, n_features). Repeat the last `period` demand values forward.
        demand_hist = x[:, :, TARGET_INDEX]
        preds = np.empty((x.shape[0], self.horizon), dtype=np.float32)
        for h in range(self.horizon):
            preds[:, h] = demand_hist[:, -self.period + (h % self.period)]
        return preds


class LinearForecaster:
    """Ridge regression on the flattened window. Closed-form, deterministic."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearForecaster":
        n = x.shape[0]
        flat = x.reshape(n, -1)
        flat = np.column_stack([np.ones(n), flat])  # bias term
        d = flat.shape[1]
        reg = self.alpha * np.eye(d)
        reg[0, 0] = 0.0  # do not regularize the bias
        self.coef_ = np.linalg.solve(flat.T @ flat + reg, flat.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "call fit first"
        n = x.shape[0]
        flat = np.column_stack([np.ones(n), x.reshape(n, -1)])
        return (flat @ self.coef_).astype(np.float32)
