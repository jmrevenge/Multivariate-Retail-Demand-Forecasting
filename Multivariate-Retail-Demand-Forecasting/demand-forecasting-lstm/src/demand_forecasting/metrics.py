"""Forecast accuracy metrics. All operate on raw (unscaled) demand units."""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true) + np.abs(y_pred), 1e-6, None)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
    }


def improvement(baseline: float, candidate: float) -> float:
    """Percent error reduction of candidate vs baseline (positive is better)."""
    if baseline == 0:
        return 0.0
    return round((baseline - candidate) / baseline * 100.0, 2)
