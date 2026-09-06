"""The LSTM sequence-to-vector forecaster.

Reads a `lookback`-day window of all features and predicts the next `horizon` days of
demand in one shot (direct multi-horizon forecasting). Direct multi-step avoids the
error-compounding of feeding predictions back in during inference.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import settings


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int = settings.horizon,
        hidden_size: int = settings.hidden_size,
        num_layers: int = settings.num_layers,
        dropout: float = settings.dropout,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # final hidden state summarizes the window
        return self.head(last)        # (batch, horizon)
