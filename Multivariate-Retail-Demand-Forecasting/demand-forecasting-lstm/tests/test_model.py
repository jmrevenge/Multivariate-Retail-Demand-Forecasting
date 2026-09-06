import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting.models import LSTMForecaster


def test_lstm_output_shape():
    model = LSTMForecaster(n_features=9, horizon=7)
    x = torch.zeros(5, 28, 9)
    out = model(x)
    assert out.shape == (5, 7)


def test_lstm_overfits_tiny_batch():
    """A functioning model must be able to drive loss down on a tiny fixed batch."""
    torch.manual_seed(0)
    model = LSTMForecaster(n_features=9, horizon=7, hidden_size=32, num_layers=1, dropout=0.0)
    x = torch.randn(8, 28, 9)
    y = torch.randn(8, 7)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    first = loss_fn(model(x), y).item()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5  # loss at least halved


def test_full_pipeline_smoke_is_fast():
    """End-to-end train on a tiny config passed directly (no env, no reload)."""
    from dataclasses import replace

    from demand_forecasting.config import settings
    from demand_forecasting.train import build_datasets, lstm_predict, train_lstm

    tiny = replace(settings, n_days=300, epochs=2, n_stores=1, n_products=2)
    ds = build_datasets(tiny)
    model = train_lstm(ds, tiny)
    pred = lstm_predict(model, ds.test[0], ds.scaler)
    assert pred.shape == ds.test_raw[1].shape
    assert pred.size > 0
    assert np.isfinite(pred).all()
