import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting.baselines import LinearForecaster, SeasonalNaiveForecaster
from demand_forecasting.metrics import improvement, mae, mape, rmse, smape


def test_metrics_zero_on_perfect_prediction():
    y = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_improvement_sign_and_magnitude():
    # candidate error 8 vs baseline 10 -> 20 percent reduction
    assert improvement(10.0, 8.0) == 20.0
    # worse candidate -> negative improvement
    assert improvement(10.0, 12.0) == -20.0


def test_seasonal_naive_uses_weekly_lag():
    # build a window where demand column repeats weekly; naive should recover it
    n, lookback, feats, horizon = 4, 14, 9, 7
    x = np.zeros((n, lookback, feats), dtype=np.float32)
    x[:, :, 0] = np.tile(np.arange(7), 2)  # 0..6 repeated
    preds = SeasonalNaiveForecaster(period=7, horizon=horizon).predict(x)
    assert preds.shape == (n, horizon)
    # first predicted day equals the value 7 steps back (the same weekday)
    assert np.allclose(preds[:, 0], x[:, -7, 0])


def test_linear_forecaster_fits_and_predicts_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 14, 9)).astype(np.float32)
    y = rng.normal(size=(50, 7)).astype(np.float32)
    model = LinearForecaster().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50, 7)
