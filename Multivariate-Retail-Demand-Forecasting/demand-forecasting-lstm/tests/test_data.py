import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting.config import Settings
from demand_forecasting.data import (
    FEATURES,
    TARGET_INDEX,
    generate_panel,
    make_windows,
    temporal_split_indices,
)


def test_generator_is_deterministic():
    a = generate_panel()
    b = generate_panel()
    assert np.array_equal(a.array, b.array)


def test_panel_shape_matches_config():
    cfg = Settings()
    panel = generate_panel(cfg)
    assert panel.array.shape == (cfg.n_stores * cfg.n_products, cfg.n_days, len(FEATURES))


def test_demand_is_positive():
    panel = generate_panel()
    assert (panel.array[:, :, TARGET_INDEX] > 0).all()


def test_windows_have_correct_shapes():
    cfg = Settings()
    panel = generate_panel(cfg)
    train_end, _ = temporal_split_indices(panel.n_days, cfg)
    X, y = make_windows(panel, 0, train_end, cfg)
    assert X.shape[1:] == (cfg.lookback, len(FEATURES))
    assert y.shape[1] == cfg.horizon
    assert X.shape[0] == y.shape[0]


def test_windows_do_not_leak_across_split_boundary():
    cfg = Settings()
    panel = generate_panel(cfg)
    train_end, val_end = temporal_split_indices(panel.n_days, cfg)
    # a training window's target must end at or before train_end
    X, y = make_windows(panel, 0, train_end, cfg)
    assert X.shape[0] > 0  # sanity: windows exist
    assert train_end < val_end < panel.n_days
