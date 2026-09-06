"""Synthetic multivariate demand generator and supervised windowing.

The generator is intentionally *realistic*, not tuned to hit a target metric. Demand for
each (store, product) series is built from additive and multiplicative components with
several deliberately NONLINEAR effects, so that a model able to learn interactions from
feature history (the LSTM) has a genuine, defensible edge over a linear model fitted on
the same features:

  * price elasticity is multiplicative in log space (constant-elasticity, so nonlinear
    in raw price), with promotions cutting price on scattered days;
  * a temperature driver affects demand through a threshold / quadratic response, not a
    straight line;
  * promotion uplift interacts with the weekend (bigger lift on weekends);
  * weekly and yearly seasonality plus a slow random-walk trend.

All randomness flows from a single seed, so a given Settings produces identical data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings

# Feature columns in the order the model receives them. `demand` is the target and is
# also fed back as an autoregressive input feature.
FEATURES = [
    "demand",
    "price",
    "on_promo",
    "temperature",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
]
TARGET_INDEX = 0


@dataclass
class Panel:
    """A panel of demand series. `array` has shape (n_series, n_days, n_features)."""

    array: np.ndarray
    series_ids: list[tuple[int, int]]
    feature_names: list[str]

    @property
    def n_series(self) -> int:
        return self.array.shape[0]

    @property
    def n_days(self) -> int:
        return self.array.shape[1]


def generate_panel(cfg=settings) -> Panel:
    rng = np.random.default_rng(cfg.seed)
    days = np.arange(cfg.n_days)
    dow = days % 7
    doy = days % 365

    dow_sin, dow_cos = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    doy_sin, doy_cos = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    is_weekend = (dow >= 5).astype(float)

    # A shared temperature signal (seasonal + noise), reused across series.
    temp = 15 + 12 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2.5, cfg.n_days)

    series_arrays = []
    series_ids = []
    for store in range(cfg.n_stores):
        for product in range(cfg.n_products):
            base = rng.uniform(40, 120)
            elasticity = rng.uniform(1.1, 2.0)          # price sensitivity
            temp_sensitivity = rng.uniform(0.4, 1.2)    # how strongly temp matters
            temp_threshold = rng.uniform(18, 24)

            # price: a base level with scattered multi-day promotions cutting it.
            price = np.full(cfg.n_days, rng.uniform(8, 20))
            on_promo = np.zeros(cfg.n_days)
            d = 0
            while d < cfg.n_days:
                if rng.random() < 0.03:  # promo starts
                    length = rng.integers(2, 6)
                    price[d:d + length] *= rng.uniform(0.6, 0.8)
                    on_promo[d:d + length] = 1.0
                    d += length
                else:
                    d += 1

            # slow random-walk trend (demand grows/shrinks gradually)
            trend = np.cumsum(rng.normal(0, 0.002, cfg.n_days))

            # weekly + yearly seasonality
            weekly = 0.15 * np.sin(2 * np.pi * dow / 7) + 0.10 * is_weekend
            yearly = 0.20 * np.sin(2 * np.pi * (doy - 330) / 365)  # holiday-season peak

            # NONLINEAR effects:
            # constant-elasticity price response (nonlinear in raw price)
            ref_price = price.mean()
            price_effect = -elasticity * np.log(price / ref_price)
            # threshold/quadratic temperature response
            over = np.maximum(temp - temp_threshold, 0.0)
            temp_effect = temp_sensitivity * (over / 10.0) ** 2
            # promo uplift that interacts with weekend
            promo_effect = on_promo * (0.25 + 0.20 * is_weekend)

            log_demand = (
                np.log(base)
                + trend
                + weekly
                + yearly
                + price_effect
                + temp_effect
                + promo_effect
            )
            noise = rng.normal(0, 0.06, cfg.n_days)
            demand = np.exp(log_demand + noise)

            feats = np.stack(
                [demand, price, on_promo, temp, dow_sin, dow_cos, doy_sin, doy_cos, is_weekend],
                axis=1,
            )
            series_arrays.append(feats)
            series_ids.append((store, product))

    return Panel(np.stack(series_arrays, axis=0), series_ids, FEATURES)


def temporal_split_indices(n_days: int, cfg=settings) -> tuple[int, int]:
    train_end = int(n_days * cfg.train_frac)
    val_end = int(n_days * (cfg.train_frac + cfg.val_frac))
    return train_end, val_end


def make_windows(
    panel: Panel, start: int, end: int, cfg=settings
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) supervised windows from day `start` up to `end`.

    X: (n_windows, lookback, n_features)
    y: (n_windows, horizon)   -- future demand for the target series
    """
    L, H = cfg.lookback, cfg.horizon
    xs, ys = [], []
    for s in range(panel.n_series):
        series = panel.array[s]
        # last input day must leave room for the horizon within [start, end)
        for t in range(start + L, end - H + 1):
            xs.append(series[t - L:t])
            ys.append(series[t:t + H, TARGET_INDEX])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
