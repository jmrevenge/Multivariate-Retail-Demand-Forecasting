"""Central configuration. Every value is overridable via environment variable so the
same code runs in a fast CI smoke test and a fuller local training run."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # data generation
    n_stores: int = _int("DF_N_STORES", 3)
    n_products: int = _int("DF_N_PRODUCTS", 4)
    n_days: int = _int("DF_N_DAYS", 1096)  # ~3 years
    seed: int = _int("DF_SEED", 7)

    # windowing
    lookback: int = _int("DF_LOOKBACK", 28)   # days of history fed to the model
    horizon: int = _int("DF_HORIZON", 7)      # days to forecast ahead

    # model
    hidden_size: int = _int("DF_HIDDEN", 64)
    num_layers: int = _int("DF_LAYERS", 2)
    dropout: float = _float("DF_DROPOUT", 0.2)

    # training
    epochs: int = _int("DF_EPOCHS", 40)
    batch_size: int = _int("DF_BATCH", 64)
    lr: float = _float("DF_LR", 1e-3)
    patience: int = _int("DF_PATIENCE", 6)     # early-stopping patience

    # temporal split (fractions of the timeline)
    train_frac: float = _float("DF_TRAIN_FRAC", 0.7)
    val_frac: float = _float("DF_VAL_FRAC", 0.15)


settings = Settings()
