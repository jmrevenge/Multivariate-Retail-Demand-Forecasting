"""Multivariate demand forecasting with LSTMs, benchmarked against strong baselines."""
from .config import settings
from .data import FEATURES, generate_panel
from .metrics import all_metrics, improvement
from .models import LSTMForecaster

__all__ = [
    "settings",
    "generate_panel",
    "FEATURES",
    "LSTMForecaster",
    "all_metrics",
    "improvement",
]
__version__ = "0.1.0"
