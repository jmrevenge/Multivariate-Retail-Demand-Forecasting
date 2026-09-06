"""Train the LSTM and baselines on identical windows and compare on the held-out test
period. Prints a metrics table and the LSTM's error reduction over each baseline, and
optionally writes results JSON.

    python eval/run_eval.py [--json results.json] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecasting.baselines import LinearForecaster, SeasonalNaiveForecaster  # noqa: E402
from demand_forecasting.config import settings  # noqa: E402
from demand_forecasting.metrics import all_metrics, improvement  # noqa: E402
from demand_forecasting.train import build_datasets, lstm_predict, train_lstm  # noqa: E402


def run(verbose: bool = False) -> dict:
    ds = build_datasets()
    y_true = ds.test_raw[1]  # raw-unit ground truth for the test horizon

    # Seasonal naive works directly on raw demand history.
    naive = SeasonalNaiveForecaster()
    naive_pred = naive.predict(ds.test_raw[0])

    # Linear regression on the same scaled windows the LSTM sees, then inverse-transform.
    linear = LinearForecaster().fit(*ds.train)
    linear_pred = ds.scaler.inverse_target(linear.predict(ds.test[0]))

    t0 = time.perf_counter()
    model = train_lstm(ds, verbose=verbose)
    train_secs = time.perf_counter() - t0
    lstm_pred = lstm_predict(model, ds.test[0], ds.scaler)

    results = {
        "seasonal_naive": all_metrics(y_true, naive_pred),
        "linear_regression": all_metrics(y_true, linear_pred),
        "lstm": all_metrics(y_true, lstm_pred),
    }
    results["improvement_vs_naive_MAE"] = improvement(
        results["seasonal_naive"]["MAE"], results["lstm"]["MAE"]
    )
    results["improvement_vs_linear_MAE"] = improvement(
        results["linear_regression"]["MAE"], results["lstm"]["MAE"]
    )
    results["improvement_vs_linear_RMSE"] = improvement(
        results["linear_regression"]["RMSE"], results["lstm"]["RMSE"]
    )
    results["meta"] = {
        "n_test_windows": int(y_true.shape[0]),
        "horizon": settings.horizon,
        "lookback": settings.lookback,
        "train_seconds": round(train_secs, 1),
        "seed": settings.seed,
    }
    return results


def _fmt(name: str, m: dict) -> str:
    return f"{name:<20}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}{m['MAPE']:>9.2f}%{m['sMAPE']:>9.2f}%"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    r = run(verbose=a.verbose)
    print("=" * 62)
    print("Multivariate Demand Forecasting — test-set results (raw units)")
    print("=" * 62)
    print(f"{'model':<20}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}{'sMAPE':>10}")
    print("-" * 62)
    print(_fmt("Seasonal naive", r["seasonal_naive"]))
    print(_fmt("Linear regression", r["linear_regression"]))
    print(_fmt("LSTM", r["lstm"]))
    print("-" * 62)
    print(f"LSTM MAE reduction vs seasonal naive : {r['improvement_vs_naive_MAE']:.1f}%")
    print(f"LSTM MAE reduction vs linear regr.   : {r['improvement_vs_linear_MAE']:.1f}%")
    print(f"LSTM RMSE reduction vs linear regr.  : {r['improvement_vs_linear_RMSE']:.1f}%")
    print(f"(trained in {r['meta']['train_seconds']}s on CPU, "
          f"{r['meta']['n_test_windows']} test windows)")
    print("=" * 62)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
