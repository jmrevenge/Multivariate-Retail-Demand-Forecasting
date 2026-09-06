"""Command-line interface: train the model and print a forecast, or run the full
evaluation. Kept thin; the real logic lives in the library modules.

    demand-forecast train           # train and report test metrics
    demand-forecast eval            # full baseline comparison (same as eval/run_eval.py)
"""
from __future__ import annotations

import argparse

from .metrics import all_metrics
from .train import build_datasets, lstm_predict, train_lstm


def _train(args) -> int:
    ds = build_datasets()
    model = train_lstm(ds, verbose=args.verbose)
    pred = lstm_predict(model, ds.test[0], ds.scaler)
    metrics = all_metrics(ds.test_raw[1], pred)
    print("LSTM test metrics (raw units):")
    for k, v in metrics.items():
        print(f"  {k:<6} {v}")
    return 0


def _eval(args) -> int:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    import run_eval

    return run_eval.main(["--verbose"] if args.verbose else [])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="demand-forecast")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    sub.add_parser("eval")
    args = parser.parse_args(argv)

    if args.command == "train":
        return _train(args)
    if args.command == "eval":
        return _eval(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
