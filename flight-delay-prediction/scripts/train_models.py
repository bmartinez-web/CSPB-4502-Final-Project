"""
Train calibrated classifiers for delay and cancellation targets.
"""
from __future__ import annotations
import argparse
from src.train import run_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/flights_joined.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    # Delay
    run_pipeline(args.data, target="delay15", out_dir=args.out, calibrate=not args.no_calibrate)
    # Cancellation
    run_pipeline(args.data, target="cancel", out_dir=args.out, calibrate=not args.no_calibrate)
