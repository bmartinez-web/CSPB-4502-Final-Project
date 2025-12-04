"""
Generate a tiny synthetic dataset to test the pipeline end-to-end.
This is NOT the BTS data—it's just for sanity checks.
"""
from __future__ import annotations
import os, argparse
import pandas as pd
import numpy as np

def main(out_csv: str, n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="H")
    idx = rng.integers(0, len(dates), size=n)
    fl_date = pd.to_datetime(dates[idx]).normalize()
    dep_hour = pd.to_datetime(dates[idx]).hour

    carriers = ["AA","DL","UA","WN","AS","B6"]
    airports = ["DEN","DFW","ATL","ORD","SFO","LAX","SEA","PHX","IAH","EWR"]
    rows = []
    for i in range(n):
        c = rng.choice(carriers)
        o = rng.choice(airports)
        d = rng.choice([a for a in airports if a != o])
        dist = max(100, rng.normal(900, 400))
        vis = max(0.1, rng.normal(9, 3))
        wind = max(0.0, rng.normal(9, 4))
        ceil = max(0.0, rng.normal(5000, 2000))
        hour = int(dep_hour[i])
        p = 0.12 + 0.06 * (hour >= 16) + 0.03 * (vis < 3) + 0.04 * (wind > 15)
        p = min(max(p, 0.01), 0.8)
        y = rng.binomial(1, p)
        cancel = rng.binomial(1, 0.02 + 0.03 * (vis < 1))
        rows.append({
            "FL_DATE": fl_date[i].date(),
            "OP_CARRIER": c,
            "ORIGIN": o,
            "DEST": d,
            "CRS_DEP_TIME": hour * 100,
            "DISTANCE": dist,
            "ARR_DELAY": rng.normal(10, 25) if y == 0 else rng.normal(40, 30),
            "CANCELLED": cancel,
            "visibility": vis,
            "wind_speed": wind,
            "ceiling": ceil
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote synthetic data: {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/flights_joined.csv")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    main(args.out, n=args.n)
