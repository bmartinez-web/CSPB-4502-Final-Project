"""
Create a processed dataset by loading BTS and weather CSVs, cleaning, and joining.

Usage:
  python scripts/make_dataset.py --bts data/raw/flights_2019_2023.csv \
                                 --wx data/raw/metar_hourly.csv \
                                 --out data/processed/flights_joined.csv
"""
from __future__ import annotations
import argparse, os
from src.utils import get_logger
from src.etl import load_bts, load_weather, clean_bts, join_weather

LOG = get_logger("make_dataset")

def main(bts_csv: str, wx_csv: str, out_csv: str):
    df = load_bts(bts_csv)
    df = clean_bts(df)
    if wx_csv:
        wx = load_weather(wx_csv)
        df = join_weather(df, wx)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOG.info("Wrote %s rows to %s", len(df), out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bts", required=True, help="Path to BTS flights CSV (2019–2023)")
    ap.add_argument("--wx", default="", help="Optional weather CSV")
    ap.add_argument("--out", default="data/processed/flights_joined.csv")
    args = ap.parse_args()
    main(args.bts, args.wx, args.out)
