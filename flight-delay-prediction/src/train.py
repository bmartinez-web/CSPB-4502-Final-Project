"""
Training pipeline that builds features, trains models, and evaluates on val/test.
It expects a cleaned and joined dataset with targets 'delay15' and/or 'cancel'.
"""
from __future__ import annotations
import os
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from .utils import get_logger, save_json, set_seed
from .etl import temporal_split
from .features import add_basic_features, build_matrix
from .models import get_models
from .evaluate import compute_metrics, reliability_plot, pr_curve, roc_curve_plot

LOG = get_logger("train")

def run_pipeline(
    bts_csv: str = "data/processed/flights_joined.csv",
    target: str = "delay15",
    out_dir: str = "models",
    calibrate: bool = True
):
    set_seed(42)
    if not os.path.exists(bts_csv):
        LOG.error("Processed dataset %s not found. Run ETL first.", bts_csv)
        return
    df = pd.read_csv(bts_csv, parse_dates=["FL_DATE","date_hour"], infer_datetime_format=True)
    train, val, test = temporal_split(df)

    Xtr, ytr = build_matrix(add_basic_features(train), target)
    Xva, yva = build_matrix(add_basic_features(val), target)
    Xte, yte = build_matrix(add_basic_features(test), target)

    models = get_models()
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for name, model in models.items():
        LOG.info("Training %s", name)
        clf = model
        if calibrate:
            clf = CalibratedClassifierCV(model, method="isotonic", cv=3)
        clf.fit(Xtr, ytr)

        p_va = clf.predict_proba(Xva)[:,1]
        yhat_va = (p_va >= 0.5).astype(int)
        m_va = compute_metrics(yva, p_va, yhat_va)

        p_te = clf.predict_proba(Xte)[:,1]
        yhat_te = (p_te >= 0.5).astype(int)
        m_te = compute_metrics(yte, p_te, yhat_te)

        model_dir = os.path.join(out_dir, f"{name}_{target}")
        os.makedirs(model_dir, exist_ok=True)
        save_json(m_va, os.path.join(model_dir, "metrics_val.json"))
        save_json(m_te, os.path.join(model_dir, "metrics_test.json"))

        reliability_plot(yva, p_va, os.path.join(model_dir, "reliability_val.png"))
        pr_curve(yva, p_va, os.path.join(model_dir, "pr_val.png"))
        roc_curve_plot(yva, p_va, os.path.join(model_dir, "roc_val.png"))

        all_results[name] = {"val": m_va, "test": m_te}

    save_json(all_results, os.path.join(out_dir, f"summary_{target}.json"))
    LOG.info("Finished. Results saved under %s", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/flights_joined.csv")
    ap.add_argument("--target", default="delay15", choices=["delay15","cancel"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()
    run_pipeline(args.data, args.target, args.out, calibrate=not args.no_calibrate)
