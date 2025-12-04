"""
Model definitions and calibration utilities.
"""
from __future__ import annotations
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def get_models() -> Dict[str, object]:
    models = {
        "logreg": LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            eval_metric="logloss"
        )
    return models
