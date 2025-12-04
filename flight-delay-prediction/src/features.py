"""
Feature engineering for flights + weather.
"""
from __future__ import annotations
import numpy as np, pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # Temporal encodings
    if "DEP_HOUR" in df.columns:
        h = df["DEP_HOUR"].fillna(12).astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    if "FL_DATE" in df.columns:
        dow = df["FL_DATE"].dt.weekday
        df["DOW"] = dow
        mon = df["FL_DATE"].dt.month
        df["MONTH"] = mon
        df["IS_SUMMER"] = df["MONTH"].isin([6,7,8]).astype(int)
        df["IS_WINTER"] = df["MONTH"].isin([12,1,2]).astype(int)
    if "DISTANCE" in df.columns:
        df["DISTANCE_LOG"] = np.log1p(df["DISTANCE"])
    for col in ["visibility","wind_speed","ceiling"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df

def build_matrix(df: pd.DataFrame, target: str):
    num_cols = [c for c in ["hour_sin","hour_cos","DOW","MONTH","DISTANCE_LOG",
                            "visibility","wind_speed","ceiling"] if c in df.columns]
    cat_cols = [c for c in ["OP_CARRIER","ORIGIN","DEST"] if c in df.columns]
    X_num = df[num_cols] if num_cols else pd.DataFrame(index=df.index)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[target] if target in df.columns else None
    return X, y
