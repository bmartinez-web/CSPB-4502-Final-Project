"""
ETL for BTS On-Time data and airport-hour weather.
This module is *leakage-safe*: only pre-departure features should be used downstream.
"""
from __future__ import annotations
import os
import pandas as pd
from typing import Optional
from .utils import get_logger

LOG = get_logger("etl")

def load_bts(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        LOG.warning("BTS CSV not found at %s. Returning empty DataFrame.", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

def load_weather(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        LOG.warning("Weather CSV not found at %s. Returning empty DataFrame.", csv_path)
        return pd.DataFrame()
    wx = pd.read_csv(csv_path)
    return wx

def clean_bts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "ARR_DELAY" in df.columns and "CANCELLED" in df.columns:
        df["delay15"] = (df["ARR_DELAY"] >= 15).astype(int)
        df["cancel"] = (df["CANCELLED"] == 1).astype(int)
    return df

def join_weather(flights: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    if flights.empty:
        return flights
    if wx.empty:
        LOG.warning("Weather empty; skipping join.")
        return flights
    df = flights.copy()
    if "CRS_DEP_TIME" in df.columns:
        dep_hour = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce") // 100
        df["DEP_HOUR"] = dep_hour
    else:
        df["DEP_HOUR"] = 12
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df["date_hour"] = df["FL_DATE"] + pd.to_timedelta(df["DEP_HOUR"], unit="h")

    w = wx.copy()
    if "date_hour" in w.columns:
        w["date_hour"] = pd.to_datetime(w["date_hour"], errors="coerce")
    for c in ["airport","origin"]:
        if c in w.columns:
            w[c] = w[c].astype(str).str.strip().str.upper()
    if "origin" in w.columns and "airport" not in w.columns:
        w["airport"] = w["origin"]

    if "ORIGIN" in df.columns and "airport" in w.columns:
        merged = df.merge(
            w.drop_duplicates(subset=["airport","date_hour"]),
            left_on=["ORIGIN","date_hour"],
            right_on=["airport","date_hour"],
            how="left",
            suffixes=("","_wx")
        )
    else:
        LOG.warning("Weather join keys missing; returning original flights.")
        merged = df
    return merged

def temporal_split(df: pd.DataFrame):
    if df.empty or "FL_DATE" not in df.columns:
        return df, df.copy(), df.copy()
    train = df[df["FL_DATE"].dt.year.isin([2019, 2020, 2021])].copy()
    val   = df[df["FL_DATE"].dt.year.eq(2022)].copy()
    test  = df[df["FL_DATE"].dt.year.eq(2023)].copy()
    return train, val, test
