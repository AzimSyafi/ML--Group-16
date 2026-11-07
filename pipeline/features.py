"""Household-level feature engineering for the London smart meter data.

Input format is the Kaggle "hhblock" layout: one row per household per day with
columns LCLid, day, hh_0 ... hh_47 (hh_0 = 00:00-00:30, hh_47 = 23:30-24:00).
The London Datastore long format is converted to this layout in prepare_data.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HH_COLS = [f"hh_{i}" for i in range(48)]

# Half-hour index ranges [start, end) for each time-of-day band
BANDS = {
    "night": (0, 12),          # 00:00-06:00
    "morning": (12, 20),       # 06:00-10:00
    "daytime": (20, 34),       # 10:00-17:00
    "evening_peak": (34, 42),  # 17:00-21:00
    "late_evening": (42, 48),  # 21:00-24:00
}
BASELOAD_SLICE = slice(2, 10)  # 01:00-05:00
PEAK_SLICE = slice(34, 42)     # 17:00-21:00
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

CLUSTER_FEATURES = [
    "mean_daily_kwh", "share_night", "share_morning", "share_daytime",
    "share_evening_peak", "share_late_evening", "baseload_kwh",
    "peak_to_avg", "weekend_ratio", "seasonal_ratio",
]
FEATURE_LABELS = {
    "mean_daily_kwh": "Mean daily use (kWh/day)",
    "share_night": "Night share (00-06)",
    "share_morning": "Morning share (06-10)",
    "share_daytime": "Daytime share (10-17)",
    "share_evening_peak": "Evening peak share (17-21)",
    "share_late_evening": "Late evening share (21-24)",
    "baseload_kwh": "Baseload (kWh per half hour)",
    "peak_to_avg": "Peak-to-average ratio",
    "weekend_ratio": "Weekend-to-weekday ratio",
    "seasonal_ratio": "Winter-to-summer ratio",
}


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= max_gap that have valid values on both sides."""
    x = x.copy()
    isnan = np.isnan(x)
    if not isnan.any():
        return x
    idx = np.arange(len(x))
    valid = ~isnan
    if valid.sum() < 2:
        return x
    # Identify NaN runs
    d = np.diff(np.concatenate(([0], isnan.astype(np.int8), [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    interp = np.interp(idx, idx[valid], x[valid])
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap and s > 0 and e < len(x):
            x[s:e] = interp[s:e]
    return x


def household_features(lclid: str, days: pd.DataFrame, year: int, max_gap: int = 4) -> dict | None:
    """Compute features for one household. `days` has a DatetimeIndex (one row per day) and HH_COLS."""
    calendar = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    days = days.reindex(calendar)
    raw = days[HH_COLS].to_numpy(dtype=float)
    raw[raw < 0] = np.nan
    n_slots = raw.size
    coverage = float(np.isfinite(raw).sum() / n_slots)
    if coverage == 0:
        return {"LCLid": lclid, "coverage": 0.0}

    filled = _fill_short_gaps(raw.ravel(), max_gap).reshape(raw.shape)
    complete = np.isfinite(filled).all(axis=1)
    if complete.sum() < 30:
        return {"LCLid": lclid, "coverage": coverage}

    cd = filled[complete]
    cdates = calendar[complete]
    daily = cd.sum(axis=1)
    profile = cd.mean(axis=0)
    total = profile.sum()

    is_weekend = cdates.dayofweek >= 5
    months = cdates.month
    winter = np.isin(months, [12, 1, 2])
    summer = np.isin(months, [6, 7, 8])

    rec = {
        "LCLid": lclid,
        "coverage": coverage,
        "complete_days": int(complete.sum()),
        "mean_daily_kwh": float(daily.mean()),
        "annual_kwh": float(daily.mean() * 365),
        "baseload_kwh": float(profile[BASELOAD_SLICE].mean()),
        "peak_to_avg": float(profile[PEAK_SLICE].mean() / profile.mean()) if total > 0 else np.nan,
        "weekend_ratio": float(daily[is_weekend].mean() / daily[~is_weekend].mean())
        if is_weekend.any() and (~is_weekend).any() else np.nan,
        "seasonal_ratio": float(daily[winter].mean() / daily[summer].mean())
        if winter.any() and summer.any() else np.nan,
        "winter_daily_kwh": float(daily[winter].mean()) if winter.any() else np.nan,
        "summer_daily_kwh": float(daily[summer].mean()) if summer.any() else np.nan,
    }
    for band, (a, b) in BANDS.items():
        rec[f"share_{band}"] = float(profile[a:b].sum() / total) if total > 0 else np.nan

    # Supporting arrays for the dashboard (not used for clustering)
    rec["_profile"] = profile.astype(np.float32)
    monthly = np.full(12, np.nan, dtype=np.float32)
    for m in range(1, 13):
        sel = months == m
        if sel.any():
            monthly[m - 1] = daily[sel].mean()
    rec["_monthly"] = monthly
    seasonal_profiles = {}
    season_of_day = np.array([SEASONS[m] for m in months])
    for s in SEASON_ORDER:
        for dt, mask in (("Weekday", ~is_weekend), ("Weekend", is_weekend)):
            sel = (season_of_day == s) & mask
            if sel.any():
                seasonal_profiles[(s, dt)] = cd[sel].mean(axis=0).astype(np.float32)
    rec["_seasonal_profiles"] = seasonal_profiles
    return rec


def block_to_records(block: pd.DataFrame, year: int, max_gap: int = 4) -> list[dict]:
    """Turn one hhblock-format DataFrame into a list of household feature records."""
    block = block.copy()
    block["day"] = pd.to_datetime(block["day"])
    block = block[block["day"].dt.year == year]
    for c in HH_COLS:
        block[c] = pd.to_numeric(block[c], errors="coerce")
    block = block.drop_duplicates(subset=["LCLid", "day"], keep="first")
    out = []
    for lclid, g in block.groupby("LCLid", sort=False):
        rec = household_features(lclid, g.set_index("day")[HH_COLS], year, max_gap)
        if rec is not None:
            out.append(rec)
    return out
