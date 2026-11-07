"""Generate SYNTHETIC demo data so the dashboard runs before the real data is processed.

The synthetic households are written in the Kaggle hhblock layout and pushed through
exactly the same feature, clustering and output code as the real pipeline, so the
dashboard behaves identically once you run prepare_data.py on the real dataset.
Every page shows a "demo data" banner while these outputs are in use.

    python pipeline/make_demo_data.py --households 1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build import build_outputs  # noqa: E402
from features import HH_COLS, block_to_records  # noqa: E402

t = np.arange(48) / 2 + 0.25  # hour of day at the middle of each half hour


def bump(center, width, height):
    return height * np.exp(-0.5 * ((t - center) / width) ** 2)


# name: (share of homes, base, shape fn, seasonal amplitude, weekend factor)
ARCHETYPES = {
    "low": (0.27, 0.07, lambda: bump(7.5, 1.0, 0.10) + bump(19, 2.0, 0.16), 0.15, 1.05),
    "evening": (0.24, 0.10, lambda: bump(7.5, 0.8, 0.12) + bump(19, 1.3, 0.55), 0.20, 1.08),
    "baseload": (0.14, 0.32, lambda: bump(8, 1.0, 0.10) + bump(19.5, 2.0, 0.22), 0.12, 1.03),
    "high": (0.12, 0.22, lambda: bump(8, 1.2, 0.35) + bump(13, 2.5, 0.25) + bump(19, 2.0, 0.65), 0.22, 1.12),
    "winter": (0.12, 0.12, lambda: bump(7, 1.2, 0.25) + bump(20, 2.5, 0.35), 0.75, 1.10),
    "daytime": (0.11, 0.10, lambda: bump(8.5, 1.5, 0.15) + bump(13, 3.0, 0.32) + bump(18.5, 1.5, 0.22), 0.20, 1.00),
}
ACORN_TILT = {  # Affluent, Comfortable, Adversity
    "low": [0.25, 0.30, 0.45], "evening": [0.30, 0.30, 0.40], "baseload": [0.40, 0.30, 0.30],
    "high": [0.60, 0.25, 0.15], "winter": [0.25, 0.25, 0.50], "daytime": [0.35, 0.35, 0.30],
}


def make_household(rng, arch, days):
    share, base, shape_fn, s_amp, wk = ARCHETYPES[arch]
    scale = rng.lognormal(0, 0.25)
    shift = rng.normal(0, 0.6)
    shape = np.interp(t - shift, t, shape_fn(), period=24)
    # blend a little of another archetype so segments are not trivially separable
    other = rng.choice(list(ARCHETYPES))
    shape = 0.8 * shape + 0.2 * ARCHETYPES[other][2]()
    base_h = base * rng.lognormal(0, 0.2)
    doy = days.dayofyear.to_numpy()
    seasonal = 1 + s_amp * rng.uniform(0.7, 1.3) * np.cos(2 * np.pi * (doy - 15) / 365)
    weekend = np.where(days.dayofweek >= 5, wk * rng.uniform(0.95, 1.05), 1.0)
    daily = rng.lognormal(0, 0.15, len(days)) * seasonal * weekend
    profile = scale * (base_h + shape[None, :] * daily[:, None])
    noise = rng.gamma(6, 1 / 6, profile.shape)
    x = (profile * noise).astype(np.float32)
    # missing data: occasional short gaps and longer outages
    for _ in range(rng.poisson(6)):
        s = rng.integers(0, x.size - 4)
        x.ravel()[s:s + rng.integers(1, 5)] = np.nan
    for _ in range(rng.poisson(0.6)):
        s = rng.integers(0, x.size - 150)
        x.ravel()[s:s + rng.integers(48, 150)] = np.nan
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=1500)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    year = 2013
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    names = list(ARCHETYPES)
    shares = np.array([ARCHETYPES[n][0] for n in names])
    archs = rng.choice(names, size=a.households, p=shares / shares.sum())

    info_rows, records, n_readings = [], [], 0
    block_rows = []
    for i, arch in enumerate(archs):
        lclid = f"MAC{i + 1:06d}"
        x = make_household(rng, arch, days)
        if rng.random() < 0.07:  # joined late or left early -> fails the 90% coverage rule
            cut = rng.integers(60, 200)
            if rng.random() < 0.5:
                x[:cut] = np.nan
            else:
                x[-cut:] = np.nan
        if i < 4:   # near-vacant properties
            x = x * 0.03
        elif i < 7:  # extreme users
            x = x * 6
        tou = rng.random() < (0.18 if arch != "evening" else 0.12)
        if tou:  # ToU homes shave a little of the evening peak
            x[:, 34:42] *= 0.9
        acorn = rng.choice(["Affluent", "Comfortable", "Adversity"], p=ACORN_TILT[arch])
        if rng.random() < 0.01:
            acorn = "ACORN-U"
        info_rows.append({"LCLid": lclid, "stdorToU": "ToU" if tou else "Std", "Acorn_grouped": acorn})
        df = pd.DataFrame(x, columns=HH_COLS)
        df.insert(0, "day", days)
        df.insert(0, "LCLid", lclid)
        df = df[df[HH_COLS].notna().any(axis=1)]
        if i % 50 == 0 and len(df):  # a few duplicate rows to exercise de-duplication
            df = pd.concat([df, df.iloc[:2]])
        n_readings += int(df[HH_COLS].notna().sum().sum())
        block_rows.append(df)
        if len(block_rows) == 100 or i == len(archs) - 1:
            records += block_to_records(pd.concat(block_rows), year)
            block_rows = []
            print(f"features: {len(records)}/{a.households}")
    build_outputs(records, pd.DataFrame(info_rows), a.out, source="demo", year=year,
                  n_readings=n_readings, reps=a.reps)


if __name__ == "__main__":
    main()
