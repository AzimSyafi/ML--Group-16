"""Build the dashboard data from the real London smart meter dataset.

Two input layouts are supported:

  Kaggle "Smart meters in London" (recommended, already split into blocks)
    python pipeline/prepare_data.py kaggle --root path/to/smart-meters-in-london

    Expects  <root>/hhblock_dataset/**/block_*.csv  and  <root>/informations_households.csv

  London Datastore raw long-format CSV(s) (the ~10 GB file or its partitions)
    python pipeline/prepare_data.py datastore --csv "path/to/LCL-*.csv"

Common options:  --year 2013  --coverage 0.9  --reps 10  --model "K-Means" --k 5  --out data
"""
from __future__ import annotations

import argparse
import glob
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build import build_outputs  # noqa: E402
from features import HH_COLS, block_to_records  # noqa: E402


def from_kaggle(root: Path, year: int):
    files = sorted(glob.glob(str(root / "hhblock_dataset" / "**" / "block_*.csv"), recursive=True))
    if not files:
        sys.exit(f"No block_*.csv files found under {root / 'hhblock_dataset'}")
    info = pd.read_csv(root / "informations_households.csv")[["LCLid", "stdorToU", "Acorn_grouped"]]
    records, n_readings = [], 0
    for i, f in enumerate(files, 1):
        block = pd.read_csv(f)
        n_readings += int(block[HH_COLS].notna().sum().sum())
        records += block_to_records(block, year)
        print(f"[{i}/{len(files)}] {Path(f).name}: {len(records)} households so far")
    return records, info, n_readings


def from_datastore(pattern: str, year: int, chunksize: int = 2_000_000):
    """Read the long format (LCLid, stdorToU, DateTime, KWH/hh (per half hour), Acorn, Acorn_grouped)
    in chunks, keep one year, pivot to hhblock layout and stage as Parquet."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files match {pattern}")
    stage = Path(tempfile.mkdtemp(prefix="lcl_stage_"))
    info_parts, part = [], 0
    for f in files:
        for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            kwh_col = [c for c in chunk.columns if c.upper().startswith("KWH")][0]
            chunk["DateTime"] = pd.to_datetime(chunk["DateTime"], errors="coerce")
            chunk = chunk[chunk["DateTime"].dt.year == year]
            if chunk.empty:
                continue
            chunk["kwh"] = pd.to_numeric(chunk[kwh_col], errors="coerce")  # "Null" -> NaN
            chunk["day"] = chunk["DateTime"].dt.normalize()
            chunk["hh"] = chunk["DateTime"].dt.hour * 2 + (chunk["DateTime"].dt.minute >= 30)
            info_parts.append(chunk[["LCLid", "stdorToU", "Acorn_grouped"]].drop_duplicates("LCLid"))
            wide = chunk.pivot_table(index=["LCLid", "day"], columns="hh", values="kwh", aggfunc="first")
            wide.columns = [f"hh_{int(c)}" for c in wide.columns]
            wide.reset_index().to_parquet(stage / f"part_{part:05d}.parquet", index=False)
            part += 1
            print(f"staged chunk {part} from {Path(f).name}")
    wide = pd.concat([pd.read_parquet(p) for p in sorted(stage.glob("*.parquet"))], ignore_index=True)
    for c in HH_COLS:
        if c not in wide:
            wide[c] = np.nan
    # Households split across chunk boundaries: merge partial days
    wide = wide.groupby(["LCLid", "day"], as_index=False)[HH_COLS].first()
    n_readings = int(wide[HH_COLS].notna().sum().sum())
    info = pd.concat(info_parts).drop_duplicates("LCLid")
    records = []
    ids = wide.LCLid.unique()
    for i in range(0, len(ids), 200):
        records += block_to_records(wide[wide.LCLid.isin(ids[i:i + 200])], year)
        print(f"features: {min(i + 200, len(ids))}/{len(ids)} households")
    return records, info, n_readings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    k = sub.add_parser("kaggle")
    k.add_argument("--root", required=True, type=Path)
    d = sub.add_parser("datastore")
    d.add_argument("--csv", required=True, help='glob pattern, e.g. "data/raw/LCL-June2015v2_*.csv"')
    for p in (k, d):
        p.add_argument("--year", type=int, default=2013)
        p.add_argument("--coverage", type=float, default=0.9)
        p.add_argument("--reps", type=int, default=10, help="80%% subsample repeats for ARI stability")
        p.add_argument("--model", choices=["K-Means", "Hierarchical (Ward)", "GMM"], default=None)
        p.add_argument("--k", type=int, default=None)
        p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    a = ap.parse_args()

    if a.source == "kaggle":
        records, info, n = from_kaggle(a.root, a.year)
    else:
        records, info, n = from_datastore(a.csv, a.year)
    build_outputs(records, info, a.out, source=a.source, year=a.year, n_readings=n,
                  coverage_min=a.coverage, model=a.model, k=a.k, reps=a.reps)


if __name__ == "__main__":
    main()
