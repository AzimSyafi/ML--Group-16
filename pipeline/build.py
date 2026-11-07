"""Clustering, evaluation and dashboard output writer.

Shared by prepare_data.py (real data) and make_demo_data.py (synthetic demo).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from features import CLUSTER_FEATURES, SEASON_ORDER

MODELS = ["K-Means", "Hierarchical (Ward)", "GMM"]

# Segment naming: name -> (feature driving the score, sign)
NAME_RULES = {
    "High baseload": ("baseload_kwh", +1),
    "Evening peak": ("share_evening_peak", +1),
    "High consumers": ("mean_daily_kwh", +1),
    "Low and steady": ("mean_daily_kwh", -1),
    "Winter heavy": ("seasonal_ratio", +1),
    "Daytime users": ("share_daytime", +1),
    "Morning users": ("share_morning", +1),
    "Weekend heavy": ("weekend_ratio", +1),
}
# name -> (recommended action, target load, default reduction %)
ACTIONS = {
    "High baseload": ("Standby-power audit and smart plugs to cut always-on load", "baseload", 25),
    "Evening peak": ("Time-of-use tariff and load-shifting advice for 17:00-21:00", "evening_peak", 15),
    "High consumers": ("Appliance upgrade incentives and a home energy audit", "total", 10),
    "Low and steady": ("Low-touch information only; already efficient", "total", 2),
    "Winter heavy": ("Heating controls, insulation and heat pump advice", "winter_excess", 15),
    "Daytime users": ("Efficient appliances and daytime solar matching", "total", 5),
    "Morning users": ("Behavioural nudges on morning routines", "morning", 10),
    "Weekend heavy": ("General efficiency messaging targeted at weekends", "total", 5),
    "Typical": ("General efficiency messaging", "total", 5),
}


def _fit(model: str, k: int, Z: np.ndarray, seed: int = 42, cov: str | None = None):
    if model == "K-Means":
        m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(Z)
        return m.labels_, {"inertia": float(m.inertia_)}
    if model == "Hierarchical (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z), {}
    # GMM: choose covariance type by BIC unless fixed
    best = None
    for ct in ([cov] if cov else ["full", "diag"]):
        g = GaussianMixture(n_components=k, covariance_type=ct, n_init=3, random_state=seed).fit(Z)
        bic = g.bic(Z)
        if best is None or bic < best[1]:
            best = (g, bic, ct)
    g, bic, ct = best
    return g.predict(Z), {"bic": float(bic), "covariance": ct, "maxprob": g.predict_proba(Z).max(axis=1)}


def run_models(Z: np.ndarray, k_range=range(2, 11), reps: int = 10, frac: float = 0.8, seed: int = 42, log=print):
    rng = np.random.default_rng(seed)
    rows, labels_store = [], {}
    n = len(Z)
    for model in MODELS:
        for k in k_range:
            labels, extra = _fit(model, k, Z, seed)
            labels_store[(model, k)] = (labels, extra)
            aris = []
            for r in range(reps):
                idx = rng.choice(n, size=int(frac * n), replace=False)
                sub, _ = _fit(model, k, Z[idx], seed + r + 1, extra.get("covariance"))
                aris.append(adjusted_rand_score(labels[idx], sub))
            rows.append({
                "model": model, "k": k,
                "silhouette": silhouette_score(Z, labels),
                "davies_bouldin": davies_bouldin_score(Z, labels),
                "calinski_harabasz": calinski_harabasz_score(Z, labels),
                "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris)),
                "inertia": extra.get("inertia", np.nan), "bic": extra.get("bic", np.nan),
                "gmm_covariance": extra.get("covariance", ""),
                "min_cluster_share": float(np.bincount(labels).min() / n),
            })
            log(f"  {model:<20} k={k:<2} silhouette={rows[-1]['silhouette']:.3f} ARI={rows[-1]['ari_mean']:.3f}")
    return pd.DataFrame(rows), labels_store


def select_model(metrics: pd.DataFrame, k_min=3, k_max=8, min_share=0.03):
    """Composite rank across Silhouette, DBI, CH and stability, restricted to interpretable k."""
    m = metrics[(metrics.k >= k_min) & (metrics.k <= k_max) & (metrics.min_cluster_share >= min_share)].copy()
    if m.empty:
        m = metrics.copy()
    m["rank"] = (m.silhouette.rank(ascending=False) + m.davies_bouldin.rank(ascending=True)
                 + m.calinski_harabasz.rank(ascending=False) + m.ari_mean.rank(ascending=False))
    best = m.sort_values(["rank", "silhouette"], ascending=[True, False]).iloc[0]
    return best.model, int(best.k)


def name_segments(hh: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    feats = list({f for f, _ in NAME_RULES.values()})
    z = (hh[feats] - hh[feats].mean()) / hh[feats].std()
    cm = z.groupby(labels).mean()
    names = list(NAME_RULES)
    score = np.array([[sign * cm.loc[c, f] for (f, sign) in NAME_RULES.values()] for c in cm.index])
    r, c = linear_sum_assignment(-score)
    out = {}
    for ci, ni in zip(r, c):
        out[int(cm.index[ci])] = names[ni] if score[ci, ni] >= 0.25 else "Typical"
    # de-duplicate any repeated "Typical"
    seen = {}
    for k_, v in sorted(out.items()):
        seen[v] = seen.get(v, 0) + 1
        if v == "Typical" and seen[v] > 1:
            out[k_] = f"Typical {seen[v]}"
    return out


def target_kwh(row: pd.Series, target: str) -> float:
    if target == "baseload":
        return row.baseload_kwh * 48 * 365
    if target == "evening_peak":
        return row.share_evening_peak * row.annual_kwh
    if target == "morning":
        return row.share_morning * row.annual_kwh
    if target == "winter_excess":
        return max(row.winter_daily_kwh - row.summer_daily_kwh, 0) * 90
    return row.annual_kwh


def build_outputs(records: list[dict], info: pd.DataFrame, out_dir: Path, *, source: str, year: int,
                  n_readings: int, coverage_min=0.9, model=None, k=None, reps=10, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k_: v for k_, v in r.items() if not k_.startswith("_")} for r in records])
    arrays = {r["LCLid"]: r for r in records}
    df = df.merge(info, on="LCLid", how="left")
    df["stdorToU"] = df["stdorToU"].fillna("Unknown")
    df["Acorn_grouped"] = df["Acorn_grouped"].fillna("Unknown")
    n_raw = len(df)
    df[["LCLid", "coverage", "stdorToU", "Acorn_grouped"]].to_csv(out_dir / "coverage.csv", index=False)

    # 1. Coverage filter
    kept = df[(df.coverage >= coverage_min)].dropna(subset=CLUSTER_FEATURES).copy()
    n_cov = len(kept)

       # 2. Outlier screen on magnitude only (near-zero homes and 3xIQR extremes) -> analysed separately.
    #    Shape features (shares, ratios) are not screened: unusual shapes are what clustering should find.
    flag = kept.mean_daily_kwh < 0.5
    # Homes that were near-empty for a whole season (e.g. vacant all summer) give absurd seasonal ratios
    flag |= (kept.winter_daily_kwh < 0.5) | (kept.summer_daily_kwh < 0.5)
    for f in ["mean_daily_kwh", "baseload_kwh"]:
        q1, q3 = kept[f].quantile([0.25, 0.75])
        iqr = q3 - q1
        flag |= (kept[f] < q1 - 3 * iqr) | (kept[f] > q3 + 3 * iqr)
    kept["outlier"] = flag
    kept[kept.outlier].drop(columns=["outlier"]).to_csv(out_dir / "outliers.csv", index=False)
    hh = kept[~kept.outlier].drop(columns=["outlier"]).reset_index(drop=True)
    log(f"Households: raw={n_raw}, coverage>={coverage_min:.0%}: {n_cov}, outliers={int(flag.sum())}, clustered={len(hh)}")

    # 3. Scale (log for skewed kWh features) and PCA
    X = hh[CLUSTER_FEATURES].copy()
    for f in ["mean_daily_kwh", "baseload_kwh"]:
        X[f] = np.log1p(X[f])
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    n_comp = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    Zall = pca.transform(Xs)
    Z = Zall[:, :n_comp]
    pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                  "explained": pca.explained_variance_ratio_}).to_csv(out_dir / "pca_variance.csv", index=False)
    hh[CLUSTER_FEATURES].corr().to_csv(out_dir / "feature_corr.csv")
    log(f"PCA: {n_comp} components keep 95% of variance")

    # 4. Models
    metrics, store = run_models(Z, reps=reps, log=log)
    metrics.to_csv(out_dir / "model_metrics.csv", index=False)
    auto_model, auto_k = select_model(metrics)
    model = model or auto_model
    k = k or auto_k
    labels, extra = store[(model, k)]
    log(f"Selected: {model}, k={k} (auto choice was {auto_model}, k={auto_k})")

    # Relabel clusters by size (0 = largest) so colours stay stable
    order = pd.Series(labels).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    hh["cluster"] = labels
    hh["pc1"], hh["pc2"] = Zall[:, 0], Zall[:, 1]
    if "maxprob" in extra:
        hh["gmm_maxprob"] = extra["maxprob"]
    # Per-household average daily profile (48 half hours) and monthly mean daily kWh,
    # so the dashboard can recompute curves when filters are applied
    prof_cols = {f"p_{h}": [float(arrays[i]["_profile"][h]) for i in hh.LCLid] for h in range(48)}
    mon_cols = {f"m_{m + 1}": [float(arrays[i]["_monthly"][m]) for i in hh.LCLid] for m in range(12)}
    hh = pd.concat([hh, pd.DataFrame(prof_cols), pd.DataFrame(mon_cols)], axis=1)
    hh.to_parquet(out_dir / "households.parquet", index=False)

    # 5. Segment names, actions and savings targets
    names = name_segments(hh, labels)
    seg_rows = []
    for c in sorted(hh.cluster.unique()):
        g = hh[hh.cluster == c]
        nm = names[c]
        action, target, red = ACTIONS.get(nm.split(" ")[0] if nm.startswith("Typical") else nm, ACTIONS["Typical"])
        seg_rows.append({
            "cluster": c, "name": nm, "households": len(g), "action": action, "target": target,
            "reduction_pct": red,
            "target_kwh_per_household": float(g.apply(lambda r: target_kwh(r, target), axis=1).mean()),
        })
    pd.DataFrame(seg_rows).to_csv(out_dir / "segment_labels.csv", index=False)

    # 6. Curves for the dashboard
    prof = np.stack([arrays[i]["_profile"] for i in hh.LCLid])
    rows = []
    for c in sorted(hh.cluster.unique()):
        p = prof[labels == c]
        for h in range(48):
            rows.append({"cluster": c, "hh": h, "mean": p[:, h].mean(),
                         "p25": np.percentile(p[:, h], 25), "p75": np.percentile(p[:, h], 75)})
    pd.DataFrame(rows).to_csv(out_dir / "centroid_curves.csv", index=False)

    mon = np.stack([arrays[i]["_monthly"] for i in hh.LCLid])
    mrows = []
    for c in ["All"] + sorted(hh.cluster.unique().tolist()):
        sel = np.ones(len(hh), bool) if c == "All" else labels == c
        for m in range(12):
            mrows.append({"cluster": c, "month": m + 1, "mean_daily_kwh": float(np.nanmean(mon[sel, m]))})
    pd.DataFrame(mrows).to_csv(out_dir / "monthly.csv", index=False)

    srows = []
    for s in SEASON_ORDER:
        for dt in ["Weekday", "Weekend"]:
            ps = [arrays[i]["_seasonal_profiles"][(s, dt)] for i in hh.LCLid
                  if (s, dt) in arrays[i]["_seasonal_profiles"]]
            if ps:
                m = np.mean(ps, axis=0)
                srows += [{"season": s, "daytype": dt, "hh": h, "kwh": float(m[h])} for h in range(48)]
    pd.DataFrame(srows).to_csv(out_dir / "seasonal_curves.csv", index=False)

    meta = {
        "source": source, "year": year, "n_readings": int(n_readings),
        "n_households_raw": int(n_raw), "n_after_coverage": int(n_cov),
        "n_outliers": int(flag.sum()), "n_clustered": int(len(hh)),
        "coverage_min": coverage_min, "pca_components": n_comp,
        "selected_model": model, "selected_k": int(k),
        "auto_model": auto_model, "auto_k": int(auto_k),
        "stability_reps": reps, "features": CLUSTER_FEATURES,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Wrote dashboard data to {out_dir}")
    return meta
