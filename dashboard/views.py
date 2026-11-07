"""The six dashboard pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from style import (AXIS, DIVERGING, FADED, GRID, HH_TICKS, INK, INK2, MUTED, SERIES, add_bands,
                   chip, color, header, kpi)

P_COLS = [f"p_{h}" for h in range(48)]
M_COLS = [f"m_{m}" for m in range(1, 13)]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HH_LABELS = [f"{h // 2:02d}:{(h % 2) * 30:02d}" for h in range(48)]
FEATURES = {
    "mean_daily_kwh": ("Mean daily use", "kWh/day", "{:.1f}"),
    "baseload_kwh": ("Baseload", "kWh per half hour", "{:.3f}"),
    "share_night": ("Night share", "%", "{:.0%}"),
    "share_morning": ("Morning share", "%", "{:.0%}"),
    "share_daytime": ("Daytime share", "%", "{:.0%}"),
    "share_evening_peak": ("Evening peak share", "%", "{:.0%}"),
    "share_late_evening": ("Late evening share", "%", "{:.0%}"),
    "peak_to_avg": ("Peak-to-average", "ratio", "{:.2f}"),
    "weekend_ratio": ("Weekend-to-weekday", "ratio", "{:.2f}"),
    "seasonal_ratio": ("Winter-to-summer", "ratio", "{:.2f}"),
}
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def show(fig, height=360):
    legend = sum(1 for t in fig.data if t.showlegend is not False and t.name) > 1 and fig.layout.showlegend is not False
    if fig.layout.margin.t is None:
        fig.update_layout(margin=dict(t=92 if legend else 48))
    if legend:
        n = sum(1 for t in fig.data if t.showlegend is not False and t.name)
        if n > 3:
            fig.update_layout(legend=dict(entrywidth=128))
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)


def names(ctx):
    return ctx["seg"]["name"].to_dict()


def fmt_int(n):
    return f"{n:,.0f}"


def target_kwh(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "baseload":
        return df.baseload_kwh * 48 * 365
    if target == "evening_peak":
        return df.share_evening_peak * df.annual_kwh
    if target == "morning":
        return df.share_morning * df.annual_kwh
    if target == "winter_excess":
        return (df.winter_daily_kwh - df.summer_daily_kwh).clip(lower=0) * 90
    return df.annual_kwh


TARGET_LABELS = {"baseload": "Always-on load", "evening_peak": "Evening peak use", "morning": "Morning use",
                 "winter_excess": "Winter excess use", "total": "Total use"}


def scenario(ctx) -> pd.DataFrame:
    hh, seg = ctx["hh"], ctx["seg"]
    adoption = st.session_state.get("adoption", 30)
    red = st.session_state.get("reductions", {})
    rows = []
    for c, s in seg.iterrows():
        g = hh[hh.cluster == c]
        tk = float(target_kwh(g, s.target).mean()) if len(g) else 0.0
        r = red.get(int(c), int(s.reduction_pct))
        kwh = len(g) * adoption / 100 * r / 100 * tk
        rows.append({"cluster": c, "Segment": s["name"], "Action": s.action,
                     "Targeted load": TARGET_LABELS.get(s.target, s.target), "Households": len(g),
                     "Target kWh per home": tk, "Reduction %": r, "kWh saved": kwh,
                     "kg CO2e avoided": kwh * ctx["ef"]})
    return pd.DataFrame(rows)


def curve_fig(hh, nm, selected=None, ribbon=True, title=""):
    fig = go.Figure()
    add_bands(fig)
    means = hh.groupby("cluster")[P_COLS].mean()
    order = [c for c in means.index if c != selected] + ([selected] if selected in means.index else [])
    for c in order:
        is_sel = selected is None or c == selected
        if c == selected and ribbon:
            g = hh[hh.cluster == c][P_COLS]
            fig.add_trace(go.Scatter(x=list(range(48)) + list(range(47, -1, -1)),
                                     y=list(g.quantile(0.75)) + list(g.quantile(0.25))[::-1],
                                     fill="toself", fillcolor=color(c), opacity=0.14, line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=list(range(48)), y=means.loc[c], mode="lines", name=nm[c],
            line=dict(color=color(c) if is_sel else FADED, width=3 if c == selected else 2, shape="spline",
                      smoothing=0.6),
            customdata=HH_LABELS,
            hovertemplate=f"<b>{nm[c]}</b><br>%{{customdata}}<br>%{{y:.3f}} kWh<extra></extra>"))
    fig.update_layout(title=title, hovermode="x unified" if selected is None else "closest",
                      xaxis=dict(**HH_TICKS, range=[0, 47.2]),
                      yaxis_title="Average kWh per half hour")
    return fig


# =====================================================================================
def overview(ctx):
    meta, hh, nm = ctx["meta"], ctx["hh"], names(ctx)
    m = ctx["metrics"]
    sel = m[(m.model == meta["selected_model"]) & (m.k == meta["selected_k"])].iloc[0]
    sc = scenario(ctx)
    header("Household electricity segments",
           f"London smart meter data, {meta['year']}. Clustering {fmt_int(meta['n_clustered'])} households by "
           "how and when they use electricity, to target efficiency programmes and cut carbon (SDG 7.3).")
    c = st.columns(5)
    kpi(c[0], "Households clustered", fmt_int(len(hh)),
        f"of {fmt_int(meta['n_households_raw'])} in the dataset" if len(hh) == meta["n_clustered"]
        else f"filtered from {fmt_int(meta['n_clustered'])}")
    kpi(c[1], "Readings processed", f"{meta['n_readings'] / 1e6:.1f}M", "half-hourly kWh values")
    kpi(c[2], "Selected model", meta["selected_model"].split(" ")[0],
        f"k = {meta['selected_k']} segments, best of 3 algorithms")
    kpi(c[3], "Silhouette score", f"{sel.silhouette:.2f}", f"stability ARI {sel.ari_mean:.2f}")
    kpi(c[4], "Potential saving", f"{sc['kg CO2e avoided'].sum() / 1000:,.0f} t CO2e",
        f"per year at {st.session_state.get('adoption', 30)}% adoption")

    st.write("")
    left, right = st.columns([5, 7])
    with left:
        counts = hh.cluster.value_counts().reindex(ctx["seg"].index, fill_value=0)
        fig = go.Figure(go.Bar(
            y=[nm[i] for i in counts.index], x=counts.values, orientation="h",
            marker=dict(color=[color(i) for i in counts.index], cornerradius=4),
            text=[f"{v:,} ({v / max(counts.sum(), 1):.0%})" for v in counts.values], textposition="outside",
            textfont=dict(color=INK2), cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,} households<extra></extra>"))
        fig.update_layout(title="Segment sizes", yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                          xaxis=dict(showticklabels=False, range=[0, counts.max() * 1.45]), bargap=0.35)
        show(fig, 380)
    with right:
        show(curve_fig(hh, nm, title="Average daily load curve by segment"), 380)

    st.markdown("### What each segment means")
    stats = hh.groupby("cluster").agg(n=("LCLid", "size"), daily=("mean_daily_kwh", "mean"),
                                      base=("baseload_kwh", "mean"), peak=("share_evening_peak", "mean"))
    rows = ""
    for cl, s in ctx["seg"].iterrows():
        if cl not in stats.index:
            continue
        r = stats.loc[cl]
        sv = sc.set_index("cluster").loc[cl, "kg CO2e avoided"] / 1000
        rows += (f"<tr><td>{chip(cl)}<b>{s['name']}</b></td><td>{int(r.n):,}</td><td>{r.daily:.1f}</td>"
                 f"<td>{r.base:.3f}</td><td>{r.peak:.0%}</td><td>{s.action}</td><td>{sv:,.1f}</td></tr>")
    st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:.9rem;background:#fcfcfb;border-radius:12px;overflow:hidden">
<thead><tr style="text-align:left;color:{MUTED};border-bottom:1px solid {AXIS}">
<th style="padding:10px 12px">Segment</th><th>Homes</th><th>kWh/day</th><th>Baseload</th>
<th>Evening share</th><th>Recommended action</th><th>t CO2e/yr</th></tr></thead>
<tbody>{rows}</tbody></table>
<style>tbody tr td {{padding:9px 12px;border-bottom:1px solid {GRID};color:{INK2}}} tbody tr td b {{color:{INK}}}</style>
""", unsafe_allow_html=True)


# =====================================================================================
def eda(ctx):
    meta = ctx["meta"]
    header("Data and exploratory analysis",
           "What the cleaned 2013 data looks like before clustering: daily routines, seasonality, coverage "
           "and the mix of tariffs and Acorn groups.")
    c = st.columns(4)
    kpi(c[0], "Households in raw data", fmt_int(meta["n_households_raw"]))
    kpi(c[1], f"Pass {meta['coverage_min']:.0%} coverage", fmt_int(meta["n_after_coverage"]),
        f"{meta['n_households_raw'] - meta['n_after_coverage']:,} dropped")
    kpi(c[2], "Outliers set aside", fmt_int(meta["n_outliers"]), "near-zero or 3×IQR extremes")
    kpi(c[3], "Households clustered", fmt_int(meta["n_clustered"]), f"{meta['pca_components']} PCA components")
    st.write("")

    dt = st.segmented_control("Day type for load curves", ["Weekday", "Weekend"], default="Weekday", key="eda_dt")
    l, r = st.columns(2)
    with l:
        s = ctx["seasonal"][ctx["seasonal"].daytype == (dt or "Weekday")]
        fig = go.Figure()
        add_bands(fig)
        for i, season in enumerate(["Winter", "Spring", "Summer", "Autumn"]):
            g = s[s.season == season]
            fig.add_trace(go.Scatter(x=g.hh, y=g.kwh, name=season, mode="lines",
                                     line=dict(color=SERIES[i], width=2, shape="spline", smoothing=0.6),
                                     customdata=HH_LABELS,
                                     hovertemplate=f"{season}: %{{y:.3f}} kWh<extra></extra>"))
        fig.update_layout(title="Average household load curve by season", hovermode="x unified",
                          xaxis=dict(**HH_TICKS, range=[0, 47.2]), yaxis_title="kWh per half hour")
        show(fig, 360)
    with r:
        mon = ctx["hh_all"][M_COLS].mean()
        fig = go.Figure(go.Scatter(x=MONTHS, y=mon.values, mode="lines+markers",
                                   line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                                   hovertemplate="%{x}: %{y:.1f} kWh/day<extra></extra>"))
        fig.update_layout(title="Mean daily consumption by month", yaxis_title="kWh per day",
                          yaxis_rangemode="tozero")
        show(fig, 360)

    l, r = st.columns(2)
    with l:
        cov = ctx["coverage"]
        fig = go.Figure(go.Histogram(x=cov.coverage * 100, xbins=dict(start=0, end=100.01, size=2.5),
                                     marker=dict(color=SERIES[0], line=dict(color="#fcfcfb", width=2)),
                                     hovertemplate="%{x}% coverage: %{y} homes<extra></extra>"))
        fig.add_vline(x=meta["coverage_min"] * 100, line=dict(color=INK, width=1.5, dash="dash"),
                      annotation_text=f"{meta['coverage_min']:.0%} cut-off", annotation_position="top left",
                      annotation_font=dict(color=INK2, size=11))
        fig.update_layout(title="Data coverage per household in 2013", xaxis_title="% of half hours recorded",
                          yaxis_title="Households", bargap=0)
        show(fig, 330)
    with r:
        h = ctx["hh_all"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Tariff type", "Acorn group"), horizontal_spacing=0.12)
        t = h.stdorToU.value_counts()
        a = h.Acorn_grouped.value_counts()
        fig.add_trace(go.Bar(x=t.index, y=t.values, marker=dict(color=SERIES[0], cornerradius=4),
                             hovertemplate="%{x}: %{y:,}<extra></extra>"), 1, 1)
        fig.add_trace(go.Bar(x=a.index, y=a.values, marker=dict(color=SERIES[0], cornerradius=4),
                             hovertemplate="%{x}: %{y:,}<extra></extra>"), 1, 2)
        fig.update_layout(title="Who is in the sample (profiling only, not clustered)", showlegend=False,
                          bargap=0.45)
        fig.update_yaxes(gridcolor=GRID)
        fig.update_annotations(font=dict(size=12, color=MUTED))
        show(fig, 330)

    l, r = st.columns(2)
    with l:
        corr = ctx["corr"]
        lab = [FEATURES[f][0] for f in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values, x=lab, y=lab, colorscale=DIVERGING, zmin=-1, zmax=1,
                                   text=np.round(corr.values, 2), texttemplate="%{text}",
                                   textfont=dict(size=10), xgap=2, ygap=2,
                                   hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
                                   colorbar=dict(thickness=10, outlinewidth=0)))
        fig.update_layout(title="Correlation between clustering features", yaxis=dict(autorange="reversed",
                          gridcolor="rgba(0,0,0,0)"), xaxis=dict(tickangle=-40))
        show(fig, 470)
    with r:
        pv = ctx["pca"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pv.component, y=pv.explained * 100, name="Per component",
                             marker=dict(color=SERIES[0], cornerradius=4),
                             hovertemplate="PC%{x}: %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=pv.component, y=pv.explained.cumsum() * 100, name="Cumulative",
                                 mode="lines+markers", line=dict(color=INK2, width=2), marker=dict(size=8),
                                 hovertemplate="Up to PC%{x}: %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=95, line=dict(color=MUTED, dash="dot", width=1), annotation_text="95% kept",
                      annotation_font=dict(color=MUTED, size=11))
        fig.update_layout(title="PCA explained variance", xaxis=dict(title="Principal component", dtick=1),
                          yaxis=dict(title="% of variance", range=[0, 105]), bargap=0.4)
        show(fig, 470)


# =====================================================================================
def models(ctx):
    meta, m = ctx["meta"], ctx["metrics"]
    k_sel, model_sel = meta["selected_k"], meta["selected_model"]
    header("Model comparison",
           "K-Means, Ward hierarchical clustering and Gaussian Mixture Models fitted for k = 2 to 10 and compared "
           "on internal validation scores and stability across 80% subsamples.")
    best = m[(m.model == model_sel) & (m.k == k_sel)].iloc[0]
    c = st.columns(4)
    kpi(c[0], "Selected", f"{model_sel.split(' ')[0]} (k = {k_sel})",
        "manual override" if (model_sel, k_sel) != (meta["auto_model"], meta["auto_k"]) else "composite rank, k = 3 to 8")
    kpi(c[1], "Silhouette", f"{best.silhouette:.3f}", "higher is better")
    kpi(c[2], "Davies-Bouldin", f"{best.davies_bouldin:.3f}", "lower is better")
    kpi(c[3], "Stability (ARI)", f"{best.ari_mean:.3f}", f"± {best.ari_std:.3f} over {meta['stability_reps']} subsamples")
    st.write("")

    panels = [("silhouette", "Silhouette score (higher is better)"),
              ("davies_bouldin", "Davies-Bouldin index (lower is better)"),
              ("calinski_harabasz", "Calinski-Harabasz index (higher is better)"),
              ("ari_mean", "Stability, mean ARI (higher is better)")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[1] for p in panels], vertical_spacing=0.16,
                        horizontal_spacing=0.08)
    for i, (col, _) in enumerate(panels):
        r, cc = i // 2 + 1, i % 2 + 1
        for j, mod in enumerate(["K-Means", "Hierarchical (Ward)", "GMM"]):
            g = m[m.model == mod]
            fig.add_trace(go.Scatter(x=g.k, y=g[col], name=mod, legendgroup=mod, showlegend=i == 0,
                                     mode="lines+markers",
                                     line=dict(color=SERIES[j], width=3 if mod == model_sel else 2),
                                     marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                                     hovertemplate=f"{mod}<br>k=%{{x}}: %{{y:.3f}}<extra></extra>"), r, cc)
        fig.add_vline(x=k_sel, line=dict(color=MUTED, dash="dash", width=1), row=r, col=cc)
    fig.update_xaxes(dtick=1, title_text="k (number of clusters)", row=2)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(gridcolor=GRID)
    fig.update_annotations(font=dict(size=13, color=INK))
    fig.update_layout(legend=dict(y=1.08), margin=dict(t=70))
    show(fig, 620)

    l, r = st.columns(2)
    with l:
        g = m[m.model == "K-Means"]
        fig = go.Figure(go.Scatter(x=g.k, y=g.inertia, mode="lines+markers", line=dict(color=SERIES[0], width=2),
                                   marker=dict(size=8), hovertemplate="k=%{x}: %{y:,.0f}<extra></extra>"))
        fig.add_vline(x=k_sel, line=dict(color=MUTED, dash="dash", width=1))
        fig.update_layout(title="Elbow method, K-Means inertia", xaxis=dict(dtick=1, title="k"),
                          yaxis_title="Within-cluster sum of squares")
        show(fig, 320)
    with r:
        g = m[m.model == "GMM"]
        fig = go.Figure(go.Scatter(x=g.k, y=g.bic, mode="lines+markers", line=dict(color=SERIES[2], width=2),
                                   marker=dict(size=8), customdata=g.gmm_covariance,
                                   hovertemplate="k=%{x}: BIC %{y:,.0f} (%{customdata})<extra></extra>"))
        fig.add_vline(x=k_sel, line=dict(color=MUTED, dash="dash", width=1))
        fig.update_layout(title="GMM Bayesian Information Criterion (lower is better)",
                          xaxis=dict(dtick=1, title="k"), yaxis_title="BIC")
        show(fig, 320)

    st.markdown(f"### Scores at k = {k_sel}")
    t = m[m.k == k_sel][["model", "silhouette", "davies_bouldin", "calinski_harabasz", "ari_mean",
                         "min_cluster_share"]].copy()
    t.columns = ["Model", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑", "Stability ARI ↑",
                 "Smallest cluster"]
    st.dataframe(t.style.format({"Silhouette ↑": "{:.3f}", "Davies-Bouldin ↓": "{:.3f}",
                                 "Calinski-Harabasz ↑": "{:,.0f}", "Stability ARI ↑": "{:.3f}",
                                 "Smallest cluster": "{:.1%}"})
                 .apply(lambda r: ["background-color:#e8f1fc; font-weight:600" if r.Model == model_sel else ""
                                   for _ in r], axis=1),
                 hide_index=True, width="stretch")
    with st.expander("All scores"):
        st.dataframe(m.round(4), hide_index=True, width="stretch")
    st.caption("Selection rule: among k = 3 to 8 with every cluster holding at least 3% of homes, the model with the "
               "best combined rank on Silhouette, Davies-Bouldin, Calinski-Harabasz and ARI stability. "
               "Interpretability of the load curves is then checked on the Segment profiles page.")


# =====================================================================================
def segments(ctx):
    hh, seg, nm = ctx["hh"], ctx["seg"], names(ctx)
    header("Segment profiles",
           "Each segment's average day, what makes it different, and where it sits in feature space. "
           "Pick a segment to highlight it across every chart.")
    opts = ["All segments"] + [nm[c] for c in seg.index]
    pick = st.segmented_control("Segment", opts, default="All segments", key="seg_pick",
                                label_visibility="collapsed") or "All segments"
    selected = None if pick == "All segments" else [c for c in seg.index if nm[c] == pick][0]

    l, r = st.columns([8, 4])
    with l:
        show(curve_fig(hh, nm, selected, title="Average daily load curve (shaded band: middle 50% of homes)"
                       if selected is not None else "Average daily load curve by segment"), 420)
    with r:
        if selected is None:
            st.markdown('<div class="card"><div class="big">All segments</div>'
                        f'<p style="color:{INK2};font-size:.9rem;margin:.5rem 0 0">Select a segment above to see '
                        'its statistics, typical range and recommended action. Shaded bands mark the 01:00 to '
                        '05:00 baseload window and the 17:00 to 21:00 evening peak.</p></div>',
                        unsafe_allow_html=True)
            counts = hh.cluster.value_counts()
            rows = "".join(f'<div class="row"><span>{chip(c)}{nm[c]}</span><b>{counts.get(c, 0):,}</b></div>'
                           for c in seg.index)
            st.markdown(f'<div class="card" style="margin-top:12px">{rows}</div>', unsafe_allow_html=True)
        else:
            g = hh[hh.cluster == selected]
            allm = hh[list(FEATURES)].mean()
            rows = ""
            for f in ["mean_daily_kwh", "baseload_kwh", "share_evening_peak", "peak_to_avg", "seasonal_ratio",
                      "weekend_ratio"]:
                lab, unit, fm = FEATURES[f]
                rows += (f'<div class="row"><span>{lab}</span><span><b>{fm.format(g[f].mean())}</b>'
                         f' <span style="color:{MUTED}">vs {fm.format(allm[f])}</span></span></div>')
            s = seg.loc[selected]
            st.markdown(f'<div class="card"><div class="big">{chip(selected)}{s["name"]}</div>'
                        f'<div style="color:{INK2};font-size:.88rem;margin:.2rem 0 .7rem">{len(g):,} households, '
                        f'{len(g) / max(len(hh), 1):.0%} of the filtered sample</div>{rows}'
                        f'<div style="margin-top:.8rem;font-size:.82rem;color:{MUTED};text-transform:uppercase;'
                        f'letter-spacing:.04em">Recommended action</div>'
                        f'<div style="color:{INK};font-size:.92rem;margin-top:2px">{s.action}</div></div>',
                        unsafe_allow_html=True)

    l, r = st.columns(2)
    with l:
        feats = list(FEATURES)
        means = hh.groupby("cluster")[feats].mean()
        z = (means - hh[feats].mean()) / hh[feats].std()
        text = [[FEATURES[f][2].format(means.loc[c, f]) for f in feats] for c in means.index]
        fig = go.Figure(go.Heatmap(
            z=z.values, x=[FEATURES[f][0] for f in feats], y=[nm[c] for c in means.index],
            colorscale=DIVERGING, zmid=0, zmin=-2, zmax=2, text=text, texttemplate="%{text}",
            textfont=dict(size=10), xgap=2, ygap=2,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<br>%{z:+.2f} SD vs average<extra></extra>",
            colorbar=dict(title=dict(text="SD", font=dict(size=11)), thickness=10, outlinewidth=0)))
        fig.update_layout(title="Feature fingerprint (colour: SD from average)",
                          yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"), xaxis=dict(tickangle=-35))
        show(fig, 430)
    with r:
        fig = go.Figure()
        for c in seg.index:
            g = hh[hh.cluster == c]
            on = selected is None or c == selected
            fig.add_trace(go.Scattergl(
                x=g.pc1, y=g.pc2, mode="markers", name=nm[c],
                marker=dict(size=7, color=color(c) if on else FADED, opacity=0.75 if on else 0.4,
                            line=dict(color="#fcfcfb", width=0.6)),
                customdata=np.c_[g.LCLid, g.mean_daily_kwh],
                hovertemplate=f"<b>{nm[c]}</b><br>%{{customdata[0]}}<br>%{{customdata[1]:.1f}} kWh/day<extra></extra>"))
        fig.update_layout(title="Households in PCA space", xaxis=dict(title="PC1", showgrid=True, gridcolor=GRID),
                          yaxis_title="PC2", legend=dict(itemclick="toggleothers"))
        show(fig, 430)

    l, r = st.columns(2)
    with l:
        f = st.selectbox("Feature", list(FEATURES), format_func=lambda x: f"{FEATURES[x][0]} ({FEATURES[x][1]})",
                         key="seg_box")
        fig = go.Figure()
        for c in seg.index:
            g = hh[hh.cluster == c]
            on = selected is None or c == selected
            fig.add_trace(go.Box(y=g[f], name=nm[c], marker=dict(color=color(c) if on else FADED, size=3),
                                 line=dict(width=1.5), fillcolor="rgba(0,0,0,0)", boxpoints="outliers",
                                 showlegend=False))
        fig.update_layout(title=f"{FEATURES[f][0]} by segment", yaxis_title=FEATURES[f][1])
        show(fig, 380)
    with r:
        st.write("")
        st.write("")
        mon = hh.groupby("cluster")[M_COLS].mean()
        fig = go.Figure()
        for c in [x for x in mon.index if x != selected] + ([selected] if selected in mon.index else []):
            on = selected is None or c == selected
            fig.add_trace(go.Scatter(x=MONTHS, y=mon.loc[c], name=nm[c], mode="lines",
                                     line=dict(color=color(c) if on else FADED, width=3 if c == selected else 2),
                                     hovertemplate=f"{nm[c]}: %{{y:.1f}} kWh/day<extra></extra>"))
        fig.update_layout(title="Seasonal pattern, mean daily kWh by month", hovermode="x unified",
                          yaxis=dict(title="kWh per day", rangemode="tozero"))
        show(fig, 380)


# =====================================================================================
def profiling(ctx):
    hh, seg, nm = ctx["hh"], ctx["seg"], names(ctx)
    header("Who is in each segment",
           "Tariff type and Acorn socio-demographic group were not used to form the clusters. Here they describe "
           "the segments after the fact.")
    order = [nm[c] for c in seg.index]

    def stacked(col, cats, colors, title):
        ct = pd.crosstab(hh.cluster.map(nm), hh[col], normalize="index").reindex(order).fillna(0)
        n = pd.crosstab(hh.cluster.map(nm), hh[col]).reindex(order).fillna(0)
        fig = go.Figure()
        for cat, colr in zip(cats, colors):
            if cat not in ct:
                continue
            fig.add_trace(go.Bar(y=ct.index, x=ct[cat] * 100, name=cat, orientation="h",
                                 marker=dict(color=colr, line=dict(color="#fcfcfb", width=2)),
                                 customdata=n[cat],
                                 text=[f"{v:.0%}" if v >= 0.07 else "" for v in ct[cat]],
                                 textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color="#ffffff" if colr not in ("#d6d5cf", "#cde2fb", "#86b6ef") else INK),
                                 hovertemplate=f"<b>%{{y}}</b><br>{cat}: %{{x:.1f}}% (%{{customdata:,}} homes)<extra></extra>"))
        overall = hh[col].value_counts(normalize=True)
        fig.update_layout(title=title, barmode="stack", legend=dict(traceorder="normal"),
                          xaxis=dict(title="% of segment", range=[0, 100.5], dtick=20),
                          yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"), bargap=0.3)
        return fig, ct, overall

    l, r = st.columns(2)
    with l:
        fig, ct_t, ov_t = stacked("stdorToU", ["Std", "ToU", "Unknown"], ["#d6d5cf", "#2a78d6", "#898781"],
                                  "Tariff type within each segment")
        show(fig, 380)
    with r:
        fig, ct_a, ov_a = stacked("Acorn_grouped", ["Affluent", "Comfortable", "Adversity", "ACORN-U", "Unknown"],
                                  ["#184f95", "#3987e5", "#86b6ef", "#d6d5cf", "#898781"],
                                  "Acorn group within each segment")
        show(fig, 380)

    st.markdown("### Over- and under-representation")
    st.caption("Index = share of the group inside the segment ÷ its share in the whole sample × 100. "
               "Above 100 means the group is more common in that segment than average.")
    idx = pd.concat([ct_t.div(ov_t, axis=1), ct_a.div(ov_a, axis=1)], axis=1) * 100
    idx = idx[[c for c in ["Std", "ToU", "Affluent", "Comfortable", "Adversity"] if c in idx]]
    fig = go.Figure(go.Heatmap(
        z=idx.values, x=idx.columns, y=idx.index, colorscale=DIVERGING, zmid=100, zmin=50, zmax=150,
        text=np.round(idx.values).astype(int), texttemplate="%{text}", textfont=dict(size=12), xgap=3, ygap=3,
        hovertemplate="<b>%{y}</b><br>%{x}: index %{z:.0f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_layout(yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"), xaxis=dict(side="top"),
                      margin=dict(t=40))
    show(fig, 60 + 46 * len(idx))


# =====================================================================================
def savings(ctx):
    seg, ef = ctx["seg"], ctx["ef"]
    header("Savings scenario",
           "If each segment adopted its recommended action, how much electricity and carbon would be saved per year? "
           "Adjust the adoption rate and the expected reduction per segment.")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.slider("Adoption rate (share of homes that act)", 0, 100, st.session_state.get("adoption", 30), 5,
                  format="%d%%", key="adoption_w",
                  on_change=lambda: st.session_state.update(adoption=st.session_state["adoption_w"]))
        st.caption(f"Emission factor: {ef:.3f} kg CO2e/kWh ({ctx['ef_label']}). Change it in the sidebar.")
    sc = scenario(ctx)
    with c2:
        ed = st.data_editor(
            sc[["cluster", "Segment", "Targeted load", "Reduction %"]], hide_index=True,
            width="stretch", disabled=["cluster", "Segment", "Targeted load"],
            column_config={"cluster": None,
                           "Reduction %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1,
                                                                        format="%d%%",
                                                                        help="Cut in the targeted load for homes that act")},
            key="red_editor")
        new = {int(r.cluster): int(r["Reduction %"]) for _, r in ed.iterrows()}
        if new != st.session_state.get("reductions"):
            st.session_state["reductions"] = new
            st.rerun()

    total_kwh = sc["kWh saved"].sum()
    total_use = ctx["hh"].annual_kwh.sum()
    avg_home = ctx["hh"].annual_kwh.mean()
    k = st.columns(4)
    kpi(k[0], "Electricity saved", f"{total_kwh / 1000:,.0f} MWh", "per year")
    kpi(k[1], "Carbon avoided", f"{total_kwh * ef / 1000:,.1f} t CO2e", "per year")
    kpi(k[2], "Equivalent homes", f"{total_kwh / avg_home:,.0f}", "average annual use of a sample home")
    kpi(k[3], "Share of sample use", f"{total_kwh / total_use:.1%}", "of total household kWh")
    st.write("")

    l, r = st.columns([5, 7])
    with l:
        s = sc.sort_values("kg CO2e avoided")
        fig = go.Figure(go.Bar(y=s.Segment, x=s["kg CO2e avoided"] / 1000, orientation="h",
                               marker=dict(color=[color(c) for c in s.cluster], cornerradius=4),
                               text=[f"{v / 1000:,.1f} t" for v in s["kg CO2e avoided"]], textposition="outside",
                               textfont=dict(color=INK2), cliponaxis=False,
                               hovertemplate="<b>%{y}</b><br>%{x:,.1f} t CO2e/yr<extra></extra>"))
        fig.update_layout(title="Carbon avoided by segment (t CO2e per year)",
                          xaxis=dict(showticklabels=False, range=[0, s["kg CO2e avoided"].max() / 1000 * 1.4]),
                          yaxis=dict(gridcolor="rgba(0,0,0,0)"), bargap=0.35)
        show(fig, 360)
    with r:
        t = pd.DataFrame({
            "Segment": sc.Segment, "Homes": sc.Households,
            "kWh saved per acting home": sc["Target kWh per home"] * sc["Reduction %"] / 100,
            "MWh saved": sc["kWh saved"] / 1000, "t CO2e": sc["kg CO2e avoided"] / 1000})
        st.markdown("### Breakdown")
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"Homes": st.column_config.NumberColumn(format="%,d"),
                                    "kWh saved per acting home": st.column_config.NumberColumn(format="%,.0f"),
                                    "MWh saved": st.column_config.NumberColumn(format="%,.1f"),
                                    "t CO2e": st.column_config.NumberColumn(format="%,.2f")})
        st.caption("Saved kWh = homes in segment × adoption rate × reduction % × targeted kWh per home. "
                   "Carbon = saved kWh × grid emission factor.")

    st.markdown(f"""
<div class="card" style="margin-top:8px"><div class="big">Link to SDG 7 and UTP Energy Security</div>
<p style="color:{INK2};font-size:.92rem;margin:.4rem 0 0">Cutting always-on load and shifting use out of the
17:00 to 21:00 peak improves end-use efficiency (SDG Target 7.3) and lowers demand at the hour when the grid relies
most on costly peaking plant, which supports energy security. The carbon figures give a measurable secondary benefit
under SDG 13. London results may not transfer directly to Malaysia, where cooling rather than heating drives
seasonal demand.</p></div>""", unsafe_allow_html=True)
