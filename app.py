"""Household Electricity Segmentation Dashboard (TEB3123 Group 16).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

if __name__ == "__main__":
    # Started with plain `python app.py` (e.g. VS Code's Run button): relaunch under Streamlit.
    from streamlit import runtime
    if not runtime.exists():
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__,
                                  "--server.headless", "false"]))

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "dashboard"))
import views as pages  # noqa: E402
from style import CSS  # noqa: E402

st.set_page_config(page_title="Household Energy Segments", page_icon=":material/bolt:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

DATA = ROOT / "data"
EMISSION_FACTORS = {  # kg CO2e per kWh, UK grid electricity (generation), DESNZ conversion factors
    "DESNZ 2025 (0.177)": 0.177,
    "DESNZ 2026 (0.131)": 0.131,
}


@st.cache_data
def load(mtime: float):
    d = {
        "meta": json.loads((DATA / "meta.json").read_text()),
        "hh": pd.read_parquet(DATA / "households.parquet"),
        "metrics": pd.read_csv(DATA / "model_metrics.csv"),
        "segments": pd.read_csv(DATA / "segment_labels.csv"),
        "seasonal": pd.read_csv(DATA / "seasonal_curves.csv"),
        "coverage": pd.read_csv(DATA / "coverage.csv"),
        "outliers": pd.read_csv(DATA / "outliers.csv"),
        "corr": pd.read_csv(DATA / "feature_corr.csv", index_col=0),
        "pca": pd.read_csv(DATA / "pca_variance.csv"),
    }
    return d


if not (DATA / "meta.json").exists():
    st.error("No dashboard data found. Run `python pipeline/make_demo_data.py` (demo) or "
             "`python pipeline/prepare_data.py kaggle --root <path>` (real data) first.")
    st.stop()

d = load((DATA / "meta.json").stat().st_mtime)
seg = d["segments"].set_index("cluster")

# ---------- Sidebar: global filters ----------
with st.sidebar:
    st.markdown("### Filters")
    tariffs = sorted(d["hh"].stdorToU.unique())
    tariff = st.selectbox("Tariff type", ["All"] + tariffs,
                          help="Std = flat rate, ToU = dynamic time-of-use trial")
    acorns = sorted(d["hh"].Acorn_grouped.unique())
    acorn = st.multiselect("Acorn group", acorns, default=acorns)
    ef_choice = st.radio("Grid emission factor", list(EMISSION_FACTORS) + ["Custom"], index=0)
    ef = (st.number_input("kg CO2e per kWh", 0.0, 1.5, 0.177, 0.001, format="%.3f")
          if ef_choice == "Custom" else EMISSION_FACTORS[ef_choice])
    st.caption("Filters apply to segment, profiling and savings pages. Model comparison always uses "
               "every clustered household.")

hh = d["hh"]
mask = hh.Acorn_grouped.isin(acorn)
if tariff != "All":
    mask &= hh.stdorToU == tariff
if mask.sum() == 0:
    st.warning("No households match the current filters. Select at least one Acorn group or another tariff.")
    st.stop()
ctx = dict(d, hh_all=hh, hh=hh[mask], seg=seg, ef=ef, ef_label=ef_choice)

nav = st.navigation({
    "Story": [
        st.Page(lambda: pages.overview(ctx), title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    ],
    "Analysis": [
        st.Page(lambda: pages.eda(ctx), title="Data and EDA", icon=":material/query_stats:", url_path="eda"),
        st.Page(lambda: pages.models(ctx), title="Model comparison", icon=":material/compare_arrows:", url_path="models"),
        st.Page(lambda: pages.segments(ctx), title="Segment profiles", icon=":material/stacked_line_chart:", url_path="segments"),
        st.Page(lambda: pages.profiling(ctx), title="Who is in each segment", icon=":material/groups:", url_path="profiling"),
    ],
    "Impact": [
        st.Page(lambda: pages.savings(ctx), title="Savings scenario", icon=":material/eco:", url_path="savings"),
    ],
})
if d["meta"]["source"] == "demo":
    st.markdown('<div class="demo"><b>Demo data.</b> These charts use synthetic households so the layout can be '
                'reviewed. Run <code>pipeline/prepare_data.py</code> on the London smart meter data to replace them.'
                '</div>', unsafe_allow_html=True)
nav.run()
