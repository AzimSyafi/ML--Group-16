"""Shared colours, Plotly template and small UI helpers."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Validated categorical palette (fixed order; colour follows the cluster id, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, PAGE = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f9f9f7"
FADED = "#d6d5cf"
PEAK_WASH = "rgba(237,161,0,0.10)"      # 17:00-21:00 evening peak band
BASE_WASH = "rgba(42,120,214,0.07)"     # 01:00-05:00 baseload band
DIVERGING = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f1a09f"], [1, "#b8302f"]]
SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]

FONT = 'system-ui, -apple-system, "Segoe UI", Inter, sans-serif'

pio.templates["energy"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=13, color=INK2),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    colorway=SERIES,
    margin=dict(l=16, r=20, t=36, b=16),
    title=dict(font=dict(size=14, color=INK), x=0.012, xanchor="left", xref="container", y=0.975, yanchor="top"),
    xaxis=dict(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, zeroline=False,
               title=dict(font=dict(size=12, color=MUTED))),
    yaxis=dict(gridcolor=GRID, gridwidth=1, linecolor="rgba(0,0,0,0)", zeroline=False,
               title=dict(font=dict(size=12, color=MUTED))),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="",
                font=dict(size=12, color=INK2)),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
))
pio.templates.default = "energy"

HH_TICKS = dict(tickmode="array", tickvals=list(range(0, 49, 6)),
                ticktext=[f"{h:02d}:00" for h in range(0, 25, 3)])


def color(c: int) -> str:
    return SERIES[int(c) % len(SERIES)]


def add_bands(fig, row=None, col=None):
    """Shade the baseload (01-05) and evening peak (17-21) windows on a 48-slot x axis."""
    kw = dict(row=row, col=col) if row else {}
    fig.add_vrect(x0=2, x1=10, fillcolor=BASE_WASH, line_width=0, layer="below",
                  annotation_text="Baseload window", annotation_position="top left",
                  annotation_font=dict(size=11, color=MUTED), **kw)
    fig.add_vrect(x0=34, x1=42, fillcolor=PEAK_WASH, line_width=0, layer="below",
                  annotation_text="Evening peak", annotation_position="top left",
                  annotation_font=dict(size=11, color=MUTED), **kw)


CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
h1 {{ font-size: 1.75rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; margin-top: .6rem !important; }}
.subtitle {{ color: {INK2}; margin-top: -0.6rem; margin-bottom: 1.2rem; font-size: 0.98rem; }}
.kpi {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08); border-radius: 12px;
        padding: 14px 16px 12px; min-height: 118px; }}
.kpi .label {{ color: {INK2}; font-size: 0.8rem; font-weight: 500; text-transform: uppercase;
               letter-spacing: .04em; }}
.kpi .value {{ color: {INK}; font-size: 1.75rem; font-weight: 650; line-height: 1.25; margin-top: 4px; }}
.kpi .note {{ color: {MUTED}; font-size: 0.8rem; margin-top: 2px; }}
.seg-chip {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px;
             vertical-align: baseline; }}
.card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08); border-radius: 12px; padding: 16px 18px; }}
.card .big {{ font-size: 1.25rem; font-weight: 650; color:{INK}; }}
.card .row {{ display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {GRID};
              font-size: .9rem; color:{INK2}; }}
.card .row b {{ color:{INK}; font-weight:600; }}
.demo {{ background:#fff7e6; border:1px solid #f3d38a; color:#5a4300; border-radius:10px; padding:8px 14px;
         font-size:.88rem; margin-bottom: 1rem; }}
div[data-testid="stPlotlyChart"] {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.08);
         border-radius:12px; padding:0; overflow:hidden; }}
section[data-testid="stSidebar"] {{ background: #f1f0ec; }}
</style>
"""


def kpi(col, label, value, note=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def header(title, subtitle):
    st.title(title)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def chip(c):
    return f'<span class="seg-chip" style="background:{color(c)}"></span>'
