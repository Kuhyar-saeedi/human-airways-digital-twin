"""
pages/7_Design_Space.py
========================
Design Landscape — understand the parameter space and what drives pressure.

Tabs
----
1. Parameter Sensitivity  — which of the 26 geometry params most affects pressure
2. Design Landscape       — 2D map of all 100 runs in POD score space, colored by pressure
3. Param–Pressure Matrix  — scatter grid of key params vs mean pressure
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import (
    STRIDE, load_doe, load_all_pressures, get_param_cols,
    PARAM_LABELS, load_precomputed_pod,
)
from core.pod import compute_pod, modes_for_energy

st.set_page_config(page_title="Design Space", page_icon="🗺️", layout="wide")
st.title("🗺️ Design Space — What Drives Airway Pressure?")
st.caption("Explore which geometric parameters matter most and how the design space is structured.")

# ── Load data ─────────────────────────────────────────────────────────────────
doe_df     = load_doe()
param_cols = get_param_cols(doe_df)
params_raw = doe_df[param_cols].values.astype(float)

with st.spinner("Loading pressure snapshots…"):
    P_all = load_all_pressures(STRIDE)

mean_pressure_per_run = P_all.mean(axis=1)   # (100,) mean pressure per snapshot

# POD for pressure (precomputed or computed)
_pre = load_precomputed_pod("pressure")
if _pre is not None:
    scores_pres = _pre["scores"]
    sv_pres     = _pre["svalues"]
else:
    @st.cache_data(show_spinner=False)
    def _pres_pod(P):
        return compute_pod(P)
    with st.spinner("Computing pressure POD…"):
        _, _, scores_pres, sv_pres = _pres_pod(P_all)

# POD for geometry (precomputed or computed)
_pre_g = load_precomputed_pod("geometry")
if _pre_g is not None:
    scores_geo = _pre_g["scores"]
else:
    from core.data_io import load_all_coords
    @st.cache_data(show_spinner=False)
    def _geo_pod(X):
        return compute_pod(X)
    with st.spinner("Computing geometry POD…"):
        _, _, scores_geo, _ = _geo_pod(load_all_coords(STRIDE))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_sens, tab_land, tab_matrix = st.tabs([
    "📊 Parameter Sensitivity",
    "🌐 Design Landscape",
    "🔢 Param–Pressure Grid",
])

# ── Tab 1: Parameter Sensitivity ──────────────────────────────────────────────
with tab_sens:
    st.subheader("Which parameter drives mean airway pressure the most?")
    st.markdown(
        "Pearson correlation coefficient between each of the 26 geometry parameters "
        "and the **mean static pressure** across the 100 DOE snapshots.  "
        "Red = positive correlation (larger → higher pressure), "
        "blue = negative."
    )

    corr = np.array([
        float(np.corrcoef(params_raw[:, i], mean_pressure_per_run)[0, 1])
        for i in range(len(param_cols))
    ])
    labels = [PARAM_LABELS.get(c, c) for c in param_cols]

    sens_df = pd.DataFrame({
        "Parameter":   labels,
        "Raw name":    param_cols,
        "Correlation": corr,
        "AbsCorr":     np.abs(corr),
    }).sort_values("AbsCorr", ascending=True)

    fig_sens = px.bar(
        sens_df, x="Correlation", y="Parameter",
        orientation="h",
        color="Correlation",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Pearson Correlation: Parameter → Mean Pressure",
        labels={"Correlation": "r"},
        height=700,
    )
    fig_sens.update_layout(yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_sens, width='stretch')

    col_top, col_bot = st.columns(2)
    top3 = sens_df.nlargest(3, "AbsCorr")[["Parameter", "Correlation"]]
    with col_top:
        st.markdown("**Top 3 most influential parameters**")
        st.dataframe(top3.round(4), hide_index=True)

    # Also show correlation with first POD mode score
    corr_mode1 = np.array([
        float(np.corrcoef(params_raw[:, i], scores_pres[:, 0])[0, 1])
        for i in range(len(param_cols))
    ])
    sens_m1 = pd.DataFrame({
        "Parameter":   labels,
        "r with Mode 1": corr_mode1,
        "AbsCorr":     np.abs(corr_mode1),
    }).sort_values("AbsCorr", ascending=False)

    with col_bot:
        st.markdown("**Top 3 params correlated with Pressure Mode 1**")
        st.dataframe(sens_m1[["Parameter","r with Mode 1"]].head(3).round(4), hide_index=True)

    st.divider()
    st.markdown("**Full sensitivity table**")
    full_df = pd.DataFrame({
        "Parameter":          labels,
        "r (mean pressure)":  corr.round(4),
        "r (pressure mode 1)":corr_mode1.round(4),
    }).sort_values("r (mean pressure)", key=np.abs, ascending=False)
    st.dataframe(full_df, hide_index=True, width='stretch')


# ── Tab 2: Design Landscape ───────────────────────────────────────────────────
with tab_land:
    st.subheader("All 100 DOE runs mapped into the pressure POD space")
    st.markdown(
        "Each point is one CFD snapshot, plotted by its first two **pressure POD scores**. "
        "Color = mean static pressure.  "
        "Hover to see the snapshot number and key geometry parameters."
    )

    # Build hover text
    hover_texts = []
    for i, row in doe_df.iterrows():
        snap = int(row["snapshot_num"])
        key_params = ", ".join(
            f"{PARAM_LABELS.get(c,c).split('(')[0].strip()}: {row[c]:.1f}"
            for c in ["A_glotis", "d_trachea", "l_trachea", "teta_branch_trachea"]
        )
        hover_texts.append(f"Run {snap}<br>{key_params}<br>Mean P: {mean_pressure_per_run[i]:.1f} Pa")

    fig_land = go.Figure(go.Scatter(
        x=scores_pres[:, 0], y=scores_pres[:, 1],
        mode="markers+text",
        text=[str(int(s)) for s in doe_df["snapshot_num"]],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=10,
            color=mean_pressure_per_run,
            colorscale="Jet",
            colorbar=dict(title="Mean P (Pa)", thickness=14),
            line=dict(width=0.5, color="white"),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig_land.update_layout(
        title="Design Landscape — Pressure POD Score Space",
        xaxis_title="Pressure POD Mode 1 score",
        yaxis_title="Pressure POD Mode 2 score",
        height=600,
        paper_bgcolor="rgb(15,17,25)", plot_bgcolor="rgb(20,22,30)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_land, width='stretch')

    # Same for geometry space
    st.subheader("Geometry POD space")
    fig_geo_land = go.Figure(go.Scatter(
        x=scores_geo[:, 0], y=scores_geo[:, 1],
        mode="markers",
        marker=dict(
            size=10, color=mean_pressure_per_run,
            colorscale="Jet",
            colorbar=dict(title="Mean P (Pa)", thickness=14),
            line=dict(width=0.5, color="white"),
        ),
        hovertext=hover_texts, hoverinfo="text",
    ))
    fig_geo_land.update_layout(
        title="Design Landscape — Geometry POD Score Space",
        xaxis_title="Geometry POD Mode 1 score",
        yaxis_title="Geometry POD Mode 2 score",
        height=500,
        paper_bgcolor="rgb(15,17,25)", plot_bgcolor="rgb(20,22,30)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_geo_land, width='stretch')

    # Geometry vs Pressure POD scores scatter
    st.subheader("Geometry POD Mode 1 vs Pressure POD Mode 1")
    # Add a linear trendline manually
    x_cross = scores_geo[:, 0]
    y_cross = scores_pres[:, 0]
    m_cross, b_cross = np.polyfit(x_cross, y_cross, 1)
    x_line = np.linspace(x_cross.min(), x_cross.max(), 50)

    fig_cross = px.scatter(
        x=x_cross, y=y_cross,
        color=mean_pressure_per_run, color_continuous_scale="Jet",
        labels={"x": "Geo Mode 1", "y": "Pres Mode 1", "color": "Mean P (Pa)"},
        title="Cross-space correlation: does geometry Mode 1 predict pressure Mode 1?",
    )
    fig_cross.add_scatter(x=x_line, y=m_cross * x_line + b_cross,
                          mode="lines", line=dict(color="white", dash="dash"),
                          name=f"OLS (slope={m_cross:.3f})", showlegend=True)
    fig_cross.update_layout(height=420)
    st.plotly_chart(fig_cross, width='stretch')


# ── Tab 3: Param–Pressure Grid ────────────────────────────────────────────────
with tab_matrix:
    st.subheader("Key parameter vs mean pressure scatter plots")

    # Pick top 6 most correlated parameters
    top_params = pd.DataFrame({
        "col":    param_cols,
        "label":  labels,
        "abscorr": np.abs(corr),
    }).nlargest(6, "abscorr")

    cols_grid = st.columns(3)
    for idx, (_, row_p) in enumerate(top_params.iterrows()):
        col = row_p["col"]
        lbl = row_p["label"]
        i = param_cols.index(col)
        with cols_grid[idx % 3]:
            xd = params_raw[:, i]
            yd = mean_pressure_per_run
            m, b = np.polyfit(xd, yd, 1)
            x_fit = np.linspace(xd.min(), xd.max(), 40)
            fig_s = px.scatter(
                x=xd, y=yd,
                labels={"x": lbl, "y": "Mean P (Pa)"},
                color=yd, color_continuous_scale="Jet",
                title=lbl.split("(")[0].strip(),
            )
            fig_s.add_scatter(x=x_fit, y=m * x_fit + b,
                              mode="lines", line=dict(color="white", dash="dash"),
                              showlegend=False)
            fig_s.update_layout(
                height=300, margin=dict(l=10, r=10, b=30, t=40),
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_s, width='stretch')

    # Full parameter correlation heatmap
    st.subheader("Parameter–parameter correlation matrix")
    corr_matrix = np.corrcoef(params_raw.T)
    fig_heat = px.imshow(
        corr_matrix,
        x=labels, y=labels,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-1, zmax=1,
        title="Pearson correlation between all 26 DOE parameters",
        height=700,
    )
    fig_heat.update_layout(
        xaxis=dict(tickfont=dict(size=9), tickangle=45),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_heat, width='stretch')
