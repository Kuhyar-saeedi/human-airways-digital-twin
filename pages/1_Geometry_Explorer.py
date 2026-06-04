"""
pages/1_Geometry_Explorer.py
============================
POD-based interactive shape viewer with real-time slider response.

Performance: uses @st.fragment so only the 3D view re-renders when sliders
move, not the entire page. Data loading and POD computation run once.

Precomputed path: if precomputed/pod_geometry.npz exists (run
scripts/precompute.py first), the SVD is skipped entirely — startup is instant.
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
    STRIDE,
    load_all_coords,
    load_doe,
    load_ref_coords,
    get_param_cols,
    load_precomputed_pod,
)
from core.pod import compute_pod, cumulative_energy, modes_for_energy, reconstruct

st.set_page_config(page_title="Geometry Explorer", page_icon="🔬", layout="wide")

st.title("🔬 Geometry Explorer — POD Shape Morphing")
st.caption(
    "Adjust mode sliders to morph the airway geometry. "
    "Uses @st.fragment — only the 3D view re-renders on each slider move."
)

# ── Data loading (runs once, cached) ──────────────────────────────────────────

doe_df = load_doe()

# Try precomputed first (instant), fall back to recomputing from raw .bin files
_pre = load_precomputed_pod("geometry")
if _pre is not None:
    mean_geo   = _pre["mean"]
    modes_geo  = _pre["modes"]
    scores_geo = _pre["scores"]
    svals_geo  = _pre["svalues"]
    st.sidebar.success("Loaded precomputed POD", icon="⚡")
else:
    with st.spinner("Loading geometry snapshots and computing POD…"):
        X_geo = load_all_coords(STRIDE)

    @st.cache_data(show_spinner=False)
    def geo_pod(X):
        return compute_pod(X)

    mean_geo, modes_geo, scores_geo, svals_geo = geo_pod(X_geo)

energy   = cumulative_energy(svals_geo)
n95      = modes_for_energy(svals_geo, 0.95)
n99      = modes_for_energy(svals_geo, 0.99)
std_devs = svals_geo / np.sqrt(max(len(scores_geo) - 1, 1))

# ── Static sidebar controls (page-level, re-run page when changed) ────────────
with st.sidebar:
    st.header("POD Mode Controls")
    st.caption(f"95 % energy → {n95} modes  |  99 % energy → {n99} modes")
    n_sliders = st.slider("Modes to display", 1, min(20, len(svals_geo)), 10)
    st.divider()
    st.caption("Tip: drag sliders below to morph the geometry in real time.")

# ── Fragment: everything that re-renders on slider change ─────────────────────

@st.fragment
def geometry_viewer(n_sliders, mean_geo, modes_geo, scores_geo, svals_geo, std_devs, energy, doe_df):
    # Initialise session state keys
    for i in range(n_sliders):
        if f"geo_mode_{i}" not in st.session_state:
            st.session_state[f"geo_mode_{i}"] = 0.0

    # Action buttons inside the fragment → only fragment re-runs
    with st.sidebar:
        btn1, btn2 = st.columns(2)
        if btn1.button("Reset to Mean", key="btn_reset"):
            for i in range(n_sliders):
                st.session_state[f"geo_mode_{i}"] = 0.0
            st.rerun(scope="fragment")

        if btn2.button("Random Shape", key="btn_random"):
            rng = np.random.default_rng()
            for i in range(n_sliders):
                st.session_state[f"geo_mode_{i}"] = float(
                    rng.uniform(-2.0 * std_devs[i], 2.0 * std_devs[i])
                )
            st.rerun(scope="fragment")

        snap_pick = st.selectbox(
            "Load a real snapshot",
            ["— choose —"] + [f"Run {i}" for i in range(1, 101)],
            key="snap_pick_geo",
        )
        if snap_pick != "— choose —":
            idx = int(snap_pick.split()[1]) - 1
            for i in range(n_sliders):
                st.session_state[f"geo_mode_{i}"] = float(scores_geo[idx, i])
            st.rerun(scope="fragment")

        st.divider()
        coeffs = np.zeros(len(svals_geo))
        for i in range(n_sliders):
            mode_energy_pct = (
                energy[i] * 100 if i == 0 else (energy[i] - energy[i - 1]) * 100
            )
            coeffs[i] = st.slider(
                f"Mode {i + 1}  ({mode_energy_pct:.1f} %)",
                min_value=float(-3.0 * std_devs[i]),
                max_value=float( 3.0 * std_devs[i]),
                value=float(st.session_state[f"geo_mode_{i}"]),
                step=float(std_devs[i] / 30),
                key=f"geo_mode_{i}",
                format="%.3f",
            )

    # Reconstruct shape — fast numpy matmul
    rec_coords = reconstruct(mean_geo, modes_geo, coeffs).reshape(-1, 3)

    col_3d, col_info = st.columns([3, 1])

    with col_3d:
        fig3d = go.Figure(
            go.Scatter3d(
                x=rec_coords[:, 0], y=rec_coords[:, 1], z=rec_coords[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="#4EB3D3", opacity=0.55),
                hoverinfo="skip",
                name="Airway",
            )
        )
        fig3d.update_layout(
            scene=dict(
                xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                aspectmode="data", bgcolor="rgb(15, 17, 25)",
                xaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgb(15,17,25)"),
                yaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgb(15,17,25)"),
                zaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgb(15,17,25)"),
            ),
            paper_bgcolor="rgb(15, 17, 25)", font=dict(color="white"),
            height=640, margin=dict(l=0, r=0, b=0, t=40),
            title="Reconstructed Airway Geometry",
        )
        st.plotly_chart(fig3d, use_container_width=True, key="geo_3d")

    with col_info:
        k_plot = min(30, len(svals_geo))
        fig_e = px.line(
            x=list(range(1, k_plot + 1)),
            y=(energy[:k_plot] * 100).tolist(),
            labels={"x": "Number of Modes", "y": "Cumulative Energy (%)"},
            title="POD Energy",
        )
        fig_e.add_hline(y=95, line_dash="dash", line_color="orange",
                        annotation_text="95 %", annotation_position="bottom right")
        fig_e.add_hline(y=99, line_dash="dash", line_color="red",
                        annotation_text="99 %", annotation_position="bottom right")
        fig_e.update_layout(height=280, margin=dict(l=10, r=10, b=10, t=40))
        st.plotly_chart(fig_e, use_container_width=True, key="geo_energy")

        st.metric("Modes for 95 %", n95)
        st.metric("Modes for 99 %", n99)
        st.metric("Total modes", len(svals_geo))

        # Deviation from mean as a simple scalar
        dev = float(np.linalg.norm(coeffs[:n_sliders]))
        st.metric("Shape deviation ‖c‖", f"{dev:.2f}")

        st.markdown("**Active coefficients**")
        coeff_df = pd.DataFrame({
            "Mode": [f"M{i+1}" for i in range(n_sliders)],
            "σᵢ":   [f"{std_devs[i]:.3g}" for i in range(n_sliders)],
            "cᵢ":   [f"{coeffs[i]:.3g}"   for i in range(n_sliders)],
        })
        st.dataframe(coeff_df, hide_index=True, height=220)


geometry_viewer(n_sliders, mean_geo, modes_geo, scores_geo, svals_geo, std_devs, energy, doe_df)
