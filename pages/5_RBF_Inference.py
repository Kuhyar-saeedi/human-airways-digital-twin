"""
pages/5_RBF_Inference.py
=========================
RBF surrogate model for pressure prediction at unseen geometries.

Workflow
--------
1. Compute pressure POD (mean + modes + scores) from the 100 training snapshots.
2. Build an RBF interpolant mapping 26 DOE parameters → pressure POD scores.
3. User adjusts the 26 geometry parameter sliders.
4. RBF predicts POD scores for the new parameter set.
5. Pressure field = mean + modes @ predicted_scores.
6. Display the predicted 3D pressure field.

Also shows
----------
  Leave-One-Out cross-validation error to quantify RBF accuracy.
  1000-sample LHS upscaling: visualise predicted mean pressure for virtual shapes.
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
    load_all_pressures,
    load_ref_coords,
    load_doe,
    get_param_cols,
    PARAM_LABELS,
)
from core.pod import compute_pod, modes_for_energy, reconstruct
from core.rbf import build_rbf, predict, loo_errors
from core.lhs import latin_hypercube, doe_bounds

st.set_page_config(page_title="RBF Inference", page_icon="🔮", layout="wide")
st.title("🔮 RBF Surrogate — Pressure Inference at New Geometries")
st.caption(
    "Radial Basis Function interpolation maps 26 geometry parameters "
    "to the full pressure field via the POD reduced space."
)

# ── Load data ──────────────────────────────────────────────────────────────────
doe_df     = load_doe()
param_cols = get_param_cols(doe_df)
ref_coords = load_ref_coords(STRIDE)

with st.spinner("Loading pressure snapshots…"):
    P_pres = load_all_pressures(STRIDE)

@st.cache_data(show_spinner=False)
def pres_pod(P):
    return compute_pod(P)

with st.spinner("Computing pressure POD…"):
    mean_pres, modes_pres, scores_pres, sv_pres = pres_pod(P_pres)

# Select number of modes for RBF (default: modes capturing 99% energy)
k_rbf_default = modes_for_energy(sv_pres, 0.99)

# DOE parameter matrix (normalised 0-1 for RBF stability)
params_raw = doe_df[param_cols].values.astype(float)
low, high  = params_raw.min(axis=0), params_raw.max(axis=0)
params_norm = (params_raw - low) / (high - low + 1e-12)

with st.sidebar:
    st.header("RBF Settings")
    k_rbf  = st.slider("POD modes for RBF", 1, min(40, len(sv_pres)), k_rbf_default)
    kernel = st.selectbox("RBF kernel",
                          ["thin_plate_spline", "multiquadric", "linear", "cubic"],
                          index=0)
    st.divider()
    st.caption(
        "Tip: thin_plate_spline is smooth and works well for "
        "scattered data in high dimensions."
    )

# Build RBF on POD scores (truncated to k_rbf modes)
@st.cache_data(show_spinner=False)
def get_rbf(params, scores, k, kern):
    return build_rbf(params, scores[:, :k], kernel=kern)

with st.spinner("Building RBF surrogate…"):
    rbf_model = get_rbf(params_norm, scores_pres, k_rbf, kernel)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_pred, tab_loo, tab_lhs = st.tabs([
    "🎯 Predict New Shape",
    "📉 LOO Validation",
    "🌐 1000 Virtual Shapes",
])

# ─ Tab 1: Prediction ───────────────────────────────────────────────────────────
with tab_pred:
    st.subheader("Set Geometry Parameters → Predict Pressure Field")

    # Organise 26 sliders into 3 expanders for readability
    new_params = np.empty(len(param_cols))

    groups = {
        "Main airway (glottis, trachea)": [
            "A_glotis", "A_epiglotis", "r_curvature",
            "d_trachea", "l_trachea", "teta_branch_trachea",
        ],
        "First-level branches (L / R)": [
            "l_l", "l_r", "teta_branch_l", "teta_branch_r",
            "l_ll", "l_lr", "l_rl", "l_rr",
            "teta_branch_ll", "teta_branch_lr", "teta_branch_rl", "teta_branch_rr",
        ],
        "Second-level sub-branches": [
            "l_lll", "l_llr", "l_lrl", "l_lrr",
            "l_rll", "l_rlr", "l_rrl", "l_rrr",
        ],
    }

    for grp_name, grp_cols in groups.items():
        with st.expander(grp_name, expanded=(grp_name == "Main airway (glottis, trachea)")):
            n_cols = 3
            cols_ui = st.columns(n_cols)
            for j, col in enumerate(grp_cols):
                i = param_cols.index(col)
                col_min = float(params_raw[:, i].min())
                col_max = float(params_raw[:, i].max())
                col_mean = float(params_raw[:, i].mean())
                new_params[i] = cols_ui[j % n_cols].slider(
                    PARAM_LABELS.get(col, col),
                    min_value=col_min,
                    max_value=col_max,
                    value=col_mean,
                    step=float((col_max - col_min) / 50),
                    key=f"rbf_{col}",
                    format="%.2f",
                )

    if st.button("🔮 Predict Pressure Field", type="primary"):
        # Normalise new params same way as training data
        new_norm = ((new_params - low) / (high - low + 1e-12)).reshape(1, -1)
        pred_scores = predict(rbf_model, new_norm)[0]  # (k,)

        # Reconstruct full pressure field
        pred_pressure = reconstruct(mean_pres, modes_pres[:, :k_rbf], pred_scores)

        # Display
        st.success("Pressure field predicted successfully!")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Min pressure",  f"{pred_pressure.min():.1f} Pa")
        p2.metric("Max pressure",  f"{pred_pressure.max():.1f} Pa")
        p3.metric("Mean pressure", f"{pred_pressure.mean():.1f} Pa")
        p4.metric("Std deviation", f"{pred_pressure.std():.1f} Pa")

        fig_pred = go.Figure(go.Scatter3d(
            x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
            mode="markers",
            marker=dict(
                size=2, color=pred_pressure, colorscale="Jet",
                colorbar=dict(title="Pressure (Pa)", thickness=14),
                opacity=0.65,
            ),
            hovertemplate="P: %{marker.color:.1f} Pa<extra></extra>",
        ))
        fig_pred.update_layout(
            scene=dict(aspectmode="data", bgcolor="rgb(12,14,20)",
                       xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
                       zaxis=dict(gridcolor="#2a2a2a")),
            paper_bgcolor="rgb(12,14,20)", font=dict(color="white"),
            height=650, margin=dict(l=0, r=0, b=0, t=40),
            title="RBF-Predicted Static Pressure",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

# ─ Tab 2: LOO validation ───────────────────────────────────────────────────────
with tab_loo:
    st.subheader("Leave-One-Out Cross-Validation")
    st.markdown("""
    Each snapshot is removed in turn; the RBF is rebuilt on the remaining 99
    and used to predict the removed one.  This gives an honest estimate of
    generalisation error without a separate test set.
    """)

    if st.button("Run LOO validation (takes ~30 s)"):
        with st.spinner("Running LOO cross-validation…"):
            errors = loo_errors(params_norm, scores_pres[:, :k_rbf], kernel=kernel)

        err_df = pd.DataFrame({
            "Snapshot":    list(range(1, 101)),
            "LOO Error":   errors.tolist(),
        })

        fig_loo = px.bar(
            err_df, x="Snapshot", y="LOO Error",
            color="LOO Error", color_continuous_scale="Reds",
            title="LOO cross-validation error per snapshot (‖predicted − actual‖ in POD score space)",
            labels={"LOO Error": "‖error‖"},
        )
        fig_loo.update_layout(height=400)
        st.plotly_chart(fig_loo, use_container_width=True)

        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Mean LOO error", f"{errors.mean():.4f}")
        lc2.metric("Max LOO error",  f"{errors.max():.4f}")
        lc3.metric("Min LOO error",  f"{errors.min():.4f}")

# ─ Tab 3: 1000 virtual shapes ─────────────────────────────────────────────────
with tab_lhs:
    st.subheader("Database Upscaling: 1000 Virtual Shapes via LHS + RBF")
    st.markdown("""
    Latin Hypercube Sampling generates 1000 new parameter vectors within the
    original DOE bounds. The RBF surrogate predicts the pressure POD scores for
    each, enabling reconstruction of 1000 pressure fields without any CFD run.
    """)

    @st.cache_data(show_spinner=False)
    def lhs_predictions(params_norm_arr, lo, hi, mean_p, modes_p, k, kern):
        low_raw = lo; high_raw = hi
        lhs_raw  = latin_hypercube(1000, low_raw, high_raw, seed=42)
        lhs_norm = (lhs_raw - lo) / (hi - lo + 1e-12)
        rbf_tmp  = build_rbf(params_norm_arr, scores_pres[:, :k], kernel=kern)
        pred_sc  = predict(rbf_tmp, lhs_norm)
        # Reconstruct mean pressure for each virtual shape
        mean_pressures = np.array([
            reconstruct(mean_p, modes_p[:, :k], pred_sc[i]).mean()
            for i in range(len(lhs_raw))
        ])
        return lhs_raw, mean_pressures

    with st.spinner("Predicting pressure for 1000 LHS shapes…"):
        lhs_raw, mean_preds = lhs_predictions(
            params_norm, low, high, mean_pres, modes_pres, k_rbf, kernel
        )

    # Show mean pressure distribution over virtual shapes
    fig_hist = px.histogram(
        x=mean_preds, nbins=60,
        labels={"x": "Mean Predicted Pressure (Pa)"},
        title="Distribution of mean static pressure across 1000 virtual airway shapes",
        color_discrete_sequence=["#4EB3D3"],
    )
    fig_hist.update_layout(height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter: first two DOE params vs mean predicted pressure
    pa_idx, pb_idx = 0, 3  # A_glotis, d_trachea
    fig_scatter = px.scatter(
        x=lhs_raw[:, pa_idx], y=lhs_raw[:, pb_idx],
        color=mean_preds,
        color_continuous_scale="Jet",
        labels={
            "x": PARAM_LABELS.get(param_cols[pa_idx], param_cols[pa_idx]),
            "y": PARAM_LABELS.get(param_cols[pb_idx], param_cols[pb_idx]),
            "color": "Mean Pressure (Pa)",
        },
        title="Mean predicted pressure across LHS design space",
    )
    fig_scatter.update_layout(height=460)
    st.plotly_chart(fig_scatter, use_container_width=True)
