"""
pages/8_Animations.py
=====================
Animated visualisations to build intuition about the POD decomposition.

Animations
----------
1. Mode Sweep  — morph the airway geometry from -3σ to +3σ along a chosen mode
2. Snapshot Reel — play through all 100 pressure field snapshots as a movie
3. Pressure Mode — animate the pressure field along a POD mode axis
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import (
    STRIDE, load_all_coords, load_all_pressures,
    load_ref_coords, load_precomputed_pod,
)
from core.pod import compute_pod, modes_for_energy, reconstruct

st.set_page_config(page_title="Animations", page_icon="🎬", layout="wide")
st.title("🎬 Animated POD Visualisations")
st.caption(
    "Watch the airway geometry and pressure field morph as you sweep through POD modes. "
    "Use the play button in the chart footer."
)

# ── Load data ─────────────────────────────────────────────────────────────────

_pre_g = load_precomputed_pod("geometry")
_pre_p = load_precomputed_pod("pressure")

with st.spinner("Loading geometry data…"):
    if _pre_g:
        mean_geo   = _pre_g["mean"]
        modes_geo  = _pre_g["modes"]
        scores_geo = _pre_g["scores"]
        svals_geo  = _pre_g["svalues"]
    else:
        X_geo = load_all_coords(STRIDE)
        @st.cache_data(show_spinner=False)
        def _geo_pod(X): return compute_pod(X)
        mean_geo, modes_geo, scores_geo, svals_geo = _geo_pod(X_geo)

with st.spinner("Loading pressure data…"):
    if _pre_p:
        mean_pres   = _pre_p["mean"]
        modes_pres  = _pre_p["modes"]
        scores_pres = _pre_p["scores"]
        svals_pres  = _pre_p["svalues"]
    else:
        P_all = load_all_pressures(STRIDE)
        @st.cache_data(show_spinner=False)
        def _pres_pod(P): return compute_pod(P)
        mean_pres, modes_pres, scores_pres, svals_pres = _pres_pod(P_all)

ref_coords = load_ref_coords(STRIDE)
std_geo    = svals_geo  / np.sqrt(max(len(scores_geo)  - 1, 1))
std_pres   = svals_pres / np.sqrt(max(len(scores_pres) - 1, 1))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_sweep, tab_reel, tab_pmode = st.tabs([
    "🔬 Geometry Mode Sweep",
    "🎞️ Pressure Snapshot Reel",
    "💨 Pressure Mode Sweep",
])


def make_animation_layout(title: str, height: int = 620) -> dict:
    return dict(
        scene=dict(
            aspectmode="data", bgcolor="rgb(12,14,20)",
            xaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
            yaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
            zaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
        ),
        paper_bgcolor="rgb(12,14,20)", font=dict(color="white"),
        height=height, margin=dict(l=0, r=0, b=0, t=50),
        title=title,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0, x=0.5, xanchor="center", yanchor="top",
            pad=dict(t=10),
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[f"frame{k}"],
                   dict(mode="immediate", frame=dict(duration=80, redraw=True))],
                   label=str(k)) for k in range(0)],  # filled per animation
            transition=dict(duration=0),
            x=0, y=0, currentvalue=dict(visible=False),
            len=1.0,
        )],
    )


# ── Tab 1: Geometry Mode Sweep ─────────────────────────────────────────────────
with tab_sweep:
    st.subheader("Airway Geometry — POD Mode Sweep")

    col_ctl, _ = st.columns([1, 3])
    with col_ctl:
        mode_idx = st.slider("POD Mode to animate", 1, min(10, len(svals_geo)), 1,
                             key="anim_geo_mode") - 1
        n_frames = st.slider("Number of frames", 20, 60, 30, key="anim_geo_frames")
        sigma_range = st.slider("σ range (±)", 1.0, 4.0, 3.0, step=0.5, key="anim_geo_sigma")

    sigma = float(std_geo[mode_idx])
    alpha_vals = np.linspace(-sigma_range * sigma, sigma_range * sigma, n_frames)

    @st.cache_data(show_spinner=False)
    def build_geo_sweep(mode_idx, n_frames, sigma_range, mean_g, modes_g, svals_g):
        std = svals_g / np.sqrt(max(99, 1))
        sigma = float(std[mode_idx])
        alphas = np.linspace(-sigma_range * sigma, sigma_range * sigma, n_frames)
        frames = []
        for alpha in alphas:
            coeffs = np.zeros(modes_g.shape[1])
            coeffs[mode_idx] = alpha
            rec = reconstruct(mean_g, modes_g, coeffs).reshape(-1, 3)
            frames.append(rec)
        return frames, alphas

    with st.spinner("Building geometry animation frames…"):
        geo_frames, alphas = build_geo_sweep(
            mode_idx, n_frames, sigma_range, mean_geo, modes_geo, svals_geo
        )

    plotly_frames = []
    for k, rec in enumerate(geo_frames):
        plotly_frames.append(go.Frame(
            data=[go.Scatter3d(
                x=rec[:, 0], y=rec[:, 1], z=rec[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="#4EB3D3", opacity=0.55),
                hoverinfo="skip",
            )],
            name=f"frame{k}",
        ))

    fig_sweep = go.Figure(
        data=[go.Scatter3d(
            x=geo_frames[0][:, 0], y=geo_frames[0][:, 1], z=geo_frames[0][:, 2],
            mode="markers",
            marker=dict(size=1.5, color="#4EB3D3", opacity=0.55),
            hoverinfo="skip",
        )],
        frames=plotly_frames,
    )
    layout = make_animation_layout(f"Geometry Mode {mode_idx+1} Sweep (±{sigma_range}σ)")
    layout["sliders"][0]["steps"] = [
        dict(method="animate",
             args=[[f"frame{k}"], dict(mode="immediate", frame=dict(duration=80, redraw=True))],
             label=f"{alphas[k]/sigma:.1f}σ")
        for k in range(n_frames)
    ]
    fig_sweep.update_layout(**layout)
    st.plotly_chart(fig_sweep, use_container_width=True, key="fig_geo_sweep")

    e_pct = float(
        svals_geo[mode_idx]**2 / (svals_geo**2).sum() * 100
    )
    st.info(f"Mode {mode_idx+1} captures **{e_pct:.1f}%** of total geometry variance.  "
            f"σ = {sigma:.4g} m  |  range = ±{sigma_range*sigma:.4g} m")


# ── Tab 2: Pressure Snapshot Reel ─────────────────────────────────────────────
with tab_reel:
    st.subheader("Play through all 100 pressure field snapshots")

    col_ctl2, _ = st.columns([1, 3])
    with col_ctl2:
        frame_step = st.slider("Step (every N snapshots)", 1, 5, 1, key="reel_step")

    snap_indices = list(range(0, 100, frame_step))

    @st.cache_data(show_spinner=False)
    def build_pressure_reel(snap_indices, mean_p, modes_p, scores_p):
        fields = []
        for idx in snap_indices:
            from core.pod import reconstruct as _rec
            field = _rec(mean_p, modes_p, scores_p[idx])
            fields.append(field)
        return fields

    with st.spinner("Building pressure reel…"):
        pressure_fields = build_pressure_reel(snap_indices, mean_pres, modes_pres, scores_pres)

    p_global_min = min(f.min() for f in pressure_fields)
    p_global_max = max(f.max() for f in pressure_fields)

    reel_frames = []
    for k, (idx, field) in enumerate(zip(snap_indices, pressure_fields)):
        reel_frames.append(go.Frame(
            data=[go.Scatter3d(
                x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=field.tolist(),
                    colorscale="Jet",
                    cmin=p_global_min, cmax=p_global_max,
                    opacity=0.65,
                ),
                hoverinfo="skip",
            )],
            name=f"reel{k}",
        ))

    fig_reel = go.Figure(
        data=[go.Scatter3d(
            x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
            mode="markers",
            marker=dict(
                size=2, color=pressure_fields[0].tolist(),
                colorscale="Jet",
                cmin=p_global_min, cmax=p_global_max,
                colorbar=dict(title="P (Pa)", thickness=14),
                opacity=0.65,
            ),
            hoverinfo="skip",
        )],
        frames=reel_frames,
    )
    reel_layout = make_animation_layout("Pressure Field — DOE Snapshot Reel")
    reel_layout["sliders"][0]["steps"] = [
        dict(method="animate",
             args=[[f"reel{k}"], dict(mode="immediate", frame=dict(duration=100, redraw=True))],
             label=f"Run {snap_indices[k]+1}")
        for k in range(len(snap_indices))
    ]
    fig_reel.update_layout(**reel_layout)
    st.plotly_chart(fig_reel, use_container_width=True, key="fig_reel")

    st.caption(f"Showing {len(snap_indices)} frames  |  global P range: {p_global_min:.0f} – {p_global_max:.0f} Pa")


# ── Tab 3: Pressure Mode Sweep ─────────────────────────────────────────────────
with tab_pmode:
    st.subheader("Pressure Field — POD Mode Sweep")
    st.markdown(
        "Animates the pressure field by sweeping a single POD mode coefficient "
        "from −3σ to +3σ, holding all others at zero (= mean pressure field)."
    )

    col_ctl3, _ = st.columns([1, 3])
    with col_ctl3:
        pmode_idx   = st.slider("Pressure POD Mode", 1, min(10, len(svals_pres)), 1,
                                key="anim_pmode") - 1
        n_frames_p  = st.slider("Frames", 20, 60, 30, key="anim_pframes")
        sigma_range_p = st.slider("σ range (±)", 1.0, 4.0, 3.0, step=0.5, key="anim_psigma")

    sigma_p  = float(std_pres[pmode_idx])
    alphas_p = np.linspace(-sigma_range_p * sigma_p, sigma_range_p * sigma_p, n_frames_p)

    @st.cache_data(show_spinner=False)
    def build_pressure_sweep(pmode_idx, n_frames, sigma_range, mean_p, modes_p, svals_p):
        std = svals_p / np.sqrt(max(99, 1))
        sigma = float(std[pmode_idx])
        alphas = np.linspace(-sigma_range * sigma, sigma_range * sigma, n_frames)
        fields = []
        for alpha in alphas:
            coeffs = np.zeros(modes_p.shape[1])
            coeffs[pmode_idx] = alpha
            fields.append(mean_p + modes_p @ coeffs)
        return fields, alphas

    with st.spinner("Building pressure mode animation…"):
        p_sweep_fields, alphas_p = build_pressure_sweep(
            pmode_idx, n_frames_p, sigma_range_p, mean_pres, modes_pres, svals_pres
        )

    p_sweep_min = min(f.min() for f in p_sweep_fields)
    p_sweep_max = max(f.max() for f in p_sweep_fields)

    pmode_frames = []
    for k, field in enumerate(p_sweep_fields):
        pmode_frames.append(go.Frame(
            data=[go.Scatter3d(
                x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=field.tolist(),
                    colorscale="Jet",
                    cmin=p_sweep_min, cmax=p_sweep_max,
                    opacity=0.65,
                ),
                hoverinfo="skip",
            )],
            name=f"pm{k}",
        ))

    fig_pmode = go.Figure(
        data=[go.Scatter3d(
            x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
            mode="markers",
            marker=dict(
                size=2, color=p_sweep_fields[0].tolist(),
                colorscale="Jet",
                cmin=p_sweep_min, cmax=p_sweep_max,
                colorbar=dict(title="P (Pa)", thickness=14),
                opacity=0.65,
            ),
            hoverinfo="skip",
        )],
        frames=pmode_frames,
    )
    pmode_layout = make_animation_layout(f"Pressure POD Mode {pmode_idx+1} Sweep (±{sigma_range_p}σ)")
    pmode_layout["sliders"][0]["steps"] = [
        dict(method="animate",
             args=[[f"pm{k}"], dict(mode="immediate", frame=dict(duration=80, redraw=True))],
             label=f"{alphas_p[k]/sigma_p:.1f}σ")
        for k in range(n_frames_p)
    ]
    fig_pmode.update_layout(**pmode_layout)
    st.plotly_chart(fig_pmode, use_container_width=True, key="fig_pmode")

    ep_pct = float(svals_pres[pmode_idx]**2 / (svals_pres**2).sum() * 100)
    st.info(f"Pressure Mode {pmode_idx+1} captures **{ep_pct:.1f}%** of total pressure variance.")
