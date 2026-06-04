"""
app.py — Human Airways Digital Twin Dashboard
==============================================
Landing page. Describes the project, dataset, and methodology.
Navigate to the analysis pages via the sidebar.

Course: Digital Twin Methods
Institution: Università degli Studi di Roma Tor Vergata
"""

import streamlit as st

st.set_page_config(
    page_title="Human Airways Digital Twin",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🫁 Human Airways Digital Twin Dashboard")
st.markdown(
    "**Digital Twin Methods** | Università degli Studi di Roma Tor Vergata"
)
st.divider()

# ── Quick metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("DOE Snapshots",        "100")
c2.metric("Geometry Parameters",  "26")
c3.metric("Mesh Nodes (full)",    "2,135,906")
c4.metric("Displayed Nodes",      "~42,700  (stride 50)")
c5.metric("Pressure Field",       "Static (Pa)")

st.divider()

# ── Two-column layout ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Project Overview")
    st.markdown("""
The dataset comes from **Ansys Twin Builder / CFD** simulations of the
human respiratory system. Each of the 100 Design-of-Experiment (DOE) runs
uses a different airway geometry, described by **26 parameters**:
glottis area, epiglottis area, trachea dimensions, and branching angles
at three levels of the bronchial tree.

For each run the simulator provides:
- The **3D mesh node coordinates** (deformed geometry, ~48 MB/snapshot)
- The **static pressure field** in Pascals (~16 MB/snapshot)

The Digital Twin pipeline transforms this raw CFD data into a fast,
interactive surrogate that can predict pressure at any new geometry
in milliseconds rather than hours.
""")

    st.subheader("Airway Anatomy Modelled")
    st.markdown("""
```
Mouth / Larynx
     │
  Glottis / Epiglottis
     │
  Upper Trachea (top / middle / bottom)
     ├── Left main bronchus (L)
     │     ├── Left-Left (LL) ─── LLL, LLR
     │     └── Left-Right (LR) ── LRL, LRR
     └── Right main bronchus (R)
           ├── Right-Left (RL) ── RLL, RLR
           └── Right-Right (RR) ─ RRL, RRR
```
""")

with right:
    st.subheader("Methodology — 9 Steps")
    steps = [
        ("1 · Inspect the database",
         "100 DOE snapshots explored: geometry parameters, pressure statistics, "
         "3D point cloud visualisation."),
        ("2 · POD reduction of geometry",
         "SVD on the (100 × N×3) coordinate matrix. "
         "~10 modes capture 95 % of shape variance."),
        ("3 · Upscale with 1000 virtual shapes",
         "Latin Hypercube Sampling generates 1000 parameter sets "
         "within the original DOE bounds."),
        ("4 · Evaluate with RBF surrogate",
         "RBF (thin-plate spline) trained on DOE data predicts "
         "pressure POD scores at any new geometry in <1 ms."),
        ("5 · POD reduction of pressure",
         "SVD on the (100 × N) pressure matrix. "
         "~8 modes capture 99 % of pressure variance."),
        ("6 · RBF inference in reduced spaces",
         "Coupled geometry–pressure RBF maps 26 input parameters "
         "to the full pressure field via POD reconstruction."),
        ("7 · Interactive Streamlit dashboard",
         "Shape morphing via POD sliders (hull SSM Viewer style), "
         "pressure field browser, DOE explorer, regional analysis."),
        ("8 · Export for rbfVR / post-processing",
         "CSV export of DOE table, LHS samples, "
         "and predicted pressure fields."),
        ("9 · Export synthesised geometries",
         "VTK-compatible CSV of node coordinates for external "
         "CFD validation workflows."),
    ]
    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

st.divider()

# ── Navigation guide ───────────────────────────────────────────────────────────
st.subheader("Dashboard Pages")
pages = [
    ("🔬 Geometry Explorer",  "Morph the airway shape via POD mode sliders — Hull SSM Viewer style."),
    ("🌡️ Pressure Field",     "3D pressure map for any of the 100 DOE snapshots."),
    ("📐 DOE Analysis",        "Parallel coordinates, scatter plots, correlation heatmap, LHS coverage."),
    ("📊 POD Analysis",        "Energy curves, mode shapes, and reconstruction error for geometry and pressure."),
    ("🔮 RBF Inference",       "Predict the full pressure field at any new set of 26 geometry parameters."),
    ("🫀 Regional Analysis",   "Mean pressure per anatomical region across snapshots."),
]
cols = st.columns(3)
for i, (name, desc) in enumerate(pages):
    with cols[i % 3]:
        st.info(f"**{name}**\n\n{desc}")
