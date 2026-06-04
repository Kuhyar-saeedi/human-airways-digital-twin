"""
core/rag.py
===========
Lightweight RAG (Retrieval-Augmented Generation) for the Human Airways project.

Architecture
------------
Retrieval: TF-IDF vectoriser (sklearn) + cosine similarity — runs fully offline,
           no model downloads.
Generation: Anthropic Claude API (claude-haiku) — requires ANTHROPIC_API_KEY.
            If no key is set, the page shows retrieved context only.

Knowledge base
--------------
Static documents covering the project methodology, dataset, and results.
Dynamically extended with computed statistics when precomputed data is available.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Knowledge base documents ─────────────────────────────────────────────────

_DOCS: List[dict] = [
    {
        "id": "project_overview",
        "title": "Project Overview",
        "content": """
The Human Airways Digital Twin is a computational surrogate model for the human
respiratory system, built for the Digital Twin Methods course at Università degli
Studi di Roma Tor Vergata.

The project processes 100 CFD snapshots from Ansys Twin Builder, each representing
a different airway geometry described by 26 anatomical parameters. The goal is to
predict the static pressure field (in Pascals) for any new airway geometry in
milliseconds — replacing hours of CFD computation.

The pipeline uses Proper Orthogonal Decomposition (POD / PCA) to compress the data
into a low-dimensional space, and Radial Basis Function (RBF) interpolation to
build a fast surrogate model. Latin Hypercube Sampling (LHS) generates 1000 virtual
airway geometries for design space exploration.
""",
    },
    {
        "id": "dataset",
        "title": "Dataset Description",
        "content": """
The dataset has 100 Design-of-Experiment (DOE) runs generated with Ansys Twin Builder
using CFD (Computational Fluid Dynamics) simulation.

Each run contains:
- 3D mesh node coordinates: 2,135,906 nodes × 3 coordinates (X, Y, Z) in metres
- Static pressure field: 2,135,906 scalar values in Pascals
- 26 geometry parameters (DOE variables)

Files are stored in Twin Builder binary format (.bin): an 8-byte little-endian uint64
header gives the number of float64 values, followed by the raw data.

For visualisation the mesh is sub-sampled at stride=50, yielding ~42,718 displayed
nodes without losing spatial patterns.
""",
    },
    {
        "id": "pod",
        "title": "POD — Proper Orthogonal Decomposition",
        "content": """
POD (Proper Orthogonal Decomposition), also known as PCA (Principal Component Analysis)
in statistics, decomposes the snapshot matrix into spatial modes and modal coefficients.

Mathematical steps:
1. Form snapshot matrix X of shape (100 snapshots × N_features)
2. Centre: X_c = X − mean(X)
3. Economy SVD: X_c = U · diag(σ) · V^T
4. Spatial modes = columns of V  (where variation occurs in space)
5. Modal scores = U · diag(σ)    (how each snapshot uses each mode)
6. Energy of mode i = σᵢ² / Σ σⱼ²

The key insight: a small number of modes captures most of the variance. For this
airway dataset, ~10 geometry modes capture 95% of shape variance, and ~8 pressure
modes capture 99% of pressure variance.

Reconstruction: field ≈ mean + modes @ coefficients
""",
    },
    {
        "id": "rbf",
        "title": "RBF Surrogate — Radial Basis Function Interpolation",
        "content": """
The RBF (Radial Basis Function) surrogate maps 26 geometry parameters to pressure
POD modal scores. Once trained on the 100 DOE snapshots it can predict the full
pressure field for any new parameter set in milliseconds.

How it works:
  f(x) = Σᵢ wᵢ φ(‖x − xᵢ‖) + polynomial tail

The default kernel is thin_plate_spline: φ(r) = r² log(r), which is smooth and works
well for scattered high-dimensional data.

Training: fit weights from DOE params (100×26) → pressure POD scores (100×k).
Inference: new 26-dim parameter vector → k predicted scores → pressure field via POD reconstruction.

Accuracy is quantified via Leave-One-Out (LOO) cross-validation: each snapshot is
predicted by a model trained on the other 99.
""",
    },
    {
        "id": "lhs",
        "title": "LHS — Latin Hypercube Sampling",
        "content": """
Latin Hypercube Sampling (LHS) generates 1000 virtual airway geometries within the
bounds of the original 100 DOE runs, providing much richer design space coverage than
pure random sampling.

Why LHS instead of random: LHS divides each parameter dimension into n equal intervals
and places exactly one sample per interval per dimension, guaranteeing uniform marginal
distributions. This avoids clustering and large empty regions.

In this project: 1000 LHS samples are evaluated using the RBF surrogate (no CFD needed),
producing 1000 predicted pressure fields in seconds rather than years of simulation time.
""",
    },
    {
        "id": "parameters",
        "title": "The 26 Geometry Parameters",
        "content": """
The 26 DOE parameters describe the human airway anatomy:

Main airway:
- A_glotis: Glottis area (mm²) — the narrowing between vocal cords
- A_epiglotis: Epiglottis area (mm²) — flap protecting the airway during swallowing
- r_curvature: Curvature radius of the trachea bend (mm)
- d_trachea: Trachea diameter (mm) — main windpipe diameter
- l_trachea: Trachea length (mm)
- teta_branch_trachea: Branching angle at the carina (°)

First-level bronchi (left/right):
- l_l, l_r: Left and right main bronchus lengths (mm)
- teta_branch_l, teta_branch_r: Left and right branching angles (°)
- l_ll, l_lr, l_rl, l_rr: Second-level branch lengths (mm)
- teta_branch_ll/lr/rl/rr: Second-level branching angles (°)

Third-level sub-branches:
- l_lll, l_llr, l_lrl, l_lrr: Left sub-branch lengths (mm)
- l_rll, l_rlr, l_rrl, l_rrr: Right sub-branch lengths (mm)
""",
    },
    {
        "id": "anatomy",
        "title": "Airway Anatomy and Regions",
        "content": """
The human airway modelled starts from the larynx (voice box) and branches down to
the third-level bronchi:

Mouth/Larynx → Glottis/Epiglottis → Trachea (main windpipe)
    ├── Left main bronchus (L)
    │     ├── Left-Left (LL) → LLL, LLR sub-branches
    │     └── Left-Right (LR) → LRL, LRR sub-branches
    └── Right main bronchus (R)
          ├── Right-Left (RL) → RLL, RLR sub-branches
          └── Right-Right (RR) → RRL, RRR sub-branches

The trachea carries air from the throat to the lungs. It splits at the carina into
left and right main bronchi. Each bronchus branches twice more, reaching the lobar
bronchi that supply the lung lobes.

Static pressure decreases from mouth to alveoli (from ~0 Pa gauge at mouth to negative
values deeper in the airway during inhalation). Resistance — pressure drop per unit flow —
is the key clinical metric: higher resistance = harder to breathe.
""",
    },
    {
        "id": "streamlit_app",
        "title": "Streamlit Dashboard Pages",
        "content": """
The dashboard has the following pages:
1. Landing page (app.py) — project overview, key metrics
2. Geometry Explorer — POD mode sliders for interactive shape morphing
3. Pressure Field — 3D pressure map for each of the 100 DOE snapshots
4. DOE Analysis — parameter distribution, parallel coordinates
5. POD Analysis — energy curves, mode shapes, reconstruction error
6. RBF Inference — predict pressure field at any new set of 26 parameters
7. Regional Analysis — mean pressure per anatomical region
8. Design Space — parameter sensitivity, design landscape map
9. Animations — animated POD mode sweeps and snapshot reels
10. Ask AI — RAG-powered Q&A about the project

Run with: streamlit run app.py
""",
    },
    {
        "id": "workflow",
        "title": "Digital Twin Workflow",
        "content": """
The complete Digital Twin pipeline:

Step 1 — Inspect the database: explore 100 DOE snapshots, geometry parameters,
          pressure statistics, and 3D point cloud visualisation.

Step 2 — POD reduction of geometry: SVD on the (100 × N×3) coordinate matrix.
          ~10 modes capture 95% of shape variance.

Step 3 — Upscale with 1000 virtual shapes: Latin Hypercube Sampling generates
          1000 parameter sets within the original DOE bounds.

Step 4 — Evaluate with RBF surrogate: trained on DOE data, predicts pressure
          POD scores at any new geometry in under 1 millisecond.

Step 5 — POD reduction of pressure: SVD on the (100 × N) pressure matrix.
          ~8 modes capture 99% of pressure variance.

Step 6 — RBF inference: 26 input parameters → POD scores → full pressure field
          via POD reconstruction.

Step 7 — Interactive dashboard: Streamlit web app and PyQt desktop app.

Step 8/9 — Export: precomputed NPY cache, VTK files for ParaView.
""",
    },
    {
        "id": "performance",
        "title": "Performance and Accuracy",
        "content": """
Precomputation (run once):
- Loading 100 binary snapshots: ~30 seconds
- SVD for geometry POD (100 × 128K matrix): ~5 seconds
- SVD for pressure POD (100 × 43K matrix): ~2 seconds
- Total: ~40 seconds to build all caches

Inference speed (after precomputation):
- App startup with cached NPZ files: < 1 second
- POD reconstruction (numpy matmul): < 1 millisecond
- RBF prediction for one new geometry: < 1 millisecond

Accuracy:
- POD reconstruction with 10 modes: <0.5% relative error on geometry
- POD reconstruction with 8 modes: <1% relative error on pressure
- RBF LOO cross-validation: typically <5% error in POD score space

The main approximations are:
- Stride-50 subsampling for visualisation (full mesh has 2,135,906 nodes)
- Truncated POD (99% energy retained)
- RBF interpolation error (quantified by LOO cross-validation)
""",
    },
    {
        "id": "qt_app",
        "title": "PyQt Desktop Application",
        "content": """
A real-time desktop application built with PyQt5 and PyVista provides instant,
sub-10ms response to slider changes — much faster than the Streamlit web interface.

Features:
- Geometry Explorer tab: POD mode sliders → real-time airway morphing in 3D (PyVista)
- Pressure Field tab: snapshot selector → instant 3D pressure field rendering
- RBF Inference tab: 26 parameter sliders → predict pressure in real time

Architecture:
- Loads precomputed NPZ files on startup (< 1 second)
- QSlider signals connected to PyVista renderer via 50ms debounce timer
- PyVista point cloud coloured by pressure scalar field

Run with: python qt_app/run.py
""",
    },
    {
        "id": "online_deployment",
        "title": "Online Deployment",
        "content": """
The Streamlit app is deployed online for free via Streamlit Community Cloud:

1. Run scripts/precompute.py --pod-only to generate precomputed/ folder (~40 MB)
2. Push code + precomputed/ folder + DOE CSV + settings.json to a GitHub repo
3. Connect the repo on share.streamlit.io
4. Set ANTHROPIC_API_KEY in Streamlit Secrets for the Ask AI page (optional)

Note: raw .bin files (100 snapshots × 64 MB = 6.4 GB) are NOT pushed to GitHub.
The deployed app runs entirely from precomputed NPZ files.

The PyQt desktop app runs locally only — it cannot be deployed to a web server.
""",
    },
]


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

class RAGKnowledgeBase:
    """TF-IDF retrieval over the project knowledge base."""

    def __init__(self):
        self._docs  = _DOCS
        self._texts = [d["title"] + "\n" + d["content"] for d in _DOCS]
        self._tfidf = None
        self._matrix = None
        self._build()

    # ── Build index ───────────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build(self):
        corpus = self._texts
        tokenised = [self._tokenise(t) for t in corpus]
        vocab = sorted({tok for doc in tokenised for tok in doc})
        self._vocab = {w: i for i, w in enumerate(vocab)}

        n_docs, n_words = len(corpus), len(vocab)
        tf = np.zeros((n_docs, n_words), dtype=np.float32)
        for d, tokens in enumerate(tokenised):
            for tok in tokens:
                if tok in self._vocab:
                    tf[d, self._vocab[tok]] += 1
            row_sum = tf[d].sum()
            if row_sum > 0:
                tf[d] /= row_sum

        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((n_docs + 1) / (df + 1)) + 1.0

        self._matrix = tf * idf
        # L2-normalise rows
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix /= norms

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        tokens = self._tokenise(query)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for tok in tokens:
            if tok in self._vocab:
                vec[self._vocab[tok]] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        scores = self._matrix @ vec
        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._docs[i]) for i in top_idx]

    def add_document(self, doc: dict):
        """Add a runtime document (e.g. computed statistics) and rebuild index."""
        self._docs.append(doc)
        self._texts.append(doc["title"] + "\n" + doc["content"])
        self._build()


# ── Claude API generation ─────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Get Anthropic API key from environment or Streamlit secrets."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def generate_answer(query: str, context_chunks: List[str]) -> str | None:
    """
    Call Claude API with retrieved context to generate an answer.
    Returns None if no API key is available.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    context = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are an expert assistant for the Human Airways Digital Twin project, "
        "built at Università degli Studi di Roma Tor Vergata. Answer concisely and "
        "technically accurately. Use the provided context. If the context does not "
        "contain enough information, say so briefly."
    )
    prompt = f"Context from project documentation:\n\n{context}\n\nQuestion: {query}"

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Singleton ─────────────────────────────────────────────────────────────────

_kb: RAGKnowledgeBase | None = None


def get_knowledge_base() -> RAGKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = RAGKnowledgeBase()
    return _kb
