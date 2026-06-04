"""
core/rbf.py
===========
Radial Basis Function (RBF) surrogate model for parameter-space inference.

How RBF inference works
-----------------------
Given training pairs  (pᵢ, yᵢ)  where pᵢ ∈ ℝᵈ are DOE parameter vectors
and yᵢ ∈ ℝᵏ are POD modal scores, the RBF interpolant is:

    f(x) = Σᵢ wᵢ φ(‖x − pᵢ‖)  +  polynomial tail

where φ is the radial basis function.  Weights wᵢ are found by solving
the linear system  Φ w = y.

Common kernels
--------------
  thin_plate_spline : φ(r) = r² log(r)    — smooth, widely used
  multiquadric      : φ(r) = √(r² + ε²)  — good for scattered data
  gaussian          : φ(r) = exp(−r²/ε²) — compact support

Usage in this project
---------------------
  1. Build RBF from DOE params (100 × 26) → pressure POD scores (100 × k)
  2. At inference time: supply new 26-dim parameter vector → get k scores
  3. Reconstruct pressure field: mean + modes @ predicted_scores
"""

import numpy as np
from scipy.interpolate import RBFInterpolator


def build_rbf(
    params: np.ndarray,
    targets: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
) -> RBFInterpolator:
    """
    Fit an RBF interpolator.

    Parameters
    ----------
    params    : (n, d)  input parameter matrix (DOE values)
    targets   : (n, k)  target matrix (POD scores)
    kernel    : RBF kernel name (scipy convention)
    smoothing : 0.0 = exact interpolation; >0 adds regularisation

    Returns
    -------
    Fitted RBFInterpolator instance
    """
    return RBFInterpolator(params, targets, kernel=kernel, smoothing=smoothing)


def predict(rbf: RBFInterpolator, new_params: np.ndarray) -> np.ndarray:
    """
    Predict POD scores at new parameter points.

    Parameters
    ----------
    rbf        : fitted RBFInterpolator
    new_params : (m, d) new parameter vectors

    Returns
    -------
    (m, k) predicted POD scores
    """
    return rbf(new_params)


def loo_errors(
    params: np.ndarray,
    targets: np.ndarray,
    kernel: str = "thin_plate_spline",
) -> np.ndarray:
    """
    Leave-One-Out cross-validation: absolute prediction error for each sample.
    Used to estimate RBF accuracy without a separate test set.

    Returns
    -------
    (n,) array of ‖predicted − actual‖ errors
    """
    n = len(params)
    errors = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rbf_loo = RBFInterpolator(params[mask], targets[mask], kernel=kernel)
        pred = rbf_loo(params[[i]])
        errors[i] = float(np.linalg.norm(pred - targets[[i]]))
    return errors
