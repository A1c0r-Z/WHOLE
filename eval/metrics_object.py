"""Object motion evaluation metrics (Table 2).

Metrics:
  ADD    mean distance between predicted and GT model points after transform
  ADD-S  symmetric ADD (nearest-point, for symmetric objects)
  AUC @ 0.3  area under ADD/ADD-S recall curve from 0→0.3m (paper uses 0.3,
             not the standard BOP 0.1 which is too strict for egocentric HOI)

All SE(3) transforms are (4, 4) numpy arrays; vertices are (V, 3) in metres.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# ADD / ADD-S per frame
# ---------------------------------------------------------------------------

def _transform_verts(T: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Apply SE(3) (4,4) to vertices (V,3) → (V,3)."""
    return verts @ T[:3, :3].T + T[:3, 3]


def add_per_frame(
    T_pred: np.ndarray,   # (4, 4)
    T_gt:   np.ndarray,   # (4, 4)
    verts:  np.ndarray,   # (V, 3)
) -> float:
    """ADD for a single frame: mean vertex displacement."""
    v_pred = _transform_verts(T_pred, verts)
    v_gt   = _transform_verts(T_gt,   verts)
    return float(np.linalg.norm(v_pred - v_gt, axis=-1).mean())


def add_s_per_frame(
    T_pred: np.ndarray,
    T_gt:   np.ndarray,
    verts:  np.ndarray,
) -> float:
    """ADD-S for a single frame: mean nearest-point vertex distance."""
    v_pred = _transform_verts(T_pred, verts)
    v_gt   = _transform_verts(T_gt,   verts)
    # Pairwise distances (V_pred, V_gt)
    diff = v_pred[:, None] - v_gt[None, :]   # (V, V, 3)
    d    = np.linalg.norm(diff, axis=-1)      # (V, V)
    return float(d.min(axis=1).mean())


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def compute_auc(
    distances:  np.ndarray,   # (T,) per-frame distances in metres
    threshold:  float = 0.3,
    n_steps:    int   = 100,
) -> float:
    """AUC of ADD/ADD-S recall curve from 0 to threshold.

    Following the paper: threshold=0.3 (instead of BOP standard 0.1).
    AUC is normalised to [0, 1] by dividing the integral by the threshold.

    Args:
        distances: per-frame ADD or ADD-S values
        threshold: upper integration limit (default 0.3 m)
        n_steps:   number of equally-spaced threshold steps
    Returns:
        AUC in [0, 1]  (higher = better)
    """
    ts      = np.linspace(0, threshold, n_steps)
    recall  = np.array([(distances <= t).mean() for t in ts])
    _trapz  = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc     = float(_trapz(recall, ts) / threshold)
    return auc


# ---------------------------------------------------------------------------
# Sequence-level ADD / AUC
# ---------------------------------------------------------------------------

def compute_object_metrics(
    T_pred_seq:  np.ndarray,   # (T, 4, 4) predicted SE(3) sequence
    T_gt_seq:    np.ndarray,   # (T, 4, 4) GT SE(3) sequence
    verts:       np.ndarray,   # (V, 3)    canonical mesh vertices (metres)
    symmetric:   bool  = False,
    auc_threshold: float = 0.3,
) -> dict[str, float]:
    """Compute ADD (or ADD-S) and AUC for a full sequence.

    Args:
        T_pred_seq:  predicted object poses
        T_gt_seq:    GT object poses
        verts:       canonical mesh vertices
        symmetric:   use ADD-S (nearest-point) instead of ADD
        auc_threshold: AUC integration upper limit (default 0.3 m)
    Returns:
        dict with 'ADD', 'ADD-S', 'AUC_ADD', 'AUC_ADD-S'
    """
    T = T_pred_seq.shape[0]
    add_vals  = np.empty(T)
    adds_vals = np.empty(T)

    for t in range(T):
        add_vals[t]  = add_per_frame(T_pred_seq[t], T_gt_seq[t], verts)
        adds_vals[t] = add_s_per_frame(T_pred_seq[t], T_gt_seq[t], verts)

    return {
        'ADD':      float(add_vals.mean()),
        'ADD-S':    float(adds_vals.mean()),
        'AUC_ADD':  compute_auc(add_vals,  auc_threshold),
        'AUC_ADD-S': compute_auc(adds_vals, auc_threshold),
    }
