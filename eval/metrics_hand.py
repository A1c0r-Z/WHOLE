"""Hand motion evaluation metrics (Table 1).

Metrics (units from supplementary):
  W-MPJPE   cm   align on first-2-frames  → MPJPE over all frames
  WA-MPJPE  cm   align on all frames      → MPJPE over all frames
  PA-MPJPE  mm   per-frame Procrustes     → MPJPE per frame
  ACC-NORM  –    ||acc_pred - acc_gt||₂   (acceleration consistency)

All inputs are numpy arrays of joint positions in metres.
"""

from __future__ import annotations

import numpy as np

from eval.alignment import global_align, per_frame_procrustes_align


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error (L2 distance, averaged over joints & frames).

    Args:
        pred: (T, J, 3)
        gt:   (T, J, 3)
    Returns:
        scalar in same units as input
    """
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def w_mpjpe(
    pred_joints: np.ndarray,   # (T, J, 3)  metres
    gt_joints:   np.ndarray,
) -> float:
    """W-MPJPE: align on first 2 frames, evaluate all. Returns cm."""
    aligned = global_align(pred_joints, gt_joints, use_frames='first2')
    return mpjpe(aligned, gt_joints) * 100.0   # m → cm


def wa_mpjpe(
    pred_joints: np.ndarray,
    gt_joints:   np.ndarray,
) -> float:
    """WA-MPJPE: align on all frames globally, evaluate all. Returns cm."""
    aligned = global_align(pred_joints, gt_joints, use_frames='all')
    return mpjpe(aligned, gt_joints) * 100.0


def pa_mpjpe(
    pred_joints: np.ndarray,
    gt_joints:   np.ndarray,
) -> float:
    """PA-MPJPE: per-frame Procrustes, evaluate all. Returns mm."""
    aligned = per_frame_procrustes_align(pred_joints, gt_joints)
    return mpjpe(aligned, gt_joints) * 1000.0  # m → mm


def acc_norm(
    pred_joints: np.ndarray,   # (T, J, 3)
    gt_joints:   np.ndarray,
) -> float:
    """ACC-NORM: mean ||acceleration_pred - acceleration_gt||₂.

    Acceleration is the second-order finite difference of joint positions.
    """
    def _accel(j):
        return j[2:] - 2 * j[1:-1] + j[:-2]   # (T-2, J, 3)

    acc_p = _accel(pred_joints)
    acc_g = _accel(gt_joints)
    return float(np.linalg.norm(acc_p - acc_g, axis=-1).mean())


def compute_hand_metrics(
    pred_joints: np.ndarray,   # (T, J, 3) metres — from MANO FK on pred x0
    gt_joints:   np.ndarray,   # (T, J, 3) metres — from MANO FK on GT x0
) -> dict[str, float]:
    """Compute all four hand motion metrics.

    Returns:
        {'W-MPJPE': cm, 'WA-MPJPE': cm, 'PA-MPJPE': mm, 'ACC-NORM': unitless}
    """
    return {
        'W-MPJPE':  w_mpjpe(pred_joints,  gt_joints),
        'WA-MPJPE': wa_mpjpe(pred_joints, gt_joints),
        'PA-MPJPE': pa_mpjpe(pred_joints, gt_joints),
        'ACC-NORM': acc_norm(pred_joints, gt_joints),
    }
