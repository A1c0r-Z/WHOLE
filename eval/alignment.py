"""Rigid/similarity alignment utilities shared across metrics.

Implements the Umeyama similarity transform (scale + rotation + translation)
used by W-MPJPE and WA-MPJPE alignment, and standard Procrustes for PA-MPJPE.

All inputs and outputs are numpy arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Umeyama similarity transform
# ---------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate similarity transform (s, R, t) mapping src → dst.

    Minimises sum ||dst_i - (s * R @ src_i + t)||²  via SVD.

    Reference: Umeyama, PAMI 1991.

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points
    Returns:
        s: scalar scale
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert src.shape == dst.shape and src.ndim == 2

    n, d = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / n

    H = (dst_c.T @ src_c) / n   # (d, d) covariance
    U, S, Vt = np.linalg.svd(H)

    det_sign = np.linalg.det(U @ Vt)
    D = np.diag([1.0] * (d - 1) + [np.sign(det_sign)])

    R = U @ D @ Vt
    s = (S * D.diagonal()).sum() / (var_src + 1e-8)
    t = mu_dst - s * R @ mu_src

    return float(s), R, t


def apply_similarity(
    pts: np.ndarray,   # (..., 3)
    s: float,
    R: np.ndarray,     # (3, 3)
    t: np.ndarray,     # (3,)
) -> np.ndarray:
    return s * (pts @ R.T) + t


# ---------------------------------------------------------------------------
# Procrustes alignment (per-frame, no scale)
# ---------------------------------------------------------------------------

def procrustes(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes: find R, t minimising ||dst - (R@src + t)||.

    Args:
        src: (N, 3)
        dst: (N, 3)
    Returns:
        R: (3, 3)
        t: (3,)
    """
    mu_s = src.mean(0); mu_d = dst.mean(0)
    sc = src - mu_s;    dc = dst - mu_d
    H  = sc.T @ dc
    U, _, Vt = np.linalg.svd(H)
    D  = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R  = Vt.T @ D @ U.T
    t  = mu_d - R @ mu_s
    return R, t


# ---------------------------------------------------------------------------
# Global alignment from selected key-point subset
# ---------------------------------------------------------------------------

def global_align(
    pred_joints: np.ndarray,   # (T, J, 3)
    gt_joints:   np.ndarray,   # (T, J, 3)
    use_frames:  str = 'all',  # 'all' | 'first2'
) -> np.ndarray:
    """Align predicted joint trajectory to GT via Umeyama.

    Args:
        pred_joints: (T, J, 3) predicted
        gt_joints:   (T, J, 3) ground truth
        use_frames:  'all'    → WA-MPJPE (global best alignment)
                     'first2' → W-MPJPE  (align on first 2 frames only)
    Returns:
        (T, J, 3) aligned predicted joints
    """
    if use_frames == 'first2':
        src = pred_joints[:2].reshape(-1, 3)
        dst = gt_joints[:2].reshape(-1, 3)
    else:
        src = pred_joints.reshape(-1, 3)
        dst = gt_joints.reshape(-1, 3)

    s, R, t = umeyama(src, dst)
    aligned = apply_similarity(pred_joints, s, R, t)
    return aligned


def per_frame_procrustes_align(
    pred_joints: np.ndarray,   # (T, J, 3)
    gt_joints:   np.ndarray,   # (T, J, 3)
) -> np.ndarray:
    """Apply per-frame Procrustes alignment (PA-MPJPE)."""
    T = pred_joints.shape[0]
    aligned = np.empty_like(pred_joints)
    for t in range(T):
        R, tr = procrustes(pred_joints[t], gt_joints[t])
        aligned[t] = pred_joints[t] @ R.T + tr
    return aligned
