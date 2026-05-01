"""MANO parameter conversion utilities for HOT3D annotations.

HOT3D stores hand pose as a 15D PCA-compressed representation:
    wrist_xform [6D]: axis-angle global orientation (3D) + world translation (3D)
    thetas      [15D]: PCA coefficients of the 45D MANO joint pose

PCA expansion:  full_pose (45D) = thetas (15D) @ hand_components (15, 45) + hand_mean (45,)

The MANO .npz files are converted from the original .pkl files once via:
    scripts/convert_mano_pkl.py  (requires hawor_h200 env with chumpy)
and stored at MANO_NPZ_DIR as MANO_RIGHT.npz (and MANO_LEFT.npz when available).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch


MANO_NPZ_DIR = Path('/scr/cezhao/workspace/HOI_recon/_DATA/mano')

# Cached MANO data per side: 'left' / 'right'
_MANO_DATA: dict[str, dict] = {}
# Cached smplx MANOLayer per side (used for FK when available)
_MANO_LAYERS: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Loading MANO model data
# ---------------------------------------------------------------------------

def _load_mano_npz(side: str) -> dict:
    """Load the MANO .npz for the given hand side.

    Falls back to a stub if the file is not found, so the rest of the
    pipeline can run (with degraded FK accuracy).
    """
    if side in _MANO_DATA:
        return _MANO_DATA[side]

    npz_path = MANO_NPZ_DIR / f'MANO_{side.upper()}.npz'
    if npz_path.exists():
        data = dict(np.load(str(npz_path), allow_pickle=False))
        _MANO_DATA[side] = data
        return data

    warnings.warn(
        f'MANO_{"RIGHT" if side == "right" else "LEFT"}.npz not found at '
        f'{npz_path}. Run scripts/convert_mano_pkl.py first. '
        f'Using identity PCA stub — FK results will be incorrect.',
        stacklevel=2,
    )
    stub = {
        'hand_components': np.eye(15, 45, dtype=np.float32),
        'hand_mean':       np.zeros(45,  dtype=np.float32),
        'v_template':      np.zeros((778, 3),    dtype=np.float32),
        'J_regressor':     np.zeros((16, 778),   dtype=np.float32),
        'parents':         np.arange(16,          dtype=np.int64),
        'lbs_weights':     np.zeros((778, 16),   dtype=np.float32),
        'faces':           np.zeros((1538, 3),   dtype=np.int64),
        'shapedirs':       np.zeros((778, 3, 10),dtype=np.float32),
        'posedirs':        np.zeros((135, 2334), dtype=np.float32),
    }
    _MANO_DATA[side] = stub
    return stub


# ---------------------------------------------------------------------------
# PCA expansion
# ---------------------------------------------------------------------------

def pca_to_axis_angle(thetas: np.ndarray, side: str = 'right') -> np.ndarray:
    """Expand 15D PCA coefficients to full 45D MANO hand pose (axis-angle).

    full_pose = thetas @ hand_components + hand_mean

    Args:
        thetas: (..., 15) PCA coefficients from HOT3D hands.json
        side:   'left' or 'right'
    Returns:
        (..., 45) axis-angle joint rotations (15 joints × 3)
    """
    data   = _load_mano_npz(side)
    comps  = data['hand_components']   # (15, 45)
    mean   = data['hand_mean']         # (45,)
    shape  = thetas.shape[:-1]
    flat   = thetas.reshape(-1, 15)    # (N, 15)
    pose   = flat @ comps + mean       # (N, 45)
    return pose.reshape(*shape, 45).astype(np.float32)


# ---------------------------------------------------------------------------
# MANO forward kinematics via smplx MANOLayer
# ---------------------------------------------------------------------------

MANO_FINGERTIP_VERT_INDICES = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky


def get_mano_layer(side: str, pkl_path: Optional[str | Path] = None) -> Optional[object]:
    """Return a cached smplx MANO model.

    Uses smplx.create (MANO class) which correctly handles PCA expansion across
    smplx versions.  Returns None if the pkl cannot be loaded (no chumpy).
    """
    if side in _MANO_LAYERS:
        return _MANO_LAYERS[side]

    if pkl_path is None:
        pkl_path = (Path('/scr/cezhao/workspace/HOI_recon/hamer/_DATA/data/mano')
                    / f'MANO_{side.upper()}.pkl')

    try:
        import smplx
        layer = smplx.create(
            str(pkl_path),
            model_type='mano',
            use_pca=True,
            num_pca_comps=15,
            is_rhand=(side == 'right'),
            flat_hand_mean=True,
            batch_size=1,   # smplx.create needs batch_size; overridden at forward time
        )
        layer.eval()
        _MANO_LAYERS[side] = layer
    except Exception as e:
        warnings.warn(f'Could not load smplx MANO for {side}: {e}. FK disabled.')
        _MANO_LAYERS[side] = None

    return _MANO_LAYERS[side]


@torch.no_grad()
def mano_forward(
    thetas:      np.ndarray,   # (T, 15)
    wrist_xform: np.ndarray,   # (T, 6)
    betas:       np.ndarray,   # (10,)
    side:        str,
    batch_size:  int = 64,
) -> dict[str, np.ndarray]:
    """Run MANO FK over T frames using smplx.

    Args:
        thetas:      (T, 15) PCA pose coefficients from HOT3D
        wrist_xform: (T, 6)  [global_orient (3) | transl (3)]
        betas:       (10,)   shape parameters
        side:        'left' or 'right'
        batch_size:  frames per forward pass
    Returns:
        {'joints': (T, 21, 3), 'vertices': (T, 778, 3)}
    """
    layer = get_mano_layer(side)
    if layer is None:
        T = thetas.shape[0]
        return {
            'joints':   np.zeros((T, 21, 3),  dtype=np.float32),
            'vertices': np.zeros((T, 778, 3), dtype=np.float32),
        }

    T = thetas.shape[0]
    go  = torch.tensor(wrist_xform[:, :3], dtype=torch.float32)
    hp  = torch.tensor(thetas,             dtype=torch.float32)
    tr  = torch.tensor(wrist_xform[:, 3:], dtype=torch.float32)
    b   = torch.tensor(betas,             dtype=torch.float32)

    all_joints:   list[np.ndarray] = []
    all_vertices: list[np.ndarray] = []

    fingertip_idx = torch.tensor(MANO_FINGERTIP_VERT_INDICES, dtype=torch.long)

    for s in range(0, T, batch_size):
        e   = min(s + batch_size, T)
        B   = e - s
        out = layer(
            global_orient = go[s:e],
            hand_pose     = hp[s:e],
            transl        = tr[s:e],
            betas         = b.unsqueeze(0).expand(B, -1),
            return_verts  = True,
        )
        verts = out.vertices                         # (B, 778, 3)
        # MANO native: 16 joints. Append 5 fingertip vertices to get 21.
        tips  = verts[:, fingertip_idx, :]           # (B, 5, 3)
        joints_21 = torch.cat([out.joints, tips], dim=1)  # (B, 21, 3)
        all_joints.append(joints_21.numpy())
        all_vertices.append(verts.numpy())

    return {
        'joints':   np.concatenate(all_joints,   axis=0),   # (T, 21, 3)
        'vertices': np.concatenate(all_vertices, axis=0),   # (T, 778, 3)
    }


def compute_joint_velocity(joints: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Finite-difference joint velocities; first frame is zero-padded."""
    vel = np.zeros_like(joints)
    vel[1:] = (joints[1:] - joints[:-1]) / dt
    return vel


# ---------------------------------------------------------------------------
# 31D per-hand feature vector
# ---------------------------------------------------------------------------

def build_hand_feature(
    thetas:      np.ndarray,  # (T, 15)
    wrist_xform: np.ndarray,  # (T, 6)
    betas:       np.ndarray,  # (10,)
) -> np.ndarray:
    """Stack into the 31D per-hand feature used in the diffusion variable.

    Layout: [global_orient (3) | transl (3) | thetas (15) | betas (10)] = 31D
    """
    T = thetas.shape[0]
    betas_rep = np.tile(betas, (T, 1))
    return np.concatenate([wrist_xform, thetas, betas_rep], axis=-1).astype(np.float32)
