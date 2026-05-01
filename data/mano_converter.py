"""MANO parameter conversion utilities for HOT3D annotations.

HOT3D stores hand pose as a 15D PCA-compressed representation:
    wrist_xform [6D]: axis-angle global orientation (3D) + world translation (3D)
    thetas      [15D]: PCA coefficients of the 45D MANO joint pose

This module handles:
  1. Loading the HOT3D PCA basis from the toolkit
  2. Converting PCA thetas -> full 45D axis-angle joint poses
  3. Building MANO input dicts compatible with the smplx library
  4. Running MANO forward kinematics to obtain joint positions J
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


# HOT3D PCA basis for MANO hand pose.
# The basis is a (45, 15) matrix: full_pose = basis @ thetas
# Loaded lazily from the HOT3D toolkit package on first use.
_PCA_BASIS: Optional[np.ndarray] = None
_PCA_MEAN:  Optional[np.ndarray] = None


def _load_pca_basis() -> tuple[np.ndarray, np.ndarray]:
    """Load HOT3D's MANO PCA basis and mean pose.

    HOT3D reduces 45D MANO hand pose to 15D via PCA.  The basis lives inside
    the `hot3d` Python package shipped with the dataset toolkit.  We try two
    strategies:
      1. Import directly from the installed hot3d package.
      2. Fall back to a zero mean + identity-like stub so the rest of the
         pipeline can run even without the official toolkit installed.

    Returns:
        basis: (45, 15) numpy array
        mean:  (45,)   numpy array
    """
    global _PCA_BASIS, _PCA_MEAN

    if _PCA_BASIS is not None:
        return _PCA_BASIS, _PCA_MEAN

    try:
        # HOT3D toolkit stores PCA components inside its hand model module
        from hot3d.data_loaders.HandModel import MANOHandModel  # type: ignore
        model = MANOHandModel()
        _PCA_BASIS = model.pose_pca_basis.numpy()   # (45, 15)
        _PCA_MEAN  = model.pose_pca_mean.numpy()    # (45,)
    except Exception:
        # Fallback: treat thetas as the first 15 dims of the pose directly.
        # This is only a placeholder — replace with actual basis when available.
        import warnings
        warnings.warn(
            "HOT3D toolkit not found; using identity PCA basis stub. "
            "Install the hot3d package for correct MANO reconstruction.",
            stacklevel=2,
        )
        _PCA_BASIS = np.zeros((45, 15), dtype=np.float32)
        _PCA_BASIS[:15, :15] = np.eye(15, dtype=np.float32)
        _PCA_MEAN  = np.zeros(45, dtype=np.float32)

    return _PCA_BASIS, _PCA_MEAN


def pca_to_axis_angle(thetas: np.ndarray) -> np.ndarray:
    """Expand 15D PCA coefficients to full 45D MANO hand pose (axis-angle).

    Args:
        thetas: (..., 15) PCA coefficients
    Returns:
        (..., 45) axis-angle joint rotations (15 joints × 3)
    """
    basis, mean = _load_pca_basis()              # (45, 15), (45,)
    shape = thetas.shape[:-1]
    flat  = thetas.reshape(-1, 15)               # (N, 15)
    pose  = (flat @ basis.T) + mean              # (N, 45)
    return pose.reshape(*shape, 45)


def build_mano_input(
    thetas:      np.ndarray,   # (T, 15)
    wrist_xform: np.ndarray,   # (T, 6)
    betas:       np.ndarray,   # (10,)
) -> dict[str, np.ndarray]:
    """Assemble a dict of MANO parameters from HOT3D's stored representation.

    HOT3D wrist_xform layout:
        [0:3]  axis-angle global orientation (world → wrist)
        [3:6]  wrist translation in world frame

    Args:
        thetas:      (T, 15) PCA articulation coefficients
        wrist_xform: (T, 6)  global orient + world translation
        betas:       (10,)   shape parameters (constant per person)
    Returns:
        dict with keys: global_orient (T,3), hand_pose (T,45),
                        transl (T,3), betas (10,)
    """
    T = thetas.shape[0]
    hand_pose = pca_to_axis_angle(thetas)        # (T, 45)

    return {
        'global_orient': wrist_xform[:, :3],     # (T, 3)
        'hand_pose':     hand_pose,              # (T, 45)
        'transl':        wrist_xform[:, 3:],     # (T, 3)
        'betas':         betas,                  # (10,)
    }


# ---------------------------------------------------------------------------
# MANO forward kinematics via smplx
# ---------------------------------------------------------------------------

_MANO_LAYERS: dict[str, object] = {}  # side -> smplx.MANO, cached


def get_mano_layer(side: str, model_path: str | Path) -> object:
    """Return a cached smplx MANO layer for the given hand side.

    Args:
        side:       'left' or 'right'
        model_path: Directory containing MANO_LEFT.pkl / MANO_RIGHT.pkl
    Returns:
        smplx.MANO model instance (eval mode, on CPU)
    """
    key = f'{side}:{model_path}'
    if key not in _MANO_LAYERS:
        import smplx
        model_path = Path(model_path)
        layer = smplx.create(
            str(model_path),
            model_type='mano',
            is_rhand=(side == 'right'),
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            batch_size=1,
        )
        layer.eval()
        _MANO_LAYERS[key] = layer
    return _MANO_LAYERS[key]


@torch.no_grad()
def mano_forward(
    mano_params:  dict[str, np.ndarray],
    side:         str,
    model_path:   str | Path,
    batch_size:   int = 64,
) -> dict[str, np.ndarray]:
    """Run MANO forward kinematics over T frames.

    Batches the T frames to avoid GPU OOM on long sequences.

    Args:
        mano_params:  Output of build_mano_input (T frames).
        side:         'left' or 'right'
        model_path:   Path to MANO model directory.
        batch_size:   Frames processed per forward pass.
    Returns:
        dict with:
            joints:   (T, 21, 3)  joint positions in world frame
            vertices: (T, 778, 3) mesh vertices in world frame
    """
    layer = get_mano_layer(side, model_path)

    T            = mano_params['global_orient'].shape[0]
    global_orient = torch.tensor(mano_params['global_orient'], dtype=torch.float32)  # (T, 3)
    hand_pose     = torch.tensor(mano_params['hand_pose'],     dtype=torch.float32)  # (T, 45)
    transl        = torch.tensor(mano_params['transl'],        dtype=torch.float32)  # (T, 3)
    betas         = torch.tensor(mano_params['betas'],         dtype=torch.float32)  # (10,)
    betas_batch   = betas.unsqueeze(0).expand(batch_size, -1)

    all_joints:   list[np.ndarray] = []
    all_vertices: list[np.ndarray] = []

    for start in range(0, T, batch_size):
        end  = min(start + batch_size, T)
        B    = end - start
        b_betas = betas.unsqueeze(0).expand(B, -1)

        output = layer(
            global_orient = global_orient[start:end],
            hand_pose     = hand_pose[start:end],
            transl        = transl[start:end],
            betas         = b_betas,
            return_verts  = True,
        )
        all_joints.append(output.joints.numpy())    # (B, 21, 3)
        all_vertices.append(output.vertices.numpy())# (B, 778, 3)

    return {
        'joints':   np.concatenate(all_joints,   axis=0),   # (T, 21, 3)
        'vertices': np.concatenate(all_vertices, axis=0),   # (T, 778, 3)
    }


# ---------------------------------------------------------------------------
# Joint velocity
# ---------------------------------------------------------------------------

def compute_joint_velocity(joints: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Finite-difference joint velocities.

    Args:
        joints: (T, J, 3) joint positions
        dt:     Time step in seconds (HOT3D clips are ~3 fps after subsampling)
    Returns:
        (T, J, 3)  velocities; first frame is zero-padded
    """
    vel = np.zeros_like(joints)
    vel[1:] = (joints[1:] - joints[:-1]) / dt
    return vel


# ---------------------------------------------------------------------------
# Convenience: full hand feature vector H (31D per hand)
# ---------------------------------------------------------------------------

def build_hand_feature(
    thetas:      np.ndarray,  # (T, 15)
    wrist_xform: np.ndarray,  # (T, 6)
    betas:       np.ndarray,  # (10,)
) -> np.ndarray:
    """Stack HOT3D hand annotations into the 31D per-hand feature used by WHOLE.

    Layout: [global_orient (3) | transl (3) | thetas (15) | betas (10)] = 31D

    Args:
        thetas:      (T, 15)
        wrist_xform: (T, 6)   [global_orient (3) | transl (3)]
        betas:       (10,)
    Returns:
        (T, 31) feature vectors
    """
    T = thetas.shape[0]
    betas_rep = np.tile(betas, (T, 1))           # (T, 10)
    return np.concatenate([wrist_xform, thetas, betas_rep], axis=-1)  # (T, 31)
