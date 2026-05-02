"""Pure-PyTorch MANO Forward Kinematics via Linear Blend Skinning (LBS).

Loads model data from the chumpy-free .npz files (MANO_RIGHT.npz /
MANO_LEFT.npz) and runs FK without any smplx or chumpy dependency.

This enables full 21-joint FK in the base Python 3.13 environment.

References:
  - Romero et al., "Embodied Hands: Modeling and Capturing Hands and Bodies
    Together" (TOG 2017) — the MANO model
  - Standard LBS formulation: v' = Σ_j w_j * T_j * v_rest
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.mano_converter import MANO_NPZ_DIR, pca_to_axis_angle

# Fingertip vertex indices (thumb, index, middle, ring, pinky)
FINGERTIP_VERT_IDS = [744, 320, 443, 554, 671]

# Cache: side -> dict of tensors on a given device
_MODEL_CACHE: dict[str, dict] = {}


def _load_model(side: str, device: torch.device) -> dict:
    """Load MANO model buffers from npz onto device."""
    key = f'{side}:{device}'
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    npz_path = MANO_NPZ_DIR / f'MANO_{side.upper()}.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f'MANO npz not found: {npz_path}. '
                                f'Run scripts/convert_mano_pkl.py first.')

    raw = dict(np.load(str(npz_path), allow_pickle=False))

    def _t(k):
        return torch.tensor(raw[k], dtype=torch.float32).to(device)

    model = {
        'hand_components': _t('hand_components'),  # (15, 45)
        'hand_mean':       _t('hand_mean'),         # (45,)
        'shapedirs':       _t('shapedirs'),          # (778, 3, 10)
        # posedirs: smplx stores as (135, 2334); reshape to (778, 3, 135)
        'posedirs':        _t('posedirs').T.reshape(778, 3, -1),  # (778, 3, 135)
        'v_template':      _t('v_template'),         # (778, 3)
        'J_regressor':     _t('J_regressor'),        # (16, 778)
        'parents':         torch.tensor(raw['parents'], dtype=torch.long).to(device),  # (16,)
        'lbs_weights':     _t('lbs_weights'),        # (778, 16)
    }
    _MODEL_CACHE[key] = model
    return model


# ---------------------------------------------------------------------------
# Rodrigues rotation
# ---------------------------------------------------------------------------

def batch_rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices.

    Args:
        aa: (..., 3) axis-angle vectors
    Returns:
        (..., 3, 3) rotation matrices
    """
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / angle
    angle = angle.squeeze(-1)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    ax, ay, az = axis.unbind(-1)

    zeros = torch.zeros_like(ax)
    K = torch.stack([zeros, -az, ay, az, zeros, -ax, -ay, ax, zeros],
                    dim=-1).reshape(*aa.shape[:-1], 3, 3)

    I  = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(*aa.shape[:-1], 3, 3)
    c  = cos_a[..., None, None]
    s  = sin_a[..., None, None]
    oo = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    return c * I + s * K + (1 - c) * oo


# ---------------------------------------------------------------------------
# LBS forward kinematics
# ---------------------------------------------------------------------------

def mano_lbs(
    global_orient:  torch.Tensor,   # (B, 3)  axis-angle
    hand_pose:      torch.Tensor,   # (B, 45) axis-angle (15 joints × 3)
    betas:          torch.Tensor,   # (B, 10) shape params
    transl:         torch.Tensor,   # (B, 3)  world translation
    model:          dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MANO Linear Blend Skinning forward pass.

    Returns:
        joints:   (B, 21, 3) — 16 MANO joints + 5 fingertip vertices
        vertices: (B, 778, 3)
    """
    B = global_orient.shape[0]
    v_t = model['v_template']          # (778, 3)
    sd  = model['shapedirs']           # (778, 3, 10)
    pd  = model['posedirs']            # (778, 3, 135)
    J_r = model['J_regressor']         # (16, 778)
    W   = model['lbs_weights']         # (778, 16)
    par = model['parents']             # (16,)

    # ---- Shape blend shapes ----
    v_shaped = v_t + torch.einsum('vcd,bd->bvc', sd, betas)   # (B, 778, 3)

    # ---- Joints in rest pose ----
    J_rest = torch.einsum('jv,bvd->bjd', J_r, v_shaped)        # (B, 16, 3)

    # ---- Pose blend shapes ----
    full_pose = torch.cat([global_orient, hand_pose], dim=-1)   # (B, 48)
    rot_mats  = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(B, 16, 3, 3)

    # Pose feature: (rot - I) flattened, excluding root (15 joints × 9 = 135)
    I    = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device)
    pose_feat = (rot_mats[:, 1:] - I).reshape(B, -1)           # (B, 135)
    v_posed   = v_shaped + torch.einsum('vcd,bd->bvc', pd, pose_feat)  # (B, 778, 3)

    # ---- Forward kinematics: global joint transforms ----
    T_global = _fk(J_rest, rot_mats, par)   # (B, 16, 4, 4)

    # Joints in world space (before translation)
    joints_16 = T_global[:, :, :3, 3]      # (B, 16, 3)

    # ---- Linear blend skinning ----
    T_blend = torch.einsum('bj,bjrc->brc', W, T_global.reshape(B, 16, 4, 4)
                           .reshape(B, 16, 16)).reshape(B, 778, 4, 4)
    # Transform rest-pose vertices
    v_hom = torch.cat([v_posed, torch.ones(B, 778, 1,
                       dtype=v_posed.dtype, device=v_posed.device)], dim=-1)
    vertices = (T_blend @ v_hom.unsqueeze(-1)).squeeze(-1)[..., :3]   # (B, 778, 3)

    # Apply global translation
    vertices = vertices + transl.unsqueeze(1)
    joints_16 = joints_16 + transl.unsqueeze(1)

    # Append fingertip vertices to get 21 joints
    tips = vertices[:, FINGERTIP_VERT_IDS, :]                  # (B, 5, 3)
    joints_21 = torch.cat([joints_16, tips], dim=1)             # (B, 21, 3)

    return joints_21, vertices


def _fk(
    J_rest:   torch.Tensor,   # (B, J, 3)
    rot_mats: torch.Tensor,   # (B, J, 3, 3)
    parents:  torch.Tensor,   # (J,)  -1 = root
) -> torch.Tensor:
    """Compute global joint transforms via kinematic chain.

    Returns:
        (B, J, 4, 4) global SE(3) transforms
    """
    B, J = J_rest.shape[:2]
    device = J_rest.device

    transforms = []
    for j in range(J):
        R_j = rot_mats[:, j]               # (B, 3, 3)
        t_j = J_rest[:, j]                 # (B, 3)

        # Local transform: translate to joint, then rotate
        T_j = torch.eye(4, dtype=R_j.dtype, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        T_j[:, :3, :3] = R_j
        T_j[:, :3,  3] = t_j

        p = parents[j].item()
        if p < 0:
            T_global_j = T_j
        else:
            T_parent = transforms[p]
            # Compose: T_global = T_parent @ (shift_to_joint @ R_j)
            # Standard kinematic: global = parent_global @ (joint_local_relative)
            # where joint_local_relative accounts for the offset from parent
            offset = J_rest[:, j] - J_rest[:, p]   # (B, 3)
            T_local = torch.eye(4, dtype=R_j.dtype, device=device).unsqueeze(0).expand(B, -1, -1).clone()
            T_local[:, :3, :3] = R_j
            T_local[:, :3,  3] = offset
            T_global_j = T_parent @ T_local

        transforms.append(T_global_j)

    T_stack = torch.stack(transforms, dim=1)   # (B, J, 4, 4)

    # Subtract rest-pose joint positions to get displacement transforms
    J_rest_hom = torch.cat([J_rest, torch.ones(B, J, 1, device=device)], dim=-1)
    rest_offsets = (T_stack @ J_rest_hom.unsqueeze(-1)).squeeze(-1)[..., :3]  # (B, J, 3)
    T_stack[:, :, :3, 3] -= rest_offsets.view(B, J, 3)

    return T_stack


# ---------------------------------------------------------------------------
# High-level FK interface (replaces smplx-based mano_forward)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mano_fk_npz(
    thetas:      np.ndarray,   # (T, 15) PCA coefficients
    wrist_xform: np.ndarray,   # (T, 6)  [global_orient (3) | transl (3)]
    betas:       np.ndarray,   # (10,)
    side:        str,
    device:      str | torch.device = 'cpu',
    batch_size:  int = 64,
) -> dict[str, np.ndarray]:
    """MANO FK using only npz data, no smplx/chumpy dependency.

    Args:
        thetas:      (T, 15) PCA joint angle coefficients
        wrist_xform: (T, 6)  [global_orient (3) | world_transl (3)]
        betas:       (10,)
        side:        'left' or 'right'
        device:      target device
    Returns:
        {'joints': (T, 21, 3), 'vertices': (T, 778, 3)}
    """
    if isinstance(device, str):
        device = torch.device(device)

    model = _load_model(side, device)

    # PCA expansion: (T, 15) → (T, 45)
    hand_pose_aa = pca_to_axis_angle(thetas, side=side)

    T = thetas.shape[0]
    go    = torch.tensor(wrist_xform[:, :3], dtype=torch.float32, device=device)
    hp    = torch.tensor(hand_pose_aa,       dtype=torch.float32, device=device)
    tr    = torch.tensor(wrist_xform[:, 3:], dtype=torch.float32, device=device)
    b     = torch.tensor(betas,              dtype=torch.float32, device=device)

    all_joints:   list[np.ndarray] = []
    all_vertices: list[np.ndarray] = []

    for s in range(0, T, batch_size):
        e    = min(s + batch_size, T)
        B    = e - s
        b_b  = b.unsqueeze(0).expand(B, -1)

        j, v = mano_lbs(go[s:e], hp[s:e], b_b, tr[s:e], model)
        all_joints.append(j.cpu().numpy())
        all_vertices.append(v.cpu().numpy())

    return {
        'joints':   np.concatenate(all_joints,   axis=0),   # (T, 21, 3)
        'vertices': np.concatenate(all_vertices, axis=0),   # (T, 778, 3)
    }
