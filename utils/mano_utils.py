"""Utilities for unpacking the 73D diffusion variable and running MANO FK.

The 73D diffusion variable x layout (per frame):
  [0:9]   object SE(3) 9D representation
  [9:11]  binary contact labels (left, right)
  [11:14] left  global_orient (axis-angle 3D)
  [14:17] left  translation   (world 3D)
  [17:32] left  thetas        (PCA 15D)
  [32:42] left  betas         (shape 10D)
  [42:45] right global_orient
  [45:48] right translation
  [48:63] right thetas
  [63:73] right betas
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from utils.rotation import se3_from_9d_repr, axis_angle_to_matrix


# Slice indices into the 73D diffusion variable
_OBJ_9D   = slice(0,  9)
_CONTACT  = slice(9, 11)
_L_ORIENT = slice(11, 14)
_L_TRANSL = slice(14, 17)
_L_THETA  = slice(17, 32)
_L_BETA   = slice(32, 42)
_R_ORIENT = slice(42, 45)
_R_TRANSL = slice(45, 48)
_R_THETA  = slice(48, 63)
_R_BETA   = slice(63, 73)


def unpack_x0(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack (B, T, 73) diffusion variable into named components.

    Returns:
        obj_9d:      (B, T, 9)   object SE(3) 9D repr
        contact:     (B, T, 2)   binary contact [left, right]
        left_orient: (B, T, 3)   left  global orientation axis-angle
        left_transl: (B, T, 3)   left  wrist world translation
        left_thetas: (B, T, 15)  left  PCA joint angles
        left_betas:  (B, T, 10)  left  shape params
        right_*:     same for right hand
    """
    return {
        'obj_9d':       x[..., _OBJ_9D],
        'contact':      x[..., _CONTACT],
        'left_orient':  x[..., _L_ORIENT],
        'left_transl':  x[..., _L_TRANSL],
        'left_thetas':  x[..., _L_THETA],
        'left_betas':   x[..., _L_BETA],
        'right_orient': x[..., _R_ORIENT],
        'right_transl': x[..., _R_TRANSL],
        'right_thetas': x[..., _R_THETA],
        'right_betas':  x[..., _R_BETA],
    }


def get_obj_transform(x: torch.Tensor) -> torch.Tensor:
    """Decode object SE(3) from 9D repr in x.

    Args:
        x: (B, T, 73)
    Returns:
        (B, T, 4, 4) SE(3) matrices
    """
    return se3_from_9d_repr(x[..., _OBJ_9D])


def apply_obj_transform(
    T_obj: torch.Tensor,       # (B, T, 4, 4)
    template_verts: torch.Tensor,  # (V, 3) canonical vertices
) -> torch.Tensor:
    """Apply per-frame object SE(3) to canonical mesh vertices.

    Args:
        T_obj:          (B, T, 4, 4) per-frame object transforms
        template_verts: (V, 3) canonical object vertices
    Returns:
        (B, T, V, 3) world-frame object vertices
    """
    R = T_obj[..., :3, :3]   # (B, T, 3, 3)
    t = T_obj[..., :3, 3]    # (B, T, 3)
    # Broadcast: (B,T,3,3) × (B,T,V,3,1) → (B,T,V,3)
    verts = (R.unsqueeze(2) @ template_verts.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(2)
    return verts


# ---------------------------------------------------------------------------
# GPU-batched MANO FK
# ---------------------------------------------------------------------------

_MANO_LAYERS: dict[str, object] = {}


def get_mano_layer_gpu(side: str, device: str | torch.device) -> Optional[object]:
    """Return a cached smplx MANO layer on the target device."""
    key = f'{side}:{device}'
    if key not in _MANO_LAYERS:
        from data.mano_converter import get_mano_layer
        layer = get_mano_layer(side)
        if layer is not None:
            _MANO_LAYERS[key] = layer.to(device)
        else:
            _MANO_LAYERS[key] = None
    return _MANO_LAYERS[key]


def fk_from_x0(
    x:       torch.Tensor,    # (B, T, 73)
    side:    str,              # 'left' or 'right'
    device:  Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Run MANO FK on predicted hand parameters; return (B, T, 21, 3) joints.

    Returns None if MANO layer is unavailable (no chumpy in env).
    Runs FK in chunks of 64 frames to stay within GPU memory.
    """
    if device is None:
        device = x.device
    layer = get_mano_layer_gpu(side, device)
    if layer is None:
        return None

    p  = unpack_x0(x)
    B, T = x.shape[:2]
    prefix = 'left' if side == 'left' else 'right'

    go    = p[f'{prefix}_orient'].reshape(B * T, 3)
    hp    = p[f'{prefix}_thetas'].reshape(B * T, 15)
    tr    = p[f'{prefix}_transl'].reshape(B * T, 3)
    # Use the mean beta across frames for each sample in the batch
    betas = p[f'{prefix}_betas'].mean(dim=1)      # (B, 10)
    betas = betas.unsqueeze(1).expand(-1, T, -1).reshape(B * T, 10)

    CHUNK = 64
    all_joints = []
    fingertip_idx = torch.tensor([744, 320, 443, 554, 671],
                                  device=device, dtype=torch.long)

    with torch.no_grad():
        for s in range(0, B * T, CHUNK):
            e   = min(s + CHUNK, B * T)
            out = layer(global_orient=go[s:e], hand_pose=hp[s:e],
                        transl=tr[s:e], betas=betas[s:e], return_verts=True)
            tips = out.vertices[:, fingertip_idx, :]          # (chunk, 5, 3)
            j21  = torch.cat([out.joints, tips], dim=1)       # (chunk, 21, 3)
            all_joints.append(j21)

    joints = torch.cat(all_joints, dim=0).reshape(B, T, 21, 3)
    return joints
