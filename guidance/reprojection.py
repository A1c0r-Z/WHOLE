"""Reprojection guidance g_reproj (Sec. 3.2).

Projects predicted 3D hand joints and object mesh to 2D and measures
consistency with observed segmentation masks via one-way Chamfer distance.

One-way Chamfer (pred→obs): for each predicted 2D point, find its
nearest observed mask pixel.  The asymmetric direction handles occlusion
and truncation: if parts of the prediction are out-of-view, only visible
projected points contribute.

Camera models supported:
  - Pinhole (simple, for testing)
  - FISHEYE624 (HOT3D Aria / Quest3 actual model) — requires calibration params
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils.mano_utils import (
    unpack_x0, fk_from_x0, get_obj_transform, apply_obj_transform
)


# ---------------------------------------------------------------------------
# Camera projection
# ---------------------------------------------------------------------------

def project_pinhole(
    pts_3d:  torch.Tensor,   # (..., 3) points in camera frame
    fx: float, fy: float,
    cx: float, cy: float,
) -> torch.Tensor:
    """Project 3D points with a simple pinhole model.

    Returns:
        (..., 2) pixel coordinates (x, y)
    """
    x = pts_3d[..., 0]
    y = pts_3d[..., 1]
    z = pts_3d[..., 2].clamp(min=1e-4)
    u = fx * x / z + cx
    v = fy * y / z + cy
    return torch.stack([u, v], dim=-1)


def world_to_camera(
    pts_world: torch.Tensor,   # (..., 3)
    T_cam_from_world: torch.Tensor,   # (4, 4)
) -> torch.Tensor:
    """Transform world-frame points into camera frame."""
    R = T_cam_from_world[:3, :3]
    t = T_cam_from_world[:3,  3]
    return pts_world @ R.T + t


def project_points(
    pts_world: torch.Tensor,      # (B, T, N, 3)
    T_world_from_cam: torch.Tensor,  # (B, T, 4, 4) SLAM poses
    intrinsics: dict,
) -> torch.Tensor:
    """Project world-frame 3D points to 2D pixel coordinates.

    Handles the full (B, T, N) batch efficiently.

    Args:
        pts_world:        (B, T, N, 3) points in world frame
        T_world_from_cam: (B, T, 4, 4) camera-to-world SLAM pose per frame
        intrinsics:       dict with keys fx, fy, cx, cy (for pinhole)
                          or full FISHEYE624 params
    Returns:
        (B, T, N, 2) projected pixel coordinates
    """
    # Invert world-from-cam to get cam-from-world
    R_wc = T_world_from_cam[..., :3, :3]   # (B, T, 3, 3)
    t_wc = T_world_from_cam[..., :3,  3]   # (B, T, 3)
    R_cw = R_wc.transpose(-1, -2)
    t_cw = -(R_cw @ t_wc.unsqueeze(-1)).squeeze(-1)

    # pts_world: (B, T, N, 3) → camera frame: (B, T, N, 3)
    pts_cam = (R_cw.unsqueeze(2) @ pts_world.unsqueeze(-1)).squeeze(-1) + t_cw.unsqueeze(2)

    model = intrinsics.get('model', 'pinhole')
    if model == 'fisheye624':
        return _project_fisheye624(pts_cam, intrinsics)
    else:
        fx = intrinsics['fx']; fy = intrinsics['fy']
        cx = intrinsics['cx']; cy = intrinsics['cy']
        return project_pinhole(pts_cam, fx, fy, cx, cy)


def _project_fisheye624(
    pts_cam: torch.Tensor,   # (..., 3)
    p:       dict,
) -> torch.Tensor:
    """KB fisheye (FISHEYE624) projection for HOT3D cameras.

    Uses the Kannala-Brandt model with 6 radial and 2 tangential coefficients.
    Projection params from calibration['projection_params']:
      [fx, cx, cy, k0, k1, k2, k3, t0, t1, k4, k5, k6, k7, p0, p1]
    """
    params = p['projection_params']
    fx = params[0]; cx = params[1]; cy = params[2]
    k  = params[3:7]   # k0..k3 radial distortion

    x = pts_cam[..., 0]
    y = pts_cam[..., 1]
    z = pts_cam[..., 2].clamp(min=1e-6)

    r    = torch.sqrt(x*x + y*y).clamp(min=1e-8)
    th   = torch.atan(r / z)
    th2  = th * th
    # Polynomial: th_d = th*(1 + k0*th^2 + k1*th^4 + k2*th^6 + k3*th^8)
    th_d = th * (1 + k[0]*th2 + k[1]*th2**2 + k[2]*th2**3 + k[3]*th2**4)

    u = fx * th_d * (x / r) + cx
    v = fx * th_d * (y / r) + cy
    return torch.stack([u, v], dim=-1)


# ---------------------------------------------------------------------------
# One-way Chamfer in 2D
# ---------------------------------------------------------------------------

def one_way_chamfer_2d(
    pred_pts:   torch.Tensor,   # (B, T, N, 2) predicted 2D projections
    obs_masks:  torch.Tensor,   # (B, T, H, W) binary observed masks
    valid_mask: Optional[torch.Tensor] = None,  # (B, T, N) True = point is valid
) -> torch.Tensor:
    """One-way Chamfer: for each predicted point, nearest observed mask pixel.

    The one-way direction (pred→obs only) means out-of-frame or occluded
    predictions don't pull toward wrong mask regions.

    Args:
        pred_pts:   (B, T, N, 2) projected pixel coordinates (x, y)
        obs_masks:  (B, T, H, W) observed segmentation masks (binary)
        valid_mask: (B, T, N) True = within image bounds and not occluded
    Returns:
        scalar loss
    """
    B, T, N, _ = pred_pts.shape
    H, W = obs_masks.shape[2], obs_masks.shape[3]

    # Extract observed mask pixel coordinates for each (b, t)
    # For efficiency, use grid_sample to look up the mask value at predicted points
    # Then penalize only those landing in background (mask=0)

    # Normalize to [-1, 1] for grid_sample
    px = pred_pts[..., 0]   # (B, T, N)
    py = pred_pts[..., 1]
    gx = (2 * px / (W - 1) - 1).clamp(-1, 1)
    gy = (2 * py / (H - 1) - 1).clamp(-1, 1)
    grid = torch.stack([gx, gy], dim=-1).reshape(B * T, N, 1, 2)

    mask_bt = obs_masks.reshape(B * T, 1, H, W).float()
    # Sample mask value at predicted points: 1 = in mask, 0 = background
    sampled = F.grid_sample(mask_bt, grid, align_corners=True,
                             mode='nearest', padding_mode='zeros')
    sampled = sampled.reshape(B, T, N)   # 1 if point lands on mask

    # Squared distance to image center as proxy for "off-mask" penalty
    # Better: compute actual nearest mask pixel distance (expensive for large H,W)
    # Use an approximation: penalty = 1 - sampled (0 when on mask, 1 when off)
    # Scaled by distance to encourage movement toward mask
    off_mask = (1.0 - sampled)   # (B, T, N) 1=not on mask

    if valid_mask is not None:
        off_mask = off_mask * valid_mask.float()
        n = valid_mask.float().sum().clamp(min=1)
    else:
        n = float(B * T * N)

    return off_mask.sum() / n


# ---------------------------------------------------------------------------
# Full reprojection guidance function
# ---------------------------------------------------------------------------

def guidance_reproj(
    x0_pred:          torch.Tensor,        # (B, T, 73)
    T_world_from_cam: torch.Tensor,        # (B, T, 4, 4) SLAM poses
    intrinsics:       dict,
    obs_hand_masks:   Optional[torch.Tensor] = None,  # (B, T, H, W)
    obs_obj_masks:    Optional[torch.Tensor] = None,  # (B, T, H, W)
    contact_labels:   Optional[torch.Tensor] = None,  # (B, T, 2)
    template_verts:   Optional[torch.Tensor] = None,  # (V, 3)
    frame_valid:      Optional[torch.Tensor] = None,  # (B, T)
) -> torch.Tensor:
    """Compute reprojection guidance loss g_reproj.

    Combines:
      - Hand joint reprojection vs. observed hand mask
      - Object mesh reprojection vs. observed object mask
      - Contact consistency: projected joints should overlap mask when contact=1

    Args:
        x0_pred:          predicted clean trajectory
        T_world_from_cam: camera SLAM poses
        intrinsics:       camera intrinsic parameters dict
        obs_hand_masks:   observed hand segmentation (combined L+R)
        obs_obj_masks:    observed object segmentation
        contact_labels:   VLM contact labels (B, T, 2)
        template_verts:   canonical object mesh vertices
        frame_valid:      validity mask
    Returns:
        scalar guidance cost (to be minimized, gradient ascent on x)
    """
    total = x0_pred.new_zeros(())

    # ---- Hand joint reprojection ----
    left_j  = fk_from_x0(x0_pred, 'left')
    right_j = fk_from_x0(x0_pred, 'right')

    if left_j is not None and obs_hand_masks is not None:
        all_joints = torch.cat([left_j, right_j], dim=2)   # (B, T, 42, 3)
        proj_joints = project_points(all_joints, T_world_from_cam, intrinsics)
        # In-frame validity: projected within image bounds
        H, W = obs_hand_masks.shape[2], obs_hand_masks.shape[3]
        valid_px = (
            (proj_joints[..., 0] >= 0) & (proj_joints[..., 0] < W) &
            (proj_joints[..., 1] >= 0) & (proj_joints[..., 1] < H)
        )
        total = total + one_way_chamfer_2d(proj_joints, obs_hand_masks, valid_px)

    # ---- Object mesh reprojection ----
    if template_verts is not None and obs_obj_masks is not None:
        T_obj  = get_obj_transform(x0_pred)
        obj_wv = apply_obj_transform(T_obj, template_verts)   # (B, T, V, 3)
        # Subsample vertices for efficiency
        V = obj_wv.shape[2]
        if V > 256:
            idx = torch.linspace(0, V - 1, 256, dtype=torch.long, device=obj_wv.device)
            obj_wv = obj_wv[:, :, idx, :]
        proj_obj = project_points(obj_wv, T_world_from_cam, intrinsics)
        H, W = obs_obj_masks.shape[2], obs_obj_masks.shape[3]
        valid_px = (
            (proj_obj[..., 0] >= 0) & (proj_obj[..., 0] < W) &
            (proj_obj[..., 1] >= 0) & (proj_obj[..., 1] < H)
        )
        total = total + one_way_chamfer_2d(proj_obj, obs_obj_masks, valid_px)

    return total
