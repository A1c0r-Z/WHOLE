"""Data preprocessing for WHOLE training.

Implements:
  1. Gravity-aware coordinate frame alignment
  2. Diffusion variable assembly (73D per frame)
  3. Training-time noise injection to simulate inaccurate hand estimates H̃
  4. BPS encoding of object templates (stub — full BPS in models/bps.py)
"""

from __future__ import annotations

import numpy as np
import torch

from utils.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    se3_to_9d_repr,
    se3_from_9d_repr,
)
from data.mano_converter import build_hand_feature


# ---------------------------------------------------------------------------
# Gravity-aligned coordinate frame
# ---------------------------------------------------------------------------

def estimate_gravity_rotation(T_world_from_cam: np.ndarray) -> np.ndarray:
    """Compute a rotation that aligns the world z-axis with gravity.

    HOT3D's gravity-aware SLAM outputs poses in a frame where gravity is
    approximately -z in world coordinates.  We want a rotation R_align such
    that in the new frame z = up (opposite to gravity).

    In practice we compute R_align from the camera's estimated up direction at
    the first frame of the window: the camera y-axis in world coordinates
    approximates the gravity-up direction.

    Args:
        T_world_from_cam: (4, 4) SE(3) of the reference camera at t=0
    Returns:
        R_align: (3, 3) rotation matrix to apply to all world-frame vectors
    """
    R_wc = T_world_from_cam[:3, :3]  # (3, 3) world <- camera

    # In Aria/Quest3, the camera y-axis points roughly upward in world space.
    # Extract the world-frame camera up vector as proxy for gravity-up.
    cam_up_world = R_wc[:, 1]  # second column = camera y in world frame

    gravity_up = cam_up_world / (np.linalg.norm(cam_up_world) + 1e-8)
    z_target   = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Rotation from gravity_up -> z_target via Rodrigues
    v     = np.cross(gravity_up, z_target)
    s     = np.linalg.norm(v)
    c     = float(np.dot(gravity_up, z_target))

    if s < 1e-6:
        # Already aligned or anti-aligned
        return np.eye(3, dtype=np.float32) if c > 0 else -np.eye(3, dtype=np.float32)

    Kv = np.array([
        [   0, -v[2],  v[1]],
        [ v[2],    0, -v[0]],
        [-v[1],  v[0],    0],
    ], dtype=np.float32)
    R_align = np.eye(3, dtype=np.float32) + Kv + Kv @ Kv * ((1 - c) / (s * s))
    return R_align.astype(np.float32)


def apply_gravity_alignment(
    poses_se3:     np.ndarray,  # (..., 4, 4)
    R_align:       np.ndarray,  # (3, 3)
    origin:        np.ndarray,  # (3,) translation offset (first-frame camera position)
) -> np.ndarray:
    """Transform SE(3) poses into the gravity-aligned local frame.

    The local frame is defined as:
      - origin at the reference camera position at t=0
      - z-axis aligned with gravity-up

    Args:
        poses_se3: (..., 4, 4) poses in world frame
        R_align:   (3, 3) gravity alignment rotation
        origin:    (3,)   world-frame origin to subtract before rotating
    Returns:
        (..., 4, 4) poses in the gravity-aligned local frame
    """
    out = poses_se3.copy()
    # Translate
    out[..., :3, 3] -= origin
    # Rotate translation
    out[..., :3, 3] = (R_align @ out[..., :3, 3, np.newaxis]).squeeze(-1)
    # Rotate rotation
    out[..., :3, :3] = R_align @ out[..., :3, :3]
    return out


def gravity_align_window(window_dict: dict) -> dict:
    """Apply gravity alignment to all SE(3) quantities in a data window.

    Modifies the window in-place and adds 'R_gravity_align' and 'world_origin'
    keys so the alignment can be inverted at inference time.

    Args:
        window_dict: Output of hot3d_loader._clip_to_dict (numpy arrays).
    Returns:
        The same dict, with poses expressed in gravity-aligned frame.
    """
    T_wc0    = window_dict['T_world_from_ref_cam0']  # (4, 4)
    R_align  = estimate_gravity_rotation(T_wc0)
    origin   = T_wc0[:3, 3].copy()                  # camera position at t=0

    # Align object SE(3)
    obj_T = window_dict['obj_T_world']               # (T, 4, 4)
    window_dict['obj_T_world'] = apply_gravity_alignment(obj_T, R_align, origin)

    # Align hand wrist translation (wrist_xform[3:6] is world-frame translation)
    for side in ('left', 'right'):
        wrist = window_dict[f'{side}_wrist'].copy()  # (T, 6)
        t_aligned = (R_align @ wrist[:, 3:].T).T     # (T, 3)
        t_aligned -= origin                          # subtract origin BEFORE rotation
        # Redo: translate first, then rotate
        t_world   = wrist[:, 3:]                     # (T, 3)
        t_local   = (R_align @ (t_world - origin).T).T  # (T, 3)
        wrist[:, 3:] = t_local
        # Also rotate global orientation axis-angle
        R_go = axis_angle_to_matrix(
            torch.tensor(wrist[:, :3], dtype=torch.float32)
        ).numpy()                                    # (T, 3, 3)
        R_go_aligned = R_align[None] @ R_go          # (T, 3, 3)
        go_aligned = matrix_to_axis_angle(
            torch.tensor(R_go_aligned, dtype=torch.float32)
        ).numpy()                                    # (T, 3)
        wrist[:, :3] = go_aligned
        window_dict[f'{side}_wrist'] = wrist

    window_dict['R_gravity_align'] = R_align         # (3, 3)
    window_dict['world_origin']    = origin          # (3,)
    return window_dict


# ---------------------------------------------------------------------------
# Diffusion variable assembly
# ---------------------------------------------------------------------------

def build_diffusion_variable(window_dict: dict) -> np.ndarray:
    """Build the 73D ground-truth diffusion variable x_0 for one window.

    Layout (per frame):
        [obj_9d (9) | contact_left (1) | contact_right (1) |
         left_orient (3) | left_transl (3) | left_thetas (15) | left_betas (10) |
         right_orient (3) | right_transl (3) | right_thetas (15) | right_betas (10)]
        = 9 + 2 + 31 + 31 = 73D

    Contact labels are set to 0 here; the VLM contact module fills them in.

    Args:
        window_dict: Gravity-aligned window dict.
    Returns:
        (T, 73) numpy array — the ground-truth x_0.
    """
    T = window_dict['left_thetas'].shape[0]

    # Object 9D representation
    obj_T    = torch.tensor(window_dict['obj_T_world'], dtype=torch.float32)  # (T,4,4)
    obj_9d   = se3_to_9d_repr(obj_T).numpy()                                  # (T, 9)

    # Contact labels (zeros; filled by VLM pipeline or GT during eval)
    contact  = np.zeros((T, 2), dtype=np.float32)

    # Hand features (31D each)
    left_feat  = build_hand_feature(
        window_dict['left_thetas'],
        window_dict['left_wrist'],
        window_dict['left_betas'],
    )  # (T, 31)
    right_feat = build_hand_feature(
        window_dict['right_thetas'],
        window_dict['right_wrist'],
        window_dict['right_betas'],
    )  # (T, 31)

    x0 = np.concatenate([obj_9d, contact, left_feat, right_feat], axis=-1)  # (T, 73)
    return x0.astype(np.float32)


# ---------------------------------------------------------------------------
# Training noise injection (simulate inaccurate H̃)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def inject_hand_noise(
    window_dict:       dict,
    traj_noise_std:    float = 0.05,
    frame_noise_std:   float = 0.02,
    drop_prob:         float = 0.15,
    rng:               np.random.Generator | None = None,
) -> dict:
    """Perturb ground-truth hand parameters to produce noisy conditioning H̃.

    Mirrors the augmentation described in Sec. 3.1:
      - Trajectory-level noise ς^g: a single offset applied to the whole sequence
      - Per-frame noise ς^t: independent noise per frame
      - Random frame dropping: simulates occlusion/truncation

    Modifies the dict in-place, adding 'left_wrist_noisy' / 'right_wrist_noisy'
    and 'left_thetas_noisy' / 'right_thetas_noisy' alongside the originals.

    Args:
        window_dict:     Data window (numpy arrays).
        traj_noise_std:  Std of trajectory-level Gaussian noise.
        frame_noise_std: Std of per-frame Gaussian noise.
        drop_prob:       Probability of zeroing out a frame.
        rng:             Random number generator (for reproducibility).
    Returns:
        Same dict with added '_noisy' variants.
    """
    if rng is None:
        rng = _RNG

    for side in ('left', 'right'):
        thetas = window_dict[f'{side}_thetas'].copy()   # (T, 15)
        wrist  = window_dict[f'{side}_wrist'].copy()    # (T, 6)
        T      = thetas.shape[0]

        # Trajectory-level offset (same for all frames)
        traj_offset_thetas = rng.normal(0, traj_noise_std, (1, 15)).astype(np.float32)
        traj_offset_wrist  = rng.normal(0, traj_noise_std, (1, 6) ).astype(np.float32)

        # Per-frame noise
        frame_noise_thetas = rng.normal(0, frame_noise_std, thetas.shape).astype(np.float32)
        frame_noise_wrist  = rng.normal(0, frame_noise_std, wrist.shape ).astype(np.float32)

        thetas_noisy = thetas + traj_offset_thetas + frame_noise_thetas
        wrist_noisy  = wrist  + traj_offset_wrist  + frame_noise_wrist

        # Random frame drop (zero out occluded frames)
        drop_mask = rng.random(T) < drop_prob
        thetas_noisy[drop_mask] = 0.0
        wrist_noisy[drop_mask]  = 0.0

        window_dict[f'{side}_thetas_noisy'] = thetas_noisy
        window_dict[f'{side}_wrist_noisy']  = wrist_noisy
        window_dict[f'{side}_drop_mask']    = drop_mask   # (T,) bool

    return window_dict


def build_noisy_hand_feature(window_dict: dict) -> np.ndarray:
    """Assemble the 62D noisy hand conditioning H̃ from perturbed params.

    Args:
        window_dict: Dict with '*_noisy' keys added by inject_hand_noise.
    Returns:
        (T, 62) noisy bimanual hand features [left (31) | right (31)]
    """
    left_feat = build_hand_feature(
        window_dict['left_thetas_noisy'],
        window_dict['left_wrist_noisy'],
        window_dict['left_betas'],
    )  # (T, 31)
    right_feat = build_hand_feature(
        window_dict['right_thetas_noisy'],
        window_dict['right_wrist_noisy'],
        window_dict['right_betas'],
    )  # (T, 31)
    return np.concatenate([left_feat, right_feat], axis=-1)  # (T, 62)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (called by DataLoader workers)
# ---------------------------------------------------------------------------

def preprocess_window(window_dict: dict, augment: bool = True) -> dict[str, np.ndarray]:
    """Run the full preprocessing pipeline on a raw data window.

    Steps:
      1. Gravity alignment
      2. Ground-truth diffusion variable x_0
      3. Noise injection -> noisy conditioning H̃  (training only)

    Args:
        window_dict: Raw window from HOT3DDataset.__getitem__.
        augment:     If True, apply noise injection (training mode).
    Returns:
        Processed dict ready for the model:
            x0:          (T, 73)  ground-truth diffusion variable
            H_tilde:     (T, 62)  noisy hand conditioning (or clean if augment=False)
            frame_valid: (T,)     bool validity mask
            R_align:     (3, 3)   gravity alignment rotation (for visualization)
    """
    window_dict = gravity_align_window(window_dict)

    x0 = build_diffusion_variable(window_dict)  # (T, 73)

    if augment:
        window_dict = inject_hand_noise(window_dict)
        H_tilde = build_noisy_hand_feature(window_dict)  # (T, 62)
    else:
        # At test time H̃ comes from HaWoR; here we use clean params as proxy
        H_tilde = np.concatenate([
            build_hand_feature(
                window_dict['left_thetas'],
                window_dict['left_wrist'],
                window_dict['left_betas'],
            ),
            build_hand_feature(
                window_dict['right_thetas'],
                window_dict['right_wrist'],
                window_dict['right_betas'],
            ),
        ], axis=-1)  # (T, 62)

    return {
        'x0':          x0,
        'H_tilde':     H_tilde,
        'frame_valid': window_dict['frame_valid'],
        'R_align':     window_dict['R_gravity_align'],
        'world_origin': window_dict['world_origin'],
        'clip_id':     window_dict['clip_id'],
    }
