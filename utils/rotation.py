"""Rotation representation utilities.

Supports conversions between: axis-angle, rotation matrix, quaternion (wxyz),
and the 9D continuous representation from Zhou et al. 2019 (CVPR).
All operations are batched and differentiable.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Quaternion (wxyz) <-> matrix
# ---------------------------------------------------------------------------

def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (wxyz) to rotation matrices.

    Args:
        q: (..., 4) tensor, order [w, x, y, z]
    Returns:
        (..., 3, 3) rotation matrices
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (wxyz).

    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        (..., 4) quaternions [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)

    # Case: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2  # 4w
    mask = trace > 0
    q[mask, 0] = 0.25 * s[mask]
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    # Case: R[0,0] is largest diagonal
    cond1 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s1 = torch.sqrt(torch.clamp(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10)) * 2
    q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / s1[cond1]
    q[cond1, 1] = 0.25 * s1[cond1]
    q[cond1, 2] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / s1[cond1]
    q[cond1, 3] = (R[cond1, 0, 2] + R[cond1, 2, 0]) / s1[cond1]

    # Case: R[1,1] is largest diagonal
    cond2 = (~mask) & (~cond1) & (R[:, 1, 1] > R[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10)) * 2
    q[cond2, 0] = (R[cond2, 0, 2] - R[cond2, 2, 0]) / s2[cond2]
    q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / s2[cond2]
    q[cond2, 2] = 0.25 * s2[cond2]
    q[cond2, 3] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / s2[cond2]

    # Case: R[2,2] is largest diagonal
    cond3 = (~mask) & (~cond1) & (~cond2)
    s3 = torch.sqrt(torch.clamp(1 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10)) * 2
    q[cond3, 0] = (R[cond3, 1, 0] - R[cond3, 0, 1]) / s3[cond3]
    q[cond3, 1] = (R[cond3, 0, 2] + R[cond3, 2, 0]) / s3[cond3]
    q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / s3[cond3]
    q[cond3, 3] = 0.25 * s3[cond3]

    return F.normalize(q, dim=-1).reshape(*batch_shape, 4)


# ---------------------------------------------------------------------------
# Axis-angle <-> matrix
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices (Rodrigues formula).

    Args:
        aa: (..., 3) axis-angle vectors (direction = axis, magnitude = angle)
    Returns:
        (..., 3, 3) rotation matrices
    """
    angle = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    angle = angle.squeeze(-1)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    ax, ay, az = axis.unbind(-1)

    K = torch.stack([
        torch.zeros_like(ax), -az,  ay,
        az, torch.zeros_like(ax), -ax,
       -ay,  ax, torch.zeros_like(ax),
    ], dim=-1).reshape(*aa.shape[:-1], 3, 3)

    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(*aa.shape[:-1], 3, 3)
    c = cos_a[..., None, None]
    s = sin_a[..., None, None]
    return c * I + s * K + (1 - c) * (axis.unsqueeze(-1) * axis.unsqueeze(-2))


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors.

    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) axis-angle vectors
    """
    # Use quaternion as intermediate to avoid numerical edge cases
    q = matrix_to_quaternion(R)
    w = q[..., 0:1].clamp(-1 + 1e-7, 1 - 1e-7)
    angle = 2 * torch.acos(w.abs())
    # Keep w positive to pick the shorter arc
    sign = torch.sign(w)
    axis = F.normalize(q[..., 1:] * sign, dim=-1)
    return axis * angle


# ---------------------------------------------------------------------------
# 9D continuous rotation (Zhou et al. 2019)
# ---------------------------------------------------------------------------

def matrix_to_9d(R: torch.Tensor) -> torch.Tensor:
    """Flatten first two columns of a rotation matrix into a 9D vector.

    Args:
        R: (..., 3, 3)
    Returns:
        (..., 9)  [r1 | r2 | r3] column-major flatten
    """
    return R.reshape(*R.shape[:-2], 9)


def matrix_from_9d(r9d: torch.Tensor) -> torch.Tensor:
    """Recover a rotation matrix from a 9D vector via Gram-Schmidt.

    Args:
        r9d: (..., 9)
    Returns:
        (..., 3, 3) orthonormal rotation matrices
    """
    r = r9d.reshape(*r9d.shape[:-1], 3, 3)
    c1 = F.normalize(r[..., :, 0], dim=-1)
    c2 = r[..., :, 1]
    c2 = F.normalize(c2 - (c2 * c1).sum(-1, keepdim=True) * c1, dim=-1)
    c3 = torch.cross(c1, c2, dim=-1)
    return torch.stack([c1, c2, c3], dim=-1)


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def se3_from_json(data: dict) -> torch.Tensor:
    """Parse HOT3D's {quaternion_wxyz, translation_xyz} dict into a 4x4 matrix.

    Args:
        data: dict with keys 'quaternion_wxyz' and 'translation_xyz'
    Returns:
        (4, 4) SE(3) matrix
    """
    q = torch.tensor(data['quaternion_wxyz'], dtype=torch.float32)
    t = torch.tensor(data['translation_xyz'], dtype=torch.float32)
    T = torch.eye(4)
    T[:3, :3] = quaternion_to_matrix(q)
    T[:3, 3] = t
    return T


def se3_to_9d_repr(T: torch.Tensor) -> torch.Tensor:
    """Convert SE(3) matrix to the 9D diffusion representation.

    Representation: [r1 (3D) | r2 (3D) | translation (3D)] = 9D.
    The third column r3 = r1 × r2 is implicit.

    Args:
        T: (..., 4, 4) SE(3) matrices
    Returns:
        (..., 9)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    r1 = R[..., :, 0]
    r2 = R[..., :, 1]
    return torch.cat([r1, r2, t], dim=-1)


def se3_from_9d_repr(r9d: torch.Tensor) -> torch.Tensor:
    """Recover SE(3) matrix from 9D representation.

    Args:
        r9d: (..., 9)
    Returns:
        (..., 4, 4)
    """
    r1 = F.normalize(r9d[..., :3], dim=-1)
    r2_raw = r9d[..., 3:6]
    r2 = F.normalize(r2_raw - (r2_raw * r1).sum(-1, keepdim=True) * r1, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)
    R = torch.stack([r1, r2, r3], dim=-1)
    t = r9d[..., 6:9]
    T = torch.zeros(*r9d.shape[:-1], 4, 4, dtype=r9d.dtype, device=r9d.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# ---------------------------------------------------------------------------
# Numpy convenience wrappers (for preprocessing, not training)
# ---------------------------------------------------------------------------

def quat_wxyz_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Numpy version for use in data preprocessing."""
    return quaternion_to_matrix(torch.tensor(q, dtype=torch.float32)).numpy()
