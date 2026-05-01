"""Unit tests for utils/rotation.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from utils.rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_9d,
    matrix_from_9d,
    se3_from_json,
    se3_to_9d_repr,
    se3_from_9d_repr,
)


def random_rotation(n: int = 1) -> torch.Tensor:
    """Sample random rotation matrices via QR decomposition."""
    A = torch.randn(n, 3, 3)
    Q, R = torch.linalg.qr(A)
    # Ensure det = +1
    sign = torch.det(Q).sign().unsqueeze(-1).unsqueeze(-1)
    return Q * sign


class TestQuaternion:
    def test_roundtrip(self):
        R = random_rotation(64)
        q = matrix_to_quaternion(R)
        R2 = quaternion_to_matrix(q)
        assert torch.allclose(R, R2, atol=1e-5), f"max err: {(R-R2).abs().max():.2e}"

    def test_unit_norm(self):
        R = random_rotation(32)
        q = matrix_to_quaternion(R)
        norms = q.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_identity(self):
        I = torch.eye(3).unsqueeze(0)
        q = matrix_to_quaternion(I)
        # w should be ~1, xyz ~0
        assert q[0, 0].abs() > 0.99

    def test_batch_shape(self):
        R = random_rotation(4 * 5).reshape(4, 5, 3, 3)
        q = matrix_to_quaternion(R)
        assert q.shape == (4, 5, 4)
        R2 = quaternion_to_matrix(q)
        assert R2.shape == (4, 5, 3, 3)

    def test_from_hot3d_json(self):
        """Quaternion from a real HOT3D cameras.json entry."""
        d = {
            'quaternion_wxyz': [0.4814041382910753, 0.6266316067707299,
                                 0.3803945499551591, -0.48050272777224967],
            'translation_xyz': [0.173, 0.159, -0.146],
        }
        T = se3_from_json(d)
        R = T[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), "Not orthogonal"
        assert abs(float(torch.det(R)) - 1.0) < 1e-5, "Det != 1"


class TestAxisAngle:
    def test_roundtrip(self):
        aa = torch.randn(32, 3) * 0.5   # small angles for numerical stability
        R  = axis_angle_to_matrix(aa)
        aa2 = matrix_to_axis_angle(R)
        R2  = axis_angle_to_matrix(aa2)
        assert torch.allclose(R, R2, atol=1e-4), f"max err: {(R-R2).abs().max():.2e}"

    def test_zero_angle(self):
        aa = torch.zeros(4, 3)
        R  = axis_angle_to_matrix(aa)
        assert torch.allclose(R, torch.eye(3).expand(4, 3, 3), atol=1e-6)

    def test_orthogonality(self):
        aa = torch.randn(64, 3)
        R  = axis_angle_to_matrix(aa)
        I  = R @ R.transpose(-1, -2)
        assert torch.allclose(I, torch.eye(3).expand_as(I), atol=1e-5)


class Test9DRotation:
    def test_roundtrip(self):
        R   = random_rotation(64)
        r9d = matrix_to_9d(R)
        R2  = matrix_from_9d(r9d)
        assert torch.allclose(R, R2, atol=1e-5)

    def test_noisy_roundtrip(self):
        """matrix_from_9d should project noisy inputs onto SO(3)."""
        R    = random_rotation(32)
        r9d  = matrix_to_9d(R) + torch.randn(32, 9) * 0.1   # add noise
        R2   = matrix_from_9d(r9d)
        I    = R2 @ R2.transpose(-1, -2)
        assert torch.allclose(I, torch.eye(3).expand_as(I), atol=1e-5)

    def test_se3_roundtrip(self):
        R = random_rotation(16)
        t = torch.randn(16, 3)
        T = torch.eye(4).unsqueeze(0).expand(16, 4, 4).clone()
        T[:, :3, :3] = R
        T[:, :3, 3]  = t
        r9d = se3_to_9d_repr(T)
        T2  = se3_from_9d_repr(r9d)
        assert torch.allclose(T[:, :3, :3], T2[:, :3, :3], atol=1e-5)
        assert torch.allclose(T[:, :3, 3],  T2[:, :3, 3],  atol=1e-5)
