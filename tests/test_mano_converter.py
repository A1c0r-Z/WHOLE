"""Tests for data/mano_converter.py.

Key checks:
  1. PCA expansion matches smplx's internal expansion (ground truth)
  2. FK output shapes and physical plausibility (joint distances)
  3. build_hand_feature layout and value consistency
  4. Left/right hand model independence
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest
from tests.conftest import requires_chumpy

from data.mano_converter import (
    pca_to_axis_angle,
    build_hand_feature,
    mano_forward,
    _load_mano_npz,
)

MANO_DIR  = Path('/scr/cezhao/workspace/HOI_recon/hamer/_DATA/data/mano')
HAS_LEFT  = (MANO_DIR / 'MANO_LEFT.pkl').exists()
HAS_RIGHT = (MANO_DIR / 'MANO_RIGHT.pkl').exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smplx_pca_expand(thetas_np: np.ndarray, side: str) -> np.ndarray:
    """Reference PCA expansion using smplx.create (ground truth)."""
    import smplx, torch
    layer = smplx.create(
        str(MANO_DIR / f'MANO_{side.upper()}.pkl'),
        model_type='mano', use_pca=True, num_pca_comps=15,
        is_rhand=(side == 'right'), flat_hand_mean=True, batch_size=1,
    )
    t = torch.tensor(thetas_np, dtype=torch.float32)
    # smplx: full_pose = thetas @ hand_components + hand_mean
    expanded = (t @ layer.hand_components + layer.hand_mean).detach().numpy()
    return expanded


def _dummy_window(T: int = 10, side: str = 'right'):
    rng = np.random.default_rng(0)
    thetas = rng.standard_normal((T, 15)).astype(np.float32) * 0.3
    wrist  = rng.standard_normal((T,  6)).astype(np.float32) * 0.1
    betas  = rng.standard_normal(10).astype(np.float32) * 0.5
    return thetas, wrist, betas


# ---------------------------------------------------------------------------
# PCA expansion
# ---------------------------------------------------------------------------

class TestPCAExpansion:
    @pytest.mark.parametrize('side', ['right', 'left'])
    def test_shape(self, side):
        if side == 'left' and not HAS_LEFT:
            pytest.skip('MANO_LEFT.pkl not found')
        thetas, _, _ = _dummy_window(T=8, side=side)
        out = pca_to_axis_angle(thetas, side=side)
        assert out.shape == (8, 45), f"Expected (8,45), got {out.shape}"

    @requires_chumpy
    @pytest.mark.parametrize('side', ['right', 'left'])
    def test_matches_smplx(self, side):
        """Our npz-based expansion must match smplx's internal expansion."""
        if side == 'left' and not HAS_LEFT:
            pytest.skip('MANO_LEFT.pkl not found')
        thetas, _, _ = _dummy_window(T=16, side=side)
        ours = pca_to_axis_angle(thetas, side=side)
        ref  = _smplx_pca_expand(thetas, side=side)
        np.testing.assert_allclose(
            ours, ref, atol=1e-5,
            err_msg=f'PCA expansion mismatch for {side} hand'
        )

    def test_batch_shape(self):
        """pca_to_axis_angle should handle arbitrary leading dims."""
        thetas = np.random.randn(3, 7, 15).astype(np.float32)
        out = pca_to_axis_angle(thetas, side='right')
        assert out.shape == (3, 7, 45)

    def test_zero_thetas(self):
        """Zero PCA coefficients should give the mean hand pose."""
        data = _load_mano_npz('right')
        thetas = np.zeros((1, 15), dtype=np.float32)
        out = pca_to_axis_angle(thetas, side='right')[0]
        np.testing.assert_allclose(out, data['hand_mean'], atol=1e-6)


# ---------------------------------------------------------------------------
# MANO Forward Kinematics
# ---------------------------------------------------------------------------

class TestMANOForwardKinematics:
    @pytest.mark.parametrize('side', ['right', 'left'])
    def test_output_shapes(self, side):
        if side == 'left' and not HAS_LEFT:
            pytest.skip('MANO_LEFT.pkl not found')
        T = 20
        thetas, wrist, betas = _dummy_window(T=T, side=side)
        out = mano_forward(thetas, wrist, betas, side=side)
        assert out['joints'].shape   == (T, 21, 3), out['joints'].shape
        assert out['vertices'].shape == (T, 778, 3), out['vertices'].shape

    @pytest.mark.parametrize('side', ['right', 'left'])
    def test_joints_plausible_scale(self, side):
        """Hand joints should be within a few cm of each other, not meters apart."""
        if side == 'left' and not HAS_LEFT:
            pytest.skip('MANO_LEFT.pkl not found')
        thetas, wrist, betas = _dummy_window(T=5, side=side)
        # Zero translation so hand is near origin
        wrist[:, 3:] = 0.0
        out = mano_forward(thetas, wrist, betas, side=side)
        J   = out['joints']   # (T, 21, 3)
        # Max inter-joint distance should be < 0.3m (hand span)
        dists = np.linalg.norm(J[:, :, np.newaxis] - J[:, np.newaxis, :], axis=-1)
        assert dists.max() < 0.3, f"Implausible joint distance: {dists.max():.3f} m"

    @requires_chumpy
    @pytest.mark.parametrize('side', ['right', 'left'])
    def test_translation_applied(self, side):
        """Translating the wrist should move all joints by the same offset."""
        if side == 'left' and not HAS_LEFT:
            pytest.skip('MANO_LEFT.pkl not found')
        T = 4
        thetas, wrist, betas = _dummy_window(T=T, side=side)
        wrist[:, 3:] = 0.0
        out0 = mano_forward(thetas, wrist.copy(), betas, side=side)

        offset = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        wrist[:, 3:] = offset
        out1 = mano_forward(thetas, wrist.copy(), betas, side=side)

        diff = out1['joints'] - out0['joints']   # (T, 21, 3)
        # Every joint of every frame should shift by exactly `offset`
        np.testing.assert_allclose(
            diff, np.broadcast_to(offset, diff.shape), atol=1e-4,
            err_msg='Translation not applied correctly',
        )

    @requires_chumpy
    @pytest.mark.parametrize('side', ['right', 'left'])
    def test_matches_smplx_direct(self, side):
        """FK via our pipeline should match smplx called with the same params."""
        if side == 'left' and not HAS_LEFT:
            pytest.skip('MANO_LEFT.pkl not found')
        import smplx, torch

        T = 8
        thetas, wrist, betas = _dummy_window(T=T, side=side)
        ours = mano_forward(thetas, wrist, betas, side=side)

        layer = smplx.create(
            str(MANO_DIR / f'MANO_{side.upper()}.pkl'),
            model_type='mano', use_pca=True, num_pca_comps=15,
            is_rhand=(side == 'right'), flat_hand_mean=True, batch_size=T,
        )
        with torch.no_grad():
            ref = layer(
                global_orient = torch.tensor(wrist[:, :3]),
                hand_pose     = torch.tensor(thetas),
                transl        = torch.tensor(wrist[:, 3:]),
                betas         = torch.tensor(betas).unsqueeze(0).expand(T, -1),
                return_verts  = True,
            )

        np.testing.assert_allclose(
            ours['vertices'],
            ref.vertices.numpy(),
            atol=1e-4,
            err_msg=f'Vertices mismatch for {side} hand — check smplx.create vs MANOLayer',
        )


# ---------------------------------------------------------------------------
# build_hand_feature
# ---------------------------------------------------------------------------

class TestBuildHandFeature:
    def test_shape(self):
        thetas, wrist, betas = _dummy_window(T=120)
        feat = build_hand_feature(thetas, wrist, betas)
        assert feat.shape == (120, 31), feat.shape

    def test_layout(self):
        """Check each slice of the 31D vector is from the right source."""
        thetas, wrist, betas = _dummy_window(T=5)
        feat = build_hand_feature(thetas, wrist, betas)

        # [0:3] = global_orient = wrist[:, :3]
        np.testing.assert_array_equal(feat[:, :3], wrist[:, :3])
        # [3:6] = transl = wrist[:, 3:]
        np.testing.assert_array_equal(feat[:, 3:6], wrist[:, 3:])
        # [6:21] = thetas
        np.testing.assert_array_equal(feat[:, 6:21], thetas)
        # [21:31] = betas (tiled)
        np.testing.assert_array_equal(feat[:, 21:31], np.tile(betas, (5, 1)))

    def test_dtype(self):
        thetas, wrist, betas = _dummy_window()
        feat = build_hand_feature(thetas, wrist, betas)
        assert feat.dtype == np.float32
