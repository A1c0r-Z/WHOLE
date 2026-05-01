"""Tests for data/preprocessing.py.

Checks:
  1. Gravity alignment produces z = up, origin at camera
  2. 73D diffusion variable x0 has correct layout
  3. Noise injection changes params without destroying validity
  4. Full pipeline on real HOT3D clip
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from data.preprocessing import (
    estimate_gravity_rotation,
    apply_gravity_alignment,
    gravity_align_window,
    build_diffusion_variable,
    inject_hand_noise,
    build_noisy_hand_feature,
    preprocess_window,
)
from data.hot3d_loader import HOT3DDataset

TRAIN_ARIA = Path('/scr/cezhao/workspace/HOI_recon/_DATA/hot3d/train_aria')
HAS_DATA   = TRAIN_ARIA.exists() and any(TRAIN_ARIA.glob('clip-*.tar'))


# ---------------------------------------------------------------------------
# Gravity alignment
# ---------------------------------------------------------------------------

class TestGravityAlignment:
    def _make_T_wc(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3]  = t
        return T

    def test_z_aligns_to_up(self):
        """After alignment the gravity-up direction should map to (0,0,1)."""
        # Camera y-axis in world = gravity-up proxy used in our estimator
        R_wc = np.eye(3, dtype=np.float32)     # identity: cam y IS world y
        T_wc = self._make_T_wc(R_wc, np.zeros(3, np.float32))
        R_align = estimate_gravity_rotation(T_wc)

        # gravity_up = R_wc[:,1] = [0,1,0]; after alignment should be [0,0,1]
        gravity_up = R_wc[:, 1]
        result = R_align @ gravity_up
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-5)

    def test_output_is_rotation(self):
        """R_align must be a proper rotation matrix."""
        T_wc = self._make_T_wc(np.eye(3, dtype=np.float32), np.array([1, 2, 3], np.float32))
        R = estimate_gravity_rotation(T_wc)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5, err_msg='Not orthogonal')
        assert abs(np.linalg.det(R) - 1.0) < 1e-5, 'Det != 1'

    def test_apply_moves_origin(self):
        """After alignment, the reference camera position should be at origin."""
        t0 = np.array([1.5, -0.3, 0.7], dtype=np.float32)
        T_wc = self._make_T_wc(np.eye(3, dtype=np.float32), t0)
        R_align = estimate_gravity_rotation(T_wc)

        # Apply to the camera pose itself
        T_aligned = apply_gravity_alignment(T_wc[np.newaxis], R_align, t0)[0]
        np.testing.assert_allclose(T_aligned[:3, 3], [0, 0, 0], atol=1e-5,
                                   err_msg='Origin not at camera position after alignment')

    def test_apply_preserves_rotation(self):
        """apply_gravity_alignment should rotate the rotation part too."""
        rng = np.random.default_rng(7)
        R_wc   = np.linalg.qr(rng.standard_normal((3, 3)))[0].astype(np.float32)
        t0     = rng.standard_normal(3).astype(np.float32)
        T_wc   = self._make_T_wc(R_wc, t0)
        R_align = estimate_gravity_rotation(T_wc)
        T_al   = apply_gravity_alignment(T_wc[np.newaxis], R_align, t0)[0]
        expected_R = R_align @ R_wc
        np.testing.assert_allclose(T_al[:3, :3], expected_R, atol=1e-5)


# ---------------------------------------------------------------------------
# Diffusion variable x0
# ---------------------------------------------------------------------------

class TestDiffusionVariable:
    @pytest.fixture
    def window(self):
        """Minimal synthetic window dict."""
        rng = np.random.default_rng(42)
        T = 120
        # Build a fake SE(3) sequence for the object
        R = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
        t = rng.standard_normal((T, 3)).astype(np.float32) * 0.1
        obj_T = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        obj_T[:, :3, :3] = R
        obj_T[:, :3,  3] = t

        # Gravity alignment already applied (identity for simplicity)
        T_wc0 = np.eye(4, dtype=np.float32)
        T_wc0[1, 1] = 1.0  # y-axis up (identity → gravity_up = y)

        return {
            'obj_T_world':            obj_T,
            'left_thetas':            rng.standard_normal((T, 15)).astype(np.float32),
            'left_wrist':             rng.standard_normal((T,  6)).astype(np.float32),
            'left_betas':             rng.standard_normal(10).astype(np.float32),
            'right_thetas':           rng.standard_normal((T, 15)).astype(np.float32),
            'right_wrist':            rng.standard_normal((T,  6)).astype(np.float32),
            'right_betas':            rng.standard_normal(10).astype(np.float32),
            'left_valid':             np.ones(T, dtype=bool),
            'right_valid':            np.ones(T, dtype=bool),
            'frame_valid':            np.ones(T, dtype=bool),
            'T_world_from_ref_cam0':  T_wc0,
            'clip_id':                'test_clip',
            'device':                 'Aria',
            'obj_name':               'test_obj',
            'obj_bop_id':             '1',
        }

    def test_x0_shape(self, window):
        w = gravity_align_window(window.copy())
        x0 = build_diffusion_variable(w)
        assert x0.shape == (120, 73), x0.shape

    def test_x0_dtype(self, window):
        w = gravity_align_window(window.copy())
        x0 = build_diffusion_variable(w)
        assert x0.dtype == np.float32

    def test_contact_zeros(self, window):
        """Contact labels should be zero (filled by VLM later)."""
        w = gravity_align_window(window.copy())
        x0 = build_diffusion_variable(w)
        np.testing.assert_array_equal(x0[:, 9:11], 0.0,
                                      err_msg='Contact dims should be 0')

    def test_obj_9d_is_valid_rotation(self, window):
        """The 9D object representation should decode to a valid rotation."""
        from utils.rotation import matrix_from_9d, se3_from_9d_repr
        w  = gravity_align_window(window.copy())
        x0 = build_diffusion_variable(w)
        obj_9d = torch.tensor(x0[:, :9])
        T_rec  = se3_from_9d_repr(obj_9d)   # (T, 4, 4)
        R = T_rec[:, :3, :3]
        I = R @ R.transpose(-1, -2)
        assert torch.allclose(I, torch.eye(3).expand_as(I), atol=1e-5)

    def test_hand_features_match_source(self, window):
        """Slices [11:14] and [14:17] should equal wrist orient and transl."""
        w  = gravity_align_window(window.copy())
        x0 = build_diffusion_variable(w)
        # Left global_orient: dims 11:14
        np.testing.assert_allclose(x0[:, 11:14], w['left_wrist'][:, :3], atol=1e-5)
        # Left transl: dims 14:17
        np.testing.assert_allclose(x0[:, 14:17], w['left_wrist'][:, 3:], atol=1e-5)


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

class TestNoiseInjection:
    def _dummy_window(self, T=30):
        rng = np.random.default_rng(0)
        return {
            'left_thetas':  rng.standard_normal((T, 15)).astype(np.float32),
            'left_wrist':   rng.standard_normal((T,  6)).astype(np.float32),
            'left_betas':   rng.standard_normal(10).astype(np.float32),
            'right_thetas': rng.standard_normal((T, 15)).astype(np.float32),
            'right_wrist':  rng.standard_normal((T,  6)).astype(np.float32),
            'right_betas':  rng.standard_normal(10).astype(np.float32),
        }

    def test_noisy_keys_added(self):
        w = self._dummy_window()
        inject_hand_noise(w)
        for side in ('left', 'right'):
            assert f'{side}_thetas_noisy' in w
            assert f'{side}_wrist_noisy'  in w
            assert f'{side}_drop_mask'    in w

    def test_shapes_preserved(self):
        T = 40
        w = self._dummy_window(T=T)
        inject_hand_noise(w)
        for side in ('left', 'right'):
            assert w[f'{side}_thetas_noisy'].shape == (T, 15)
            assert w[f'{side}_wrist_noisy'].shape  == (T, 6)
            assert w[f'{side}_drop_mask'].shape    == (T,)

    def test_different_from_clean(self):
        """Noisy params should differ from clean (with high probability)."""
        w = self._dummy_window(T=120)
        orig_l = w['left_thetas'].copy()
        inject_hand_noise(w, traj_noise_std=0.1, frame_noise_std=0.05)
        assert not np.allclose(w['left_thetas_noisy'], orig_l), \
            'Noisy thetas identical to clean — noise not applied'

    def test_dropped_frames_zero(self):
        """Dropped frames must be exactly zero."""
        w = self._dummy_window(T=200)
        inject_hand_noise(w, drop_prob=0.5)
        for side in ('left', 'right'):
            mask = w[f'{side}_drop_mask']
            assert mask.sum() > 0, 'No frames dropped (unlikely at p=0.5)'
            np.testing.assert_array_equal(w[f'{side}_thetas_noisy'][mask], 0.0)
            np.testing.assert_array_equal(w[f'{side}_wrist_noisy'][mask],  0.0)

    def test_noisy_hand_feature_shape(self):
        w = self._dummy_window(T=120)
        inject_hand_noise(w)
        H_tilde = build_noisy_hand_feature(w)
        assert H_tilde.shape == (120, 62), H_tilde.shape


# ---------------------------------------------------------------------------
# Full pipeline on real HOT3D data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DATA, reason='HOT3D train_aria not available')
class TestFullPipeline:
    def test_preprocess_window_shapes(self):
        ds   = HOT3DDataset(TRAIN_ARIA, max_clips=1, load_images=False)
        item = ds[0]
        out  = preprocess_window(item, augment=True)
        assert out['x0'].shape      == (120, 73)
        assert out['H_tilde'].shape == (120, 62)
        assert out['frame_valid'].shape == (120,)

    def test_x0_finite(self):
        ds   = HOT3DDataset(TRAIN_ARIA, max_clips=2, load_images=False)
        for i in range(min(4, len(ds))):
            out = preprocess_window(ds[i], augment=True)
            assert np.isfinite(out['x0']).all(),      f'x0 has NaN/Inf in window {i}'
            assert np.isfinite(out['H_tilde']).all(), f'H_tilde has NaN/Inf in window {i}'

    def test_gravity_align_z_up(self):
        """After gravity alignment, mean object z should be reasonable (not NaN)."""
        ds   = HOT3DDataset(TRAIN_ARIA, max_clips=1, load_images=False)
        item = ds[0]
        out  = preprocess_window(item, augment=False)
        x0   = out['x0']
        obj_z = x0[:, 8]   # translation z from 9D repr (last of 9)
        assert np.isfinite(obj_z).all()
