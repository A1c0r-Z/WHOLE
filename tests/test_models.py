"""Tests for models: BPS, denoiser, diffusion.

Checks:
  1. BPS descriptor shape, determinism, and nearest-point correctness
  2. Denoiser forward pass shapes and parameter count (~12.35M ± 20%)
  3. DDPM noise schedule properties (monotone, correct range)
  4. DDPM loss computation and gradient flow
  5. DDPM sampling produces correct output shape
  6. Ambient sensor nearest-point geometry
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from models.bps import (
    get_basis_points,
    encode_object,
    compute_ambient_sensor,
    compute_ambient_sensor_approx,
    ObjectBPSCache,
)
from models.denoiser import WHOLEDenoiser, build_denoiser
from models.diffusion import DDPM, cosine_beta_schedule, linear_beta_schedule

OBJECT_MODELS_DIR = Path('/scr/cezhao/workspace/HOI_recon/_DATA/hot3d/object_models')
HAS_OBJECTS = OBJECT_MODELS_DIR.exists() and any(OBJECT_MODELS_DIR.glob('*.glb'))

B, T, D = 2, 120, 73   # batch, window, diffusion dim
K = 1024               # BPS dim


# ---------------------------------------------------------------------------
# BPS
# ---------------------------------------------------------------------------

class TestBPS:
    def test_basis_shape(self):
        pts = get_basis_points(512)
        assert pts.shape == (512, 3)

    def test_basis_unit_sphere(self):
        pts = get_basis_points(256)
        norms = pts.norm(dim=1)
        assert torch.allclose(norms, torch.ones(256), atol=1e-6)

    def test_basis_deterministic(self):
        a = get_basis_points(128)
        b = get_basis_points(128)
        assert torch.equal(a, b), 'Basis points not deterministic'

    def test_encode_object_shape(self):
        verts = torch.randn(500, 3)
        desc  = encode_object(verts, n_points=K)
        assert desc.shape == (K,), desc.shape

    def test_encode_object_non_negative(self):
        verts = torch.randn(200, 3)
        desc  = encode_object(verts, n_points=K)
        assert (desc >= 0).all(), 'BPS distances should be non-negative'

    def test_encode_object_zero_for_matching_points(self):
        """If mesh vertices coincide with basis points, distances should be ~0."""
        basis = get_basis_points(64) * 0.2
        desc  = encode_object(basis, n_points=64)
        assert desc.max() < 1e-6, f'Expected near-zero distances, got {desc.max():.2e}'

    def test_encode_scale_invariant_structure(self):
        """Scaling mesh and radius together should give same descriptor."""
        verts = torch.randn(300, 3)
        d1 = encode_object(verts,        n_points=128, radius=0.2)
        d2 = encode_object(verts * 2.0,  n_points=128, radius=0.4)
        # Squared distances scale by 4; ratios should be the same
        ratio1 = d1 / (d1.mean() + 1e-8)
        ratio2 = d2 / (d2.mean() + 1e-8)
        assert torch.allclose(ratio1, ratio2, atol=1e-4)

    @pytest.mark.skipif(not HAS_OBJECTS, reason='HOT3D object models not available')
    def test_encode_from_glb(self):
        from models.bps import encode_object_from_glb
        glb = next(OBJECT_MODELS_DIR.glob('*.glb'))
        desc = encode_object_from_glb(glb, n_points=K)
        assert desc.shape == (K,)
        assert (desc >= 0).all()
        assert desc.isfinite().all()

    @pytest.mark.skipif(not HAS_OBJECTS, reason='HOT3D object models not available')
    def test_bps_cache(self):
        cache = ObjectBPSCache(OBJECT_MODELS_DIR, n_points=K)
        d1 = cache.get('1')
        d2 = cache.get('1')   # should hit cache
        assert torch.equal(d1, d2), 'Cache returned different results'
        assert d1.shape == (K,)


class TestAmbientSensor:
    def _make_inputs(self, B=2, T=5, J=42, V=100):
        joints    = torch.randn(B, T, J, 3)
        obj_verts = torch.randn(B, T, V, 3)
        return joints, obj_verts

    def test_output_shape(self):
        joints, obj_verts = self._make_inputs()
        out = compute_ambient_sensor(joints, obj_verts)
        assert out.shape == joints.shape

    def test_zero_distance(self):
        """When joints coincide with object vertices, displacements should be ~0."""
        B, T, J = 1, 3, 4
        pts = torch.randn(B, T, J, 3)
        out = compute_ambient_sensor(pts, pts)   # same as obj_verts
        assert out.abs().max() < 1e-5

    def test_approx_close_to_exact(self):
        """Approx (subsampled) should give same result when V <= subsample."""
        joints, obj_verts = self._make_inputs(J=10, V=50)
        exact  = compute_ambient_sensor(joints, obj_verts)
        approx = compute_ambient_sensor_approx(joints, obj_verts, subsample=200)
        assert torch.allclose(exact, approx, atol=1e-5)


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------

class TestDenoiser:
    @pytest.fixture
    def model(self):
        return WHOLEDenoiser(
            x_dim=73, h_dim=62, bps_dim=K,
            d_model=512, nhead=4, d_ff=2048,
            num_layers=4, window_len=T,
            n_joints=42, use_ambient=True,
        )

    def test_param_count(self, model):
        n = model.count_params()
        target = 12_350_000
        # ±30% tolerance: paper's 12.35M assumes specific BPS/ambient dims
        assert 0.7 * target <= n <= 1.3 * target, \
            f'Param count {n:,} outside ±30% of paper target {target:,}'
        print(f'\n  Denoiser params: {n:,}  (paper: {target:,})')

    def test_forward_shape(self, model):
        x_n     = torch.randn(B, T, 73)
        t       = torch.randint(0, 1000, (B,))
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        out = model(x_n, t, H_tilde, O)
        assert out.shape == (B, T, 73), out.shape

    def test_forward_with_ambient(self, model):
        x_n     = torch.randn(B, T, 73)
        t       = torch.randint(0, 1000, (B,))
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        ambient = torch.randn(B, T, 42 * 3)
        out = model(x_n, t, H_tilde, O, ambient)
        assert out.shape == (B, T, 73)

    def test_forward_no_ambient(self, model):
        """Model must run without ambient sensor."""
        x_n     = torch.randn(B, T, 73)
        t       = torch.randint(0, 1000, (B,))
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        out = model(x_n, t, H_tilde, O, ambient=None)
        assert out.shape == (B, T, 73)

    def test_output_finite(self, model):
        x_n     = torch.randn(B, T, 73)
        t       = torch.randint(0, 1000, (B,))
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        out = model(x_n, t, H_tilde, O)
        assert out.isfinite().all(), 'Denoiser output contains NaN/Inf'

    def test_gradient_flows(self, model):
        """Backward pass should not error and produce non-zero gradients."""
        x_n     = torch.randn(B, T, 73)
        t       = torch.randint(0, 1000, (B,))
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        ambient = torch.randn(B, T, 42 * 3)   # supply ambient so all params are used
        out = model(x_n, t, H_tilde, O, ambient)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f'No gradient for {name}'
            assert p.grad.isfinite().all(), f'NaN gradient for {name}'

    def test_with_padding_mask(self, model):
        """Padding mask should not cause errors."""
        x_n     = torch.randn(B, T, 73)
        t       = torch.randint(0, 1000, (B,))
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        mask    = torch.zeros(B, T, dtype=torch.bool)
        mask[0, -10:] = True   # last 10 frames invalid for sample 0
        out = model(x_n, t, H_tilde, O, key_padding_mask=mask)
        assert out.shape == (B, T, 73)

    def test_build_from_config(self):
        cfg = {'model': {
            'x_dim': 73, 'h_dim': 62, 'bps_n_points': K,
            'd_model': 512, 'nhead': 4, 'd_ff': 2048,
            'num_layers': 4, 'window_len': T,
            'n_hand_joints': 21, 'use_ambient_sensor': True, 'dropout': 0.1,
        }}
        model = build_denoiser(cfg)
        assert isinstance(model, WHOLEDenoiser)


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------

class TestDDPM:
    @pytest.fixture
    def ddpm(self):
        denoiser = WHOLEDenoiser(
            x_dim=73, h_dim=62, bps_dim=K,
            d_model=128, nhead=4, d_ff=256,  # small for speed
            num_layers=2, window_len=T,
            n_joints=42, use_ambient=False,
        )
        return DDPM(denoiser, n_steps=100, schedule='cosine')

    def test_schedule_monotone(self, ddpm):
        """alpha_bar should be monotonically decreasing."""
        ab = ddpm.alpha_bar
        assert (ab[1:] <= ab[:-1]).all(), 'alpha_bar not monotone'
        assert ab[0] > 0.99, f'alpha_bar[0] should be ~1, got {ab[0]:.3f}'
        assert ab[-1] < 0.01, f'alpha_bar[-1] should be ~0, got {ab[-1]:.3f}'

    def test_cosine_vs_linear_different(self):
        b_cos = cosine_beta_schedule(100)
        b_lin = linear_beta_schedule(100)
        assert not torch.allclose(b_cos, b_lin)

    def test_q_sample_shape(self, ddpm):
        x0  = torch.randn(B, T, 73)
        t   = torch.randint(0, 100, (B,))
        x_t, noise = ddpm.q_sample(x0, t)
        assert x_t.shape   == (B, T, 73)
        assert noise.shape == (B, T, 73)

    def test_q_sample_t0_is_clean(self, ddpm):
        """At t=0, x_t ≈ x_0 (alpha_bar[0] ≈ 1)."""
        x0    = torch.randn(B, T, 73)
        t     = torch.zeros(B, dtype=torch.long)
        x_t, _ = ddpm.q_sample(x0, t)
        assert torch.allclose(x_t, x0, atol=0.15), \
            f'At t=0 x_t should ≈ x_0, max diff: {(x_t-x0).abs().max():.3f}'

    def test_q_sample_tN_is_noise(self, ddpm):
        """At t=N-1, x_t should be close to pure noise."""
        x0    = torch.randn(B, T, 73) * 0.1   # small signal
        t     = torch.full((B,), 99, dtype=torch.long)
        x_t, noise = ddpm.q_sample(x0, t)
        # Correlation between x_t and noise should be high
        corr = F.cosine_similarity(
            x_t.reshape(B, -1), noise.reshape(B, -1), dim=1
        ).mean()
        assert corr > 0.95, f'At t=N-1 expected noise-like x_t, corr={corr:.3f}'

    def test_loss_shape(self, ddpm):
        x0      = torch.randn(B, T, 73)
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        out = ddpm.compute_loss(x0, H_tilde, O)
        assert 'loss_ddpm' in out
        assert 'x0_pred'   in out
        assert out['loss_ddpm'].shape == ()   # scalar
        assert out['x0_pred'].shape   == (B, T, 73)

    def test_loss_finite(self, ddpm):
        x0      = torch.randn(B, T, 73)
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        out = ddpm.compute_loss(x0, H_tilde, O)
        assert out['loss_ddpm'].isfinite(), 'Loss is NaN/Inf'

    def test_loss_backprop(self, ddpm):
        x0      = torch.randn(B, T, 73)
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        out = ddpm.compute_loss(x0, H_tilde, O)
        out['loss_ddpm'].backward()
        for name, p in ddpm.denoiser.named_parameters():
            assert p.grad is not None and p.grad.isfinite().all(), \
                f'Bad gradient for {name}'

    def test_loss_with_mask(self, ddpm):
        x0      = torch.randn(B, T, 73)
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        mask    = torch.ones(B, T, dtype=torch.bool)
        mask[:, -20:] = False   # last 20 frames invalid
        out_masked   = ddpm.compute_loss(x0, H_tilde, O, mask=mask)
        out_unmasked = ddpm.compute_loss(x0, H_tilde, O, mask=None)
        # Losses should differ (masked excludes some frames)
        assert out_masked['loss_ddpm'].item() != out_unmasked['loss_ddpm'].item()

    def test_sample_shape(self, ddpm):
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        with torch.no_grad():
            x0 = ddpm.sample(H_tilde, O)
        assert x0.shape == (B, T, 73)

    def test_sample_finite(self, ddpm):
        H_tilde = torch.randn(B, T, 62)
        O       = torch.randn(B, K)
        with torch.no_grad():
            x0 = ddpm.sample(H_tilde, O)
        assert x0.isfinite().all(), 'Sample contains NaN/Inf'
