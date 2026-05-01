"""Tests for losses/ and training step.

Checks:
  1. L_smooth: shape, zero for constant trajectory, finite with mask
  2. L_consistency: wrist fallback and (when FK available) joint version
  3. L_interaction: contact distance and near-rigid transport geometry
  4. training_step: runs end-to-end and produces finite losses
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from losses.smoothness  import loss_smooth
from losses.consistency import loss_consistency, _wrist_consistency
from losses.interaction import (
    loss_interaction, _contact_distance, _near_rigid_transport, _wrist_interaction,
)
from utils.mano_utils import unpack_x0, apply_obj_transform, get_obj_transform

B, T = 2, 120


def _rand_x0(B=B, T=T) -> torch.Tensor:
    """Random valid-ish 73D diffusion variable."""
    x = torch.randn(B, T, 73) * 0.1
    # Make 9D obj block decode to a valid rotation (use identity + small noise)
    eye9 = torch.zeros(9)
    eye9[0] = 1.0; eye9[4] = 1.0; eye9[8] = 1.0  # identity matrix columns
    x[..., :9] = eye9 + torch.randn(B, T, 9) * 0.01
    return x


def _rand_verts(V=100) -> torch.Tensor:
    return torch.randn(V, 3) * 0.05   # small object


# ---------------------------------------------------------------------------
# L_smooth
# ---------------------------------------------------------------------------

class TestLossSmooth:
    def test_zero_for_constant(self):
        """Constant trajectory has zero acceleration → loss = 0."""
        x = torch.ones(B, T, 73)
        assert loss_smooth(x).item() == pytest.approx(0.0, abs=1e-6)

    def test_zero_for_linear(self):
        """Linear trajectory has zero acceleration."""
        t   = torch.linspace(0, 1, T).view(1, T, 1)
        x   = t.expand(B, T, 73)
        assert loss_smooth(x).item() == pytest.approx(0.0, abs=1e-5)

    def test_nonzero_for_random(self):
        x = torch.randn(B, T, 73)
        assert loss_smooth(x).item() > 0

    def test_finite(self):
        x = torch.randn(B, T, 73)
        assert loss_smooth(x).isfinite()

    def test_with_mask(self):
        x    = torch.randn(B, T, 73)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, -10:] = False
        l_masked   = loss_smooth(x, mask)
        l_unmasked = loss_smooth(x, None)
        assert l_masked.isfinite()
        assert l_masked.item() != l_unmasked.item()

    def test_gradient_flows(self):
        x = torch.randn(B, T, 73, requires_grad=True)
        loss_smooth(x).backward()
        assert x.grad is not None and x.grad.isfinite().all()


# ---------------------------------------------------------------------------
# L_consistency
# ---------------------------------------------------------------------------

class TestLossConsistency:
    def test_zero_when_equal(self):
        """When pred == GT, consistency loss should be 0."""
        x = _rand_x0()
        assert _wrist_consistency(x, x, None).item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_different(self):
        x_pred = _rand_x0()
        x_gt   = _rand_x0()
        assert _wrist_consistency(x_pred, x_gt, None).item() > 0

    def test_finite(self):
        x_pred = _rand_x0(); x_gt = _rand_x0()
        assert loss_consistency(x_pred, x_gt).isfinite()

    def test_gradient_flows(self):
        x_pred = _rand_x0().requires_grad_(True)
        x_gt   = _rand_x0()
        _wrist_consistency(x_pred, x_gt, None).backward()
        assert x_pred.grad is not None and x_pred.grad.isfinite().all()

    def test_with_mask(self):
        x_pred = _rand_x0(); x_gt = _rand_x0()
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, -20:] = False
        l = loss_consistency(x_pred, x_gt, mask)
        assert l.isfinite()


# ---------------------------------------------------------------------------
# L_interaction
# ---------------------------------------------------------------------------

class TestLossInteraction:
    def test_contact_distance_zero_when_on_surface(self):
        """Joints placed on object surface → contact distance ≈ 0."""
        V = 50
        obj_v = _rand_verts(V).unsqueeze(0).unsqueeze(0).expand(B, T, V, 3)
        # Place all joints on first object vertex
        joints = obj_v[:, :, :5, :]   # (B, T, 5, 3) — on surface
        contact = torch.ones(B, T, 5)
        loss = _contact_distance(joints, obj_v, contact, None)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_near_rigid_matches_static_object(self):
        """If object doesn't move, near-rigid transport error should be ~0."""
        V  = 30
        # Static object (same pose every frame)
        template = _rand_verts(V)
        obj_v = template.unsqueeze(0).unsqueeze(0).expand(B, T, V, 3).clone()
        # Identity transforms for all frames
        T_obj = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, 4, 4)
        # Place joints on surface
        joints  = obj_v[:, :, :4, :]   # (B, T, 4, 3)
        contact = torch.ones(B, T, 4)
        loss = _near_rigid_transport(joints, obj_v, T_obj, contact, None)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_interaction_finite(self):
        x = _rand_x0()
        template = _rand_verts(50)
        l = loss_interaction(x, template)
        assert l.isfinite(), f'L_inter = {l.item()}'

    def test_interaction_with_gt_contact(self):
        x        = _rand_x0()
        template = _rand_verts(50)
        contact  = (torch.rand(B, T, 2) > 0.5).float()
        l = loss_interaction(x, template, contact_gt=contact)
        assert l.isfinite()

    def test_wrist_fallback_finite(self):
        x        = _rand_x0()
        template = _rand_verts(30)
        T_obj    = get_obj_transform(x)
        obj_v    = apply_obj_transform(T_obj, template)
        contact  = (torch.rand(B, T, 2) > 0.5).float()
        l = _wrist_interaction(x, obj_v, contact, None)
        assert l.isfinite()

    def test_gradient_flows(self):
        x        = _rand_x0().requires_grad_(True)
        template = _rand_verts(50)
        loss_interaction(x, template).backward()
        assert x.grad is not None and x.grad.isfinite().all()


# ---------------------------------------------------------------------------
# End-to-end training step smoke test
# ---------------------------------------------------------------------------

class TestTrainingStep:
    @pytest.fixture
    def setup(self):
        """Small model + random batch for smoke-testing training_step."""
        import yaml
        from models import build_denoiser, build_diffusion
        from train import training_step

        cfg = yaml.safe_load(open('configs/default.yaml'))
        # Override to tiny model for speed
        cfg['model'].update({'d_model': 64, 'nhead': 4, 'd_ff': 128,
                             'num_layers': 2, 'bps_n_points': 64})
        cfg['training']['aux_loss_start'] = 0   # enable aux losses immediately

        device   = torch.device('cpu')
        denoiser = build_denoiser(cfg).to(device)
        diffusion = build_diffusion(denoiser, cfg).to(device)

        batch = {
            'x0':          torch.randn(2, T, 73),
            'H_tilde':     torch.randn(2, T, 62),
            'O':           torch.randn(2, 64),
            'frame_valid': torch.ones(2, T, dtype=torch.bool),
            'obj_bop_id':  ['1', '1'],
        }
        return diffusion, denoiser, batch, cfg, device, training_step

    def test_all_losses_finite(self, setup):
        diffusion, denoiser, batch, cfg, device, training_step = setup
        template_cache = {'1': _rand_verts(50)}
        losses = training_step(batch, denoiser, diffusion, cfg, step=0,
                               device=device, obj_template_cache=template_cache)
        for k, v in losses.items():
            assert v.isfinite(), f'{k} = {v.item()}'

    def test_backprop_succeeds(self, setup):
        diffusion, denoiser, batch, cfg, device, training_step = setup
        template_cache = {'1': _rand_verts(50)}
        losses = training_step(batch, denoiser, diffusion, cfg, step=0,
                               device=device, obj_template_cache=template_cache)
        losses['loss'].backward()
        any_grad = any(
            p.grad is not None
            for p in denoiser.parameters()
            if p.requires_grad
        )
        assert any_grad, 'No gradients after backward'

    def test_ddpm_only_before_curriculum(self, setup):
        diffusion, denoiser, batch, cfg, device, training_step = setup
        cfg['training']['aux_loss_start'] = 9999
        template_cache = {}
        losses = training_step(batch, denoiser, diffusion, cfg, step=0,
                               device=device, obj_template_cache=template_cache)
        assert 'loss_inter'  not in losses
        assert 'loss_const'  not in losses
        assert 'loss_smooth' not in losses
        assert 'loss_ddpm'   in losses

    def test_aux_losses_after_curriculum(self, setup):
        diffusion, denoiser, batch, cfg, device, training_step = setup
        template_cache = {'1': _rand_verts(50)}
        losses = training_step(batch, denoiser, diffusion, cfg, step=0,
                               device=device, obj_template_cache=template_cache)
        assert 'loss_inter'  in losses
        assert 'loss_const'  in losses
        assert 'loss_smooth' in losses
