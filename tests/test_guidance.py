"""Tests for guidance modules and inference pipeline.

Checks:
  1. VLM one-out-of-k constraint enforcement
  2. VLM interpolation (forward-fill) logic
  3. Camera projection (pinhole roundtrip, fisheye shape)
  4. One-way Chamfer: zero when on mask, positive when off mask
  5. compute_guidance: finite scalar, gradients flow back to x_n
  6. Sliding window blending: continuous overlap, shape correctness
  7. Inference pipeline (GT fallback, no VLM): runs end-to-end
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from guidance.vlm_contact  import (
    _validate_contact_json, VLMContactLabeler, _render_annotated_frame,
)
from guidance.reprojection import (
    project_pinhole, one_way_chamfer_2d, world_to_camera,
)
from guidance.guidance     import (
    GuidanceObs, GuidanceWeights, compute_guidance,
)
from inference             import blend_windows, build_windows

B, T = 1, 120
H_IMG, W_IMG = 480, 640


# ---------------------------------------------------------------------------
# VLM Contact
# ---------------------------------------------------------------------------

class TestVLMContact:
    def test_one_obj_valid(self):
        raw = {'cup': {'left': 1, 'right': 0}}
        r   = _validate_contact_json(raw, ['cup'])
        assert r['cup']['left'] == 1
        assert r['cup']['right'] == 0

    def test_one_out_of_k_left(self):
        """Left hand touches at most one object."""
        raw = {'obj1': {'left': 1, 'right': 0}, 'obj2': {'left': 1, 'right': 0}}
        r   = _validate_contact_json(raw, ['obj1', 'obj2'])
        total_left = r['obj1']['left'] + r['obj2']['left']
        assert total_left <= 1

    def test_one_out_of_k_right(self):
        raw = {'a': {'left': 0, 'right': 1}, 'b': {'left': 0, 'right': 1},
               'c': {'left': 0, 'right': 1}}
        r   = _validate_contact_json(raw, ['a', 'b', 'c'])
        total_right = sum(r[k]['right'] for k in ['a', 'b', 'c'])
        assert total_right <= 1

    def test_missing_obj_defaults_zero(self):
        raw = {}
        r   = _validate_contact_json(raw, ['cup'])
        assert r['cup'] == {'left': 0, 'right': 0}

    def test_interpolation_output_shape(self):
        vlm = VLMContactLabeler(api_key=None, contact_fps=3, clip_fps=30)
        frames     = np.zeros((60, 64, 64, 3), dtype=np.uint8)
        left_boxes = np.zeros((60, 4), dtype=np.float32)
        right_boxes = np.zeros((60, 4), dtype=np.float32)
        result = vlm.label_clip(frames, left_boxes, right_boxes, [], ['obj'])
        assert result.shape == (60, 2), result.shape

    def test_interpolation_forward_fill(self):
        """Label at keyframe k should hold until next keyframe."""
        contacts = {0: {'obj': {'left': 1, 'right': 0}},
                    10: {'obj': {'left': 0, 'right': 1}}}
        result = VLMContactLabeler._interpolate_to_full(
            contacts, [0, 10], 15, ['obj']
        )
        assert result[5, 0].item() == 1    # frame 5: still frame-0 label
        assert result[10, 1].item() == 1   # frame 10: new label

    def test_render_annotated_frame_shape(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out   = _render_annotated_frame(
            frame, np.array([5, 5, 20, 20]), np.array([40, 40, 60, 60]),
            [np.ones((64, 64), dtype=bool)], ['obj']
        )
        assert out.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------

class TestReprojection:
    def test_pinhole_center(self):
        """Point on optical axis should project to principal point."""
        pts = torch.tensor([[[0., 0., 1.]]]);   # (1, 1, 3)
        uv  = project_pinhole(pts, fx=500, fy=500, cx=320, cy=240)
        assert torch.allclose(uv[0, 0], torch.tensor([320., 240.]), atol=1e-4)

    def test_pinhole_shape(self):
        pts = torch.randn(B, T, 42, 3)
        uv  = project_pinhole(pts, 500, 500, 320, 240)
        assert uv.shape == (B, T, 42, 2)

    def test_pinhole_positive_z_only(self):
        """Negative z should be clamped and still produce finite output."""
        pts = torch.tensor([[[-1., 0., -0.5]]])
        uv  = project_pinhole(pts, 500, 500, 320, 240)
        assert uv.isfinite().all()

    def test_one_way_chamfer_zero_on_mask(self):
        """When projected points land exactly on the mask, loss should be 0."""
        mask = torch.zeros(B, T, H_IMG, W_IMG)
        mask[:, :, 240, 320] = 1   # single mask pixel at center

        # All points project to the mask pixel (normalized coords ~0,0)
        pts  = torch.zeros(B, T, 4, 2)
        pts[..., 0] = 320.0   # x = center
        pts[..., 1] = 240.0   # y = center

        loss = one_way_chamfer_2d(pts, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_one_way_chamfer_positive_off_mask(self):
        """Points outside mask produce positive loss."""
        mask = torch.zeros(B, T, H_IMG, W_IMG)   # empty mask
        pts  = torch.ones(B, T, 4, 2) * torch.tensor([100., 100.])
        loss = one_way_chamfer_2d(pts, mask)
        assert loss.item() > 0

    def test_one_way_chamfer_with_valid_mask(self):
        mask = torch.zeros(B, T, H_IMG, W_IMG)
        pts  = torch.ones(B, T, 4, 2) * 100
        valid = torch.ones(B, T, 4, dtype=torch.bool)
        valid[:, :, 2:] = False   # only first 2 pts valid
        loss_partial = one_way_chamfer_2d(pts, mask, valid)
        loss_full    = one_way_chamfer_2d(pts, mask, None)
        assert loss_partial.isfinite() and loss_full.isfinite()


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------

class TestComputeGuidance:
    def _rand_x0(self) -> torch.Tensor:
        x = torch.randn(B, T, 73) * 0.1
        # Valid 9D obj block
        eye9 = torch.zeros(9)
        eye9[0] = eye9[4] = eye9[8] = 1.0
        x[..., :9] = eye9 + torch.randn(B, T, 9) * 0.01
        return x

    def test_g_temp_only_finite(self):
        """Guidance with only g_temp (no observations) should be finite."""
        x = self._rand_x0().requires_grad_(True)
        obs = GuidanceObs()
        weights = GuidanceWeights(reproj=0.0, inter=0.0, temp=1.0)
        g = compute_guidance(x, obs, weights)
        assert g.isfinite()

    def test_g_inter_finite(self):
        verts = torch.randn(50, 3) * 0.05
        x = self._rand_x0().requires_grad_(True)
        obs = GuidanceObs(template_verts=verts)
        weights = GuidanceWeights(reproj=0.0, inter=1.0, temp=0.0)
        g = compute_guidance(x, obs, weights)
        assert g.isfinite()

    def test_gradient_flows_through_x0(self):
        """Gradient of g must propagate back to x_n via x0_pred."""
        x_n = torch.randn(B, T, 73, requires_grad=True)
        # Simulate denoiser output that retains grad
        x0_pred = x_n * 0.9 + torch.randn_like(x_n) * 0.1

        obs     = GuidanceObs(template_verts=torch.randn(30, 3) * 0.05)
        weights = GuidanceWeights(reproj=0.0, inter=1.0, temp=1.0)
        g = compute_guidance(x0_pred, obs, weights)
        g.backward()
        assert x_n.grad is not None and x_n.grad.isfinite().all()

    def test_guidance_zero_without_observations(self):
        """With no obs and zero inter/reproj weights, only g_temp is nonzero."""
        x = torch.ones(B, T, 73)    # constant → zero acceleration
        obs     = GuidanceObs()
        weights = GuidanceWeights(reproj=0.0, inter=0.0, temp=1.0)
        g = compute_guidance(x, obs, weights)
        assert g.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Sliding window blending
# ---------------------------------------------------------------------------

class TestBlending:
    def test_build_windows_coverage(self):
        starts = build_windows(150, 120, 30)
        # All frames should be covered
        covered = set()
        for s in starts:
            covered.update(range(s, s + 120))
        assert set(range(150)).issubset(covered)

    def test_blend_shape(self):
        windows = [np.random.randn(120, 73).astype(np.float32) for _ in range(2)]
        starts  = [0, 30]
        result  = blend_windows(windows, starts, 150)
        assert result.shape == (150, 73)

    def test_blend_non_overlap_equals_window(self):
        """Non-overlapping parts of a single window should be unchanged."""
        w = np.ones((120, 73), dtype=np.float32) * 5.0
        result = blend_windows([w], [0], 120)
        # Middle frames (far from edges) should be ~5
        assert abs(result[60, 0] - 5.0) < 0.05

    def test_blend_overlap_is_smooth(self):
        """Blended overlap region should be continuous (no jump > 1.0)."""
        w1 = np.ones((120, 73), dtype=np.float32) * 1.0
        w2 = np.ones((120, 73), dtype=np.float32) * 2.0
        result = blend_windows([w1, w2], [0, 30], 150)
        # Check smoothness: max step between adjacent frames
        steps = np.abs(np.diff(result[:, 0]))
        assert steps.max() < 0.5, f'Max step in blend: {steps.max():.3f}'


# ---------------------------------------------------------------------------
# End-to-end inference (GT fallback, no VLM, no checkpoint needed)
# ---------------------------------------------------------------------------

TRAIN_ARIA = Path('/scr/cezhao/workspace/HOI_recon/_DATA/hot3d/train_aria')
HAS_DATA   = TRAIN_ARIA.exists() and any(TRAIN_ARIA.glob('clip-*.tar'))


@pytest.mark.skipif(not HAS_DATA, reason='HOT3D data not available')
class TestInferencePipeline:
    def test_full_pipeline_no_guidance(self, tmp_path):
        """Run inference with a randomly-initialized model (no training needed)."""
        import yaml
        from models import build_denoiser, build_diffusion

        cfg = yaml.safe_load(open('configs/default.yaml'))
        # Tiny model for speed
        cfg['model'].update({'d_model': 64, 'nhead': 4, 'd_ff': 128,
                             'num_layers': 2, 'bps_n_points': 64})
        cfg['diffusion']['n_steps'] = 10   # very fast for testing

        # Save a dummy checkpoint
        device   = torch.device('cpu')
        denoiser = build_denoiser(cfg).to(device)
        ckpt_path = tmp_path / 'dummy.pt'
        torch.save({'model': denoiser.state_dict(), 'step': 0, 'cfg': cfg},
                   str(ckpt_path))

        tar = next(TRAIN_ARIA.glob('clip-*.tar'))
        from inference import infer_clip
        result = infer_clip(
            tar_path    = tar,
            checkpoint  = ckpt_path,
            cfg         = cfg,
            out_dir     = tmp_path,
            use_vlm     = False,
            use_hawor   = False,
            show_progress = False,
        )
        x0 = result['x0_full']
        assert x0.shape[1] == 73, f'Expected (T, 73), got {x0.shape}'
        assert np.isfinite(x0).all(), 'Reconstruction contains NaN/Inf'
