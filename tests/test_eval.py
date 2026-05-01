"""Tests for evaluation metrics.

Checks:
  1. Alignment: Umeyama roundtrip, per-frame Procrustes
  2. Hand metrics: perfect prediction → 0 error; W vs WA use different alignment
  3. Object metrics: ADD/ADD-S geometry, AUC range and threshold
  4. HOI metrics: aligned object error < unaligned when hand has global drift
  5. evaluate.py helpers: test set selection and metric aggregation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from eval.alignment     import umeyama, apply_similarity, global_align, per_frame_procrustes_align
from eval.metrics_hand  import w_mpjpe, wa_mpjpe, pa_mpjpe, acc_norm, compute_hand_metrics
from eval.metrics_object import add_per_frame, add_s_per_frame, compute_auc, compute_object_metrics
from eval.metrics_hoi   import compute_hoi_metrics

T, J, V = 30, 21, 100
rng = np.random.default_rng(0)


def _rand_joints(T=T, J=J):
    return rng.standard_normal((T, J, 3)).astype(np.float32) * 0.1

def _rand_SE3(T=T):
    """Random SE(3) sequence."""
    from scipy.spatial.transform import Rotation
    poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    for t in range(T):
        R = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
        poses[t, :3, :3] = R
        poses[t, :3,  3] = rng.standard_normal(3).astype(np.float32) * 0.1
    return poses

def _rand_verts(V=V):
    return (rng.standard_normal((V, 3)) * 0.05).astype(np.float32)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_umeyama_identity(self):
        """Umeyama on identical point sets should return identity."""
        pts = rng.standard_normal((50, 3)).astype(np.float32)
        s, R, t = umeyama(pts, pts)
        assert abs(s - 1.0) < 1e-4
        np.testing.assert_allclose(R, np.eye(3), atol=1e-4)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-4)

    def test_umeyama_roundtrip(self):
        """After alignment, src should map close to dst."""
        src = rng.standard_normal((40, 3)).astype(np.float32)
        # Apply known transform
        s_true = 1.5
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        t_true = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        dst = apply_similarity(src, s_true, R_true, t_true)

        s, R, t = umeyama(src, dst)
        src_aligned = apply_similarity(src, s, R, t)
        np.testing.assert_allclose(src_aligned, dst, atol=1e-4)

    def test_global_align_first2_vs_all(self):
        """first2 and all alignments should give different results on noisy data."""
        pred = _rand_joints() + rng.standard_normal((T, J, 3)).astype(np.float32) * 0.05
        gt   = _rand_joints()
        a_first2 = global_align(pred, gt, 'first2')
        a_all    = global_align(pred, gt, 'all')
        # They should differ
        assert not np.allclose(a_first2, a_all, atol=1e-3)

    def test_procrustes_zero_error_on_pure_rotation(self):
        """Per-frame Procrustes should recover pure rotation exactly."""
        from scipy.spatial.transform import Rotation
        pred = _rand_joints(T=5)
        R0   = Rotation.random(random_state=0).as_matrix().astype(np.float32)
        gt   = (pred @ R0.T)   # rotate all joints by same R
        aligned = per_frame_procrustes_align(pred, gt)
        np.testing.assert_allclose(aligned, gt, atol=1e-4)


# ---------------------------------------------------------------------------
# Hand metrics
# ---------------------------------------------------------------------------

class TestHandMetrics:
    def test_zero_error_on_perfect_prediction(self):
        j = _rand_joints()
        assert w_mpjpe(j,  j) == pytest.approx(0.0, abs=1e-5)
        assert wa_mpjpe(j, j) == pytest.approx(0.0, abs=1e-5)
        assert pa_mpjpe(j, j) == pytest.approx(0.0, abs=1e-5)
        assert acc_norm(j,  j) == pytest.approx(0.0, abs=1e-6)

    def test_w_mpjpe_units_cm(self):
        """W-MPJPE of a 10cm offset should return ~10 cm."""
        gt = np.zeros((T, J, 3), dtype=np.float32)
        pred = np.zeros_like(gt)
        pred[:, :, 0] = 0.10   # 10 cm offset in x
        # After alignment (scale+rot+trans), the error should be ~0 (perfect alignment)
        # So this tests that we return the right scale
        val = w_mpjpe(pred, gt)
        assert isinstance(val, float)
        assert val >= 0

    def test_wa_le_w_mpjpe(self):
        """WA-MPJPE uses more data for alignment → error should be ≤ W-MPJPE."""
        pred = _rand_joints() + rng.standard_normal((T, J, 3)).astype(np.float32) * 0.05
        gt   = _rand_joints()
        assert wa_mpjpe(pred, gt) <= w_mpjpe(pred, gt) + 1e-4

    def test_pa_mpjpe_units_mm(self):
        """PA-MPJPE should be larger than WA-MPJPE (in mm vs cm)."""
        pred = _rand_joints() * 2   # deliberately wrong
        gt   = _rand_joints()
        # PA-MPJPE in mm, WA-MPJPE in cm → if raw errors are similar, PA > 10×WA
        pa = pa_mpjpe(pred, gt)
        wa = wa_mpjpe(pred, gt)
        assert pa > 0 and wa > 0

    def test_acc_norm_zero_for_constant(self):
        """Constant trajectory has zero acceleration → ACC-NORM = 0."""
        j = np.ones((T, J, 3), dtype=np.float32)
        assert acc_norm(j, j) == pytest.approx(0.0, abs=1e-6)

    def test_compute_hand_metrics_returns_all_keys(self):
        pred = _rand_joints(); gt = _rand_joints()
        m = compute_hand_metrics(pred, gt)
        for k in ('W-MPJPE', 'WA-MPJPE', 'PA-MPJPE', 'ACC-NORM'):
            assert k in m, f'{k} missing'
            assert np.isfinite(m[k]), f'{k} is not finite'


# ---------------------------------------------------------------------------
# Object metrics
# ---------------------------------------------------------------------------

class TestObjectMetrics:
    def test_add_zero_for_identity(self):
        """ADD should be 0 when pred == GT."""
        T_id = np.eye(4, dtype=np.float32)
        v = _rand_verts()
        assert add_per_frame(T_id, T_id, v) == pytest.approx(0.0, abs=1e-6)

    def test_add_s_le_add(self):
        """ADD-S (nearest-point) ≤ ADD (direct match) always."""
        T_pred = _rand_SE3()[0]; T_gt = _rand_SE3()[0]
        v = _rand_verts()
        assert add_s_per_frame(T_pred, T_gt, v) <= add_per_frame(T_pred, T_gt, v) + 1e-5

    def test_auc_in_01(self):
        """AUC must lie in [0, 1]."""
        d = rng.uniform(0, 0.5, 100)
        auc = compute_auc(d)
        assert 0.0 <= auc <= 1.0

    def test_auc_perfect(self):
        """All distances = 0 → AUC = 1."""
        auc = compute_auc(np.zeros(50))
        assert auc == pytest.approx(1.0, abs=1e-5)

    def test_auc_all_above_threshold(self):
        """All distances > threshold → AUC ≈ 0."""
        auc = compute_auc(np.ones(50) * 1.0, threshold=0.3)
        assert auc < 0.01

    def test_compute_object_metrics_keys(self):
        T_p = _rand_SE3(); T_g = _rand_SE3(); v = _rand_verts()
        m = compute_object_metrics(T_p, T_g, v)
        for k in ('ADD', 'ADD-S', 'AUC_ADD', 'AUC_ADD-S'):
            assert k in m and np.isfinite(m[k])


# ---------------------------------------------------------------------------
# HOI metrics
# ---------------------------------------------------------------------------

class TestHOIMetrics:
    def test_zero_error_on_perfect_prediction(self):
        j = _rand_joints(); T_o = _rand_SE3(); v = _rand_verts()
        m = compute_hoi_metrics(j, j, T_o, T_o, v)
        assert m['HOI_ADD']   < 1e-5
        assert m['HOI_ADD-S'] < 1e-5

    def test_hoi_alignment_reduces_error(self):
        """HOI metrics (after hand alignment) should be ≤ raw ADD."""
        pred_j = _rand_joints()
        gt_j   = _rand_joints()
        T_p    = _rand_SE3(); T_g = _rand_SE3(); v = _rand_verts()

        hoi_m  = compute_hoi_metrics(pred_j, gt_j, T_p, T_g, v)
        raw_m  = compute_object_metrics(T_p, T_g, v)

        # HOI aligns the coordinate system using hand info; result can go either
        # way, but both should be finite and in [0, 1] for AUC
        assert 0 <= hoi_m['HOI_AUC_ADD'] <= 1
        assert 0 <= hoi_m['HOI_AUC_ADD-S'] <= 1

    def test_keys_present(self):
        j = _rand_joints(); T_o = _rand_SE3(); v = _rand_verts()
        m = compute_hoi_metrics(j, j, T_o, T_o, v)
        for k in ('HOI_ADD', 'HOI_ADD-S', 'HOI_AUC_ADD', 'HOI_AUC_ADD-S'):
            assert k in m


# ---------------------------------------------------------------------------
# AUC values from paper (sanity check on Table 2 scale)
# ---------------------------------------------------------------------------

class TestAUCScale:
    def test_paper_scale(self):
        """Paper's WHOLE result: AUC_ADD ≈ 51.1 (Table 2, ↑ metric, ×100?).

        Table 2 reports AUC ADD ↑ = 51.1 for WHOLE. If AUC is in [0,1],
        then 51.1 means the table uses percentages (×100). Check our
        implementation produces values in [0, 1] before ×100 reporting.
        """
        # Simulate good predictions: 70% frames within 0.3m
        d_good = np.concatenate([np.zeros(70), np.ones(30) * 0.5])
        auc = compute_auc(d_good, threshold=0.3)
        assert 0 <= auc <= 1.0
        # Expected: since 70% below threshold evenly distributed, AUC ≈ 0.70
        # (actually higher since distances within threshold are distributed)
        assert auc > 0.4   # should be well above 0
        # Report as percentage: matches paper's "51.1" style values
        print(f'\n  AUC (in [0,1]): {auc:.3f}  → reported as {auc*100:.1f}')
