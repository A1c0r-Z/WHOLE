"""HaWoR wrapper for world-grounded hand motion estimation.

HaWoR [Zhang et al., CVPR 2025] estimates world-space MANO trajectories
from a monocular egocentric video.  It lives in a separate conda environment
(hawor_h200) so we call it via subprocess and exchange results through a
temporary .npz file.

At test time this provides the noisy hand conditioning H̃ (B=1, T, 62).
For development / ablation, a GT-based fallback is provided.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data.mano_converter import build_hand_feature


HAWOR_ENV_PYTHON = (
    '/scr/cezhao/workspace/HOI_recon/.conda_envs/hawor_h200/bin/python'
)
HAWOR_ROOT = '/scr/cezhao/workspace/HOI_recon'


class HaWoRWrapper:
    """Run HaWoR on a clip and return H̃ (1, T, 62).

    Args:
        use_gt_fallback: If True and HaWoR fails (or for development), use
                         GT hand params from the HOT3DClip with added noise.
                         Set to False for real inference.
        noise_std: Noise scale for GT fallback (mimics inaccurate estimation).
    """

    def __init__(
        self,
        use_gt_fallback: bool = True,
        noise_std: float = 0.05,
    ):
        self.use_gt_fallback = use_gt_fallback
        self.noise_std = noise_std
        self._hawor_available = self._check_hawor()

    def _check_hawor(self) -> bool:
        try:
            result = subprocess.run(
                [HAWOR_ENV_PYTHON, '-c', 'import hawor; print("ok")'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0 and 'ok' in result.stdout
        except Exception:
            return False

    def estimate(
        self,
        frames:          np.ndarray,      # (T, H, W, 3) uint8 RGB
        slam_poses:      np.ndarray,      # (T, 4, 4) T_world_from_cam
        clip_id:         str = '',
        gt_left_thetas:  Optional[np.ndarray] = None,   # (T, 15) for fallback
        gt_left_wrist:   Optional[np.ndarray] = None,   # (T, 6)
        gt_left_betas:   Optional[np.ndarray] = None,   # (10,)
        gt_right_thetas: Optional[np.ndarray] = None,
        gt_right_wrist:  Optional[np.ndarray] = None,
        gt_right_betas:  Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Estimate noisy hand conditioning H̃.

        Returns:
            (1, T, 62) float32 tensor — H̃ for one clip
        """
        if self._hawor_available and not self.use_gt_fallback:
            return self._run_hawor(frames, slam_poses, clip_id)

        if gt_left_thetas is not None:
            return self._gt_fallback(
                gt_left_thetas, gt_left_wrist, gt_left_betas,
                gt_right_thetas, gt_right_wrist, gt_right_betas,
            )

        # No information — return zeros
        T = frames.shape[0] if frames is not None else slam_poses.shape[0]
        return torch.zeros(1, T, 62)

    def _run_hawor(
        self,
        frames:     np.ndarray,
        slam_poses: np.ndarray,
        clip_id:    str,
    ) -> torch.Tensor:
        """Run HaWoR via subprocess; exchange data through a temp .npz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_npz  = Path(tmpdir) / 'input.npz'
            output_npz = Path(tmpdir) / 'output.npz'

            np.savez(str(input_npz),
                     frames=frames, slam_poses=slam_poses, clip_id=clip_id)

            script = Path(__file__).parent / '_hawor_runner.py'
            result = subprocess.run(
                [HAWOR_ENV_PYTHON, str(script),
                 '--input', str(input_npz),
                 '--output', str(output_npz),
                 '--hawor_root', HAWOR_ROOT],
                capture_output=True, text=True, timeout=300,
            )

            if result.returncode != 0 or not output_npz.exists():
                import warnings
                warnings.warn(
                    f'HaWoR failed for {clip_id}: {result.stderr[:200]}. '
                    f'Returning zeros.'
                )
                T = slam_poses.shape[0]
                return torch.zeros(1, T, 62)

            out = np.load(str(output_npz))
            H_tilde = torch.tensor(out['H_tilde'], dtype=torch.float32)
            return H_tilde.unsqueeze(0)   # (1, T, 62)

    def _gt_fallback(
        self,
        left_thetas, left_wrist, left_betas,
        right_thetas, right_wrist, right_betas,
    ) -> torch.Tensor:
        """Perturb GT params to simulate inaccurate estimation."""
        rng = np.random.default_rng()

        def _perturb(arr, std):
            if arr is None:
                return arr
            return arr + rng.normal(0, std, arr.shape).astype(np.float32)

        T = left_thetas.shape[0] if left_thetas is not None else 1
        std = self.noise_std

        left_feat = build_hand_feature(
            _perturb(left_thetas,  std) if left_thetas  is not None else np.zeros((T,15), np.float32),
            _perturb(left_wrist,   std) if left_wrist   is not None else np.zeros((T, 6), np.float32),
            left_betas  if left_betas  is not None else np.zeros(10, np.float32),
        )   # (T, 31)

        right_feat = build_hand_feature(
            _perturb(right_thetas, std) if right_thetas is not None else np.zeros((T,15), np.float32),
            _perturb(right_wrist,  std) if right_wrist  is not None else np.zeros((T, 6), np.float32),
            right_betas if right_betas is not None else np.zeros(10, np.float32),
        )   # (T, 31)

        H_tilde = np.concatenate([left_feat, right_feat], axis=-1)  # (T, 62)
        return torch.tensor(H_tilde, dtype=torch.float32).unsqueeze(0)  # (1, T, 62)
