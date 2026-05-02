"""Fast training dataset backed by pre-computed .npz cache files.

Replaces HOT3DDataset for training: instead of reading tar files and
running FK per step, just does np.load() + noise injection.

Each .npz (one per clip) contains:
    x0:           (T, 73)     ground-truth diffusion variable
    H_left/right: (T, 31)     clean hand features
    left/right_joints: (T,21,3) FK-computed joint positions
    obj_T_world:  (T, 4, 4)   object SE(3)
    frame_valid:  (T,)        bool validity mask
    obj_bop_id:   (1,)        str BOP ID for BPS lookup
    R_gravity:    (3, 3)      gravity alignment rotation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.bps import ObjectBPSCache
from data.preprocessing import inject_hand_noise


WINDOW_LEN = 120
STRIDE     = 30


class CachedHOT3DDataset(Dataset):
    """Dataset backed by pre-computed .npz files.

    Args:
        cache_dirs:     list of directories containing clip-XXXXXX.npz files
        object_models_dir: path to HOT3D object_models/*.glb
        bps_n_points:   BPS resolution
        bps_radius:     BPS sphere radius
        window_len:     frames per window (default 120)
        stride:         sliding window stride (default 30)
        augment:        apply hand noise injection
        max_clips:      cap dataset size (for debugging)
    """

    def __init__(
        self,
        cache_dirs:        list[str],
        object_models_dir: str,
        bps_n_points:      int   = 1024,
        bps_radius:        float = 0.2,
        window_len:        int   = WINDOW_LEN,
        stride:            int   = STRIDE,
        augment:           bool  = True,
        max_clips:         int | None = None,
    ):
        self.window_len = window_len
        self.stride     = stride
        self.augment    = augment
        self.bps_cache  = ObjectBPSCache(object_models_dir, bps_n_points, bps_radius)

        # Index: list of (npz_path, start_frame)
        self._index: list[tuple[Path, int]] = []
        n_clips = 0
        for d in cache_dirs:
            npzs = sorted(Path(d).glob('clip-*.npz'))
            for npz in npzs:
                if max_clips and n_clips >= max_clips:
                    break
                # Each 150-frame clip yields a few windows
                clip_len = 150  # HOT3D clips are always 150 frames
                n_win = max(1, (clip_len - window_len) // stride + 1)
                for i in range(n_win):
                    self._index.append((npz, i * stride))
                n_clips += 1

        # Pre-load all npz files into RAM (351 MB compressed → ~1 GB decompressed).
        # Eliminates all disk I/O during training, removing the DataLoader bottleneck.
        unique_npzs = sorted({str(p) for p, _ in self._index})
        self._cache: dict[str, dict] = {}
        self._cache_max = len(unique_npzs) + 1   # never evict
        for path in unique_npzs:
            raw = np.load(path, allow_pickle=True)
            self._cache[path] = dict(raw)

    def __len__(self) -> int:
        return len(self._index)

    def _load_npz(self, path: Path) -> dict:
        key = str(path)
        if key not in self._cache:
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            raw = np.load(str(path), allow_pickle=True)
            self._cache[key] = dict(raw)
        return self._cache[key]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        npz_path, start = self._index[idx]
        end = start + self.window_len
        d   = self._load_npz(npz_path)

        x0          = d['x0'][start:end]            # (T, 73)
        H_left      = d['H_left'][start:end]        # (T, 31)
        H_right     = d['H_right'][start:end]       # (T, 31)
        frame_valid = d['frame_valid'][start:end]   # (T,)
        obj_bop_id  = str(d['obj_bop_id'][0])

        # Noise injection to produce H_tilde
        if self.augment:
            tmp = {
                'left_thetas':  x0[:, 17:32],   # thetas slice in 31D block
                'left_wrist':   x0[:, 11:17],
                'left_betas':   x0[0, 32:42],
                'right_thetas': x0[:, 48:63],
                'right_wrist':  x0[:, 42:48],
                'right_betas':  x0[0, 63:73],
            }
            inject_hand_noise(tmp)
            H_left  = np.concatenate([
                tmp['left_wrist_noisy'], tmp['left_thetas_noisy'],
                np.tile(tmp['left_betas'], (self.window_len, 1))
            ], axis=-1)
            H_right = np.concatenate([
                tmp['right_wrist_noisy'], tmp['right_thetas_noisy'],
                np.tile(tmp['right_betas'], (self.window_len, 1))
            ], axis=-1)

        H_tilde = np.concatenate([H_left, H_right], axis=-1).astype(np.float32)

        # BPS descriptor
        O = self.bps_cache.get(obj_bop_id)  # (K,)

        # Pre-computed FK joints (for L_const)
        left_joints  = d.get('left_joints',  np.zeros((self.window_len, 21, 3), np.float32))
        right_joints = d.get('right_joints', np.zeros((self.window_len, 21, 3), np.float32))
        if left_joints.shape[0] > self.window_len:
            left_joints  = left_joints[start:end]
            right_joints = right_joints[start:end]

        return {
            'x0':           torch.tensor(x0,          dtype=torch.float32),
            'H_tilde':      torch.tensor(H_tilde,     dtype=torch.float32),
            'O':            O,
            'frame_valid':  torch.tensor(frame_valid, dtype=torch.bool),
            'left_joints':  torch.tensor(left_joints, dtype=torch.float32),
            'right_joints': torch.tensor(right_joints,dtype=torch.float32),
            'obj_bop_id':   obj_bop_id,
        }
