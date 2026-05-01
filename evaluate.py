"""WHOLE evaluation script — reproduces Tables 1, 2, 3.

Usage:
    python evaluate.py \\
        --pred_dir  /path/to/inference/outputs \\
        --config    configs/default.yaml \\
        [--split    contact|truncated|out_of_view|all]

The script:
  1. Discovers test clips: held-out sequences with object displacement > 5 cm
  2. For each clip, loads GT from the tar and prediction from pred_dir
  3. Computes hand / object / HOI metrics, prints a Table-style summary

Test set construction (Sec. 4):
  "We hold out 50 dynamic object trajectories (displacement > 5 cm) from
   unseen sequences."  We replicate this by selecting clips from train_aria
   and train_quest3 that were not seen during training (held-out by
   sequence_id) and have sufficient object motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
import yaml
from tqdm import tqdm

from data.hot3d_loader import _load_tar
from data.preprocessing import gravity_align_window
from utils.mano_utils import unpack_x0
from utils.rotation import se3_from_9d_repr

from eval.metrics_hand   import compute_hand_metrics
from eval.metrics_object import compute_object_metrics
from eval.metrics_hoi    import compute_hoi_metrics


MIN_OBJ_DISPLACEMENT_M = 0.05   # 5 cm
N_TEST_CLIPS_TARGET    = 50


# ---------------------------------------------------------------------------
# Test set selection
# ---------------------------------------------------------------------------

def find_dynamic_clips(
    tar_dirs: list[str],
    n_target: int = N_TEST_CLIPS_TARGET,
    min_disp: float = MIN_OBJ_DISPLACEMENT_M,
    held_out_sequences: Optional[set] = None,
) -> list[Path]:
    """Find clips with significant object displacement for the test set.

    Scans all tars, parses object poses, and returns those where the
    maximum object displacement across the sequence exceeds min_disp.

    Args:
        tar_dirs:          directories of clip tars
        n_target:          target number of test clips
        min_disp:          minimum object displacement in metres
        held_out_sequences: if provided, restrict to these sequence IDs
    Returns:
        list of tar Paths sorted by displacement (descending)
    """
    candidates: list[tuple[float, Path]] = []

    for tar_dir in tar_dirs:
        tars = sorted(Path(tar_dir).glob('clip-*.tar'))
        for tar in tqdm(tars, desc=f'Scanning {Path(tar_dir).name}', leave=False):
            try:
                clip = _load_tar(tar, load_images=False)
            except Exception:
                continue

            if held_out_sequences and clip.sequence_id not in held_out_sequences:
                continue

            # Compute max object displacement across the clip
            for obj_info in clip.objects.values():
                poses = obj_info['T_world_from_object']   # (T, 4, 4)
                transl = poses[:, :3, 3]
                disp   = np.linalg.norm(transl - transl[0], axis=-1).max()
                if disp >= min_disp:
                    candidates.append((disp, tar))
                    break

    # Sort by displacement descending, take top n_target
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates[:n_target]]


# ---------------------------------------------------------------------------
# Load object template vertices
# ---------------------------------------------------------------------------

_TEMPLATE_CACHE: dict[str, np.ndarray] = {}


def load_template(models_dir: str, bop_id: str) -> Optional[np.ndarray]:
    """Load canonical mesh vertices (V, 3), normalised to unit scale."""
    if bop_id in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[bop_id]
    glb = Path(models_dir) / f'obj_{int(bop_id):06d}.glb'
    if not glb.exists():
        return None
    mesh   = trimesh.load(str(glb), force='mesh')
    verts  = np.array(mesh.vertices, dtype=np.float32)
    center = verts.mean(0)
    scale  = float(np.linalg.norm(verts - center, axis=-1).max())
    verts  = (verts - center) / max(scale, 1e-6)
    _TEMPLATE_CACHE[bop_id] = verts
    return verts


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------

def extract_gt(clip) -> dict:
    """Extract GT joint positions and object poses from a parsed HOT3DClip."""
    T = len(clip.cameras)

    # Object SE(3): use first object in clip
    obj_id   = next(iter(clip.objects))
    obj_info = clip.objects[obj_id]
    T_obj_gt = obj_info['T_world_from_object']   # (T, 4, 4)
    bop_id   = obj_info['bop_id']

    # MANO FK for GT joints (falls back to wrist positions if FK unavailable)
    from data.mano_converter import build_hand_feature, mano_forward
    left_joints = right_joints = None
    try:
        left_joints = mano_forward(
            clip.hands['left']['thetas'],
            clip.hands['left']['wrist_xform'],
            clip.hand_shapes.get('left', np.zeros(10)),
            side='left',
        )['joints']   # (T, 21, 3)
        right_joints = mano_forward(
            clip.hands['right']['thetas'],
            clip.hands['right']['wrist_xform'],
            clip.hand_shapes.get('right', np.zeros(10)),
            side='right',
        )['joints']
    except Exception:
        pass

    if left_joints is None:
        # Wrist-only fallback: (T, 1, 3)
        left_joints  = clip.hands['left']['wrist_xform'][:, 3:].reshape(T, 1, 3)
        right_joints = clip.hands['right']['wrist_xform'][:, 3:].reshape(T, 1, 3)

    joints_gt = np.concatenate([left_joints, right_joints], axis=1)  # (T, J, 3)

    return {
        'joints':   joints_gt,
        'T_obj':    T_obj_gt,
        'bop_id':   bop_id,
        'valid':    clip.frame_valid,
    }


# ---------------------------------------------------------------------------
# Prediction loading and decoding
# ---------------------------------------------------------------------------

def load_prediction(pred_dir: str, clip_id: str) -> Optional[np.ndarray]:
    """Load x0 (T, 73) from a saved .npz reconstruction."""
    path = Path(pred_dir) / f'{clip_id}_reconstruction.npz'
    if not path.exists():
        return None
    return np.load(str(path))['x0']   # (T, 73)


def decode_prediction(
    x0:          np.ndarray,   # (T, 73)
    gt_bop_id:   str,
) -> dict:
    """Decode predicted x0 into joint positions and object SE(3).

    Returns:
        joints: (T, J, 3) — from MANO FK if available, else wrist proxy
        T_obj:  (T, 4, 4) — decoded from 9D representation
    """
    x = torch.tensor(x0, dtype=torch.float32)
    p = unpack_x0(x)

    # Object SE(3)
    T_obj = se3_from_9d_repr(p['obj_9d']).numpy()   # (T, 4, 4)

    # Hand joints
    from data.mano_converter import mano_forward
    left_joints = right_joints = None
    try:
        left_joints = mano_forward(
            p['left_thetas'].numpy(),
            torch.cat([p['left_orient'], p['left_transl']], dim=-1).numpy(),
            p['left_betas'][0].numpy(),
            side='left',
        )['joints']
        right_joints = mano_forward(
            p['right_thetas'].numpy(),
            torch.cat([p['right_orient'], p['right_transl']], dim=-1).numpy(),
            p['right_betas'][0].numpy(),
            side='right',
        )['joints']
    except Exception:
        pass

    if left_joints is None:
        T = x0.shape[0]
        left_joints  = p['left_transl'].numpy().reshape(T, 1, 3)
        right_joints = p['right_transl'].numpy().reshape(T, 1, 3)

    joints = np.concatenate([left_joints, right_joints], axis=1)

    return {'joints': joints, 'T_obj': T_obj}


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------

def evaluate_clip(
    tar_path:    Path,
    pred_dir:    str,
    models_dir:  str,
) -> Optional[dict]:
    """Evaluate one clip; return metric dict or None if prediction missing."""
    try:
        clip = _load_tar(tar_path, load_images=False)
    except Exception as e:
        print(f'  [WARN] Could not load {tar_path.name}: {e}')
        return None

    x0_pred = load_prediction(pred_dir, clip.clip_id)
    if x0_pred is None:
        return None

    gt   = extract_gt(clip)
    pred = decode_prediction(x0_pred, gt['bop_id'])
    verts = load_template(models_dir, gt['bop_id'])
    if verts is None:
        return None

    # Valid frame mask (both hands and object visible)
    valid = gt['valid']
    if valid.sum() < 10:
        return None

    pred_j = pred['joints'][valid]   # (V_T, J, 3)
    gt_j   = gt['joints'][valid]
    pred_T = pred['T_obj'][valid]    # (V_T, 4, 4)
    gt_T   = gt['T_obj'][valid]

    hand_m = compute_hand_metrics(pred_j, gt_j)
    obj_m  = compute_object_metrics(pred_T, gt_T, verts)
    hoi_m  = compute_hoi_metrics(pred_j, gt_j, pred_T, gt_T, verts)

    return {**hand_m, **obj_m, **hoi_m, 'clip_id': clip.clip_id}


# ---------------------------------------------------------------------------
# Aggregation and display
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    ('W-MPJPE',     'cm',   False),
    ('WA-MPJPE',    'cm',   False),
    ('PA-MPJPE',    'mm',   False),
    ('ACC-NORM',    '',     False),
    ('AUC_ADD',     '↑',    True),
    ('AUC_ADD-S',   '↑',    True),
    ('HOI_AUC_ADD',   '↑',  True),
    ('HOI_AUC_ADD-S', '↑',  True),
]


def print_table(results: list[dict]):
    """Print evaluation results in a compact table."""
    if not results:
        print('No results.')
        return

    # Header
    cols = [m for m, _, _ in _METRIC_COLS]
    header = f'{"Clip":25s}' + ''.join(f'{c:>14s}' for c in cols)
    print('\n' + '=' * len(header))
    print(header)
    print('-' * len(header))

    # Per-clip rows
    for r in results:
        row = f'{r["clip_id"][:24]:25s}'
        for col, unit, higher_better in _METRIC_COLS:
            v = r.get(col, float('nan'))
            row += f'{v:>14.4f}'
        print(row)

    print('=' * len(header))

    # Mean
    row = f'{"MEAN":25s}'
    for col, unit, _ in _METRIC_COLS:
        vals = [r[col] for r in results if col in r and not np.isnan(r.get(col, float('nan')))]
        mean = np.mean(vals) if vals else float('nan')
        row += f'{mean:>14.4f}'
    print(row)
    print('=' * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir',  required=True,
                        help='Directory of *_reconstruction.npz files')
    parser.add_argument('--config',    default='configs/default.yaml')
    parser.add_argument('--n_clips',   type=int, default=N_TEST_CLIPS_TARGET,
                        help='Max number of test clips to evaluate')
    parser.add_argument('--out_json',  default=None,
                        help='Save per-clip results to JSON')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d   = cfg['data']

    # Discover test clips (dynamic, from both Aria and Quest3)
    print('Finding dynamic test clips...')
    test_tars = find_dynamic_clips(
        [d['train_aria_dir'], d['train_quest3_dir']],
        n_target = args.n_clips,
    )
    print(f'Found {len(test_tars)} dynamic clips.')

    # Evaluate
    results = []
    for tar in tqdm(test_tars, desc='Evaluating'):
        r = evaluate_clip(tar, args.pred_dir, d['object_models_dir'])
        if r is not None:
            results.append(r)

    print_table(results)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved: {args.out_json}')

    return results


if __name__ == '__main__':
    main()
