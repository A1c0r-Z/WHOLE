"""One-time conversion of MANO .pkl files to a chumpy-free .npz format.

Run this script once in an environment that has chumpy (e.g. hawor_h200):

    /scr/cezhao/workspace/HOI_recon/.conda_envs/hawor_h200/bin/python \
        scripts/convert_mano_pkl.py \
        --mano_dir /scr/cezhao/workspace/HOI_recon/hamer/_DATA/data/mano \
        --out_dir   /scr/cezhao/workspace/HOI_recon/_DATA/mano

The output .npz files are loadable with plain numpy/torch and have no
chumpy dependency, making them compatible with Python 3.11+.

Extracted arrays (both left and right hands):
    hand_components   (45, 15)  PCA basis: full_pose = betas @ hand_components + hands_mean
    hands_mean        (45,)     Mean hand pose in axis-angle space
    shapedirs         (778, 3, 10)  Shape blend shapes
    posedirs          (778, 3, 135) Pose blend shapes
    v_template        (778, 3)  Template mesh vertices
    J_regressor       (16, 778) Joint regressor
    kintree_table     (2, 16)   Kinematic tree
    faces             (1538, 3) Triangle faces
"""

import argparse
from pathlib import Path

import numpy as np


def convert_mano_pkl(pkl_path: Path, out_path: Path):
    """Load a MANO pkl (requires chumpy) and save numpy arrays to .npz."""
    import smplx, torch

    is_rhand = 'RIGHT' in pkl_path.name.upper()
    print(f"  Loading {'right' if is_rhand else 'left'} hand: {pkl_path}")

    layer = smplx.create(
        str(pkl_path),
        model_type='mano',
        use_pca=True,
        num_pca_comps=15,
        is_rhand=is_rhand,
        flat_hand_mean=True,
        batch_size=1,
    )

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    data = {
        # (15, 45): rows = PCA components, cols = pose dims
        # full_pose (45D) = thetas (15D) @ hand_components + hand_mean
        'hand_components': _to_np(layer.hand_components),
        'hand_mean':       _to_np(layer.hand_mean),
        'pose_mean':       _to_np(layer.pose_mean),
        'shapedirs':       _to_np(layer.shapedirs),
        'posedirs':        _to_np(layer.posedirs),
        'v_template':      _to_np(layer.v_template),
        'J_regressor':     _to_np(layer.J_regressor),
        'parents':         _to_np(layer.parents),
        'lbs_weights':     _to_np(layer.lbs_weights),
        'faces':           _to_np(layer.faces_tensor),
        'is_rhand':        np.array([is_rhand]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **data)
    print(f"  Saved: {out_path}")
    for k, v in data.items():
        print(f"    {k}: {v.shape} {v.dtype}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mano_dir', type=str,
                   default='/scr/cezhao/workspace/HOI_recon/hamer/_DATA/data/mano')
    p.add_argument('--out_dir',  type=str,
                   default='/scr/cezhao/workspace/HOI_recon/_DATA/mano')
    args = p.parse_args()

    mano_dir = Path(args.mano_dir)
    out_dir  = Path(args.out_dir)

    for side in ('RIGHT', 'LEFT'):
        pkl = mano_dir / f'MANO_{side}.pkl'
        if not pkl.exists():
            print(f"  Skipping {pkl} (not found)")
            continue
        convert_mano_pkl(pkl, out_dir / f'MANO_{side}.npz')

    print("\nDone. Re-run the main pipeline with MANO_DIR pointing to the .npz files.")


if __name__ == '__main__':
    main()
