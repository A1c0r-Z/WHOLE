"""Smoke-test the data pipeline on a handful of clips.

Usage:
    python scripts/verify_data.py --tar_dir /path/to/hot3d/train_aria
    python scripts/verify_data.py --tar_dir /path/to/hot3d/train_aria --no_images --max_clips 5
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.hot3d_loader import HOT3DDataset, collate_fn
from data.preprocessing import preprocess_window
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tar_dir',   type=str,
                   default='/scr/cezhao/workspace/HOI_recon/_DATA/hot3d/train_aria')
    p.add_argument('--max_clips', type=int, default=3)
    p.add_argument('--no_images', action='store_true')
    p.add_argument('--out_dir',   type=str, default='/tmp/whole_verify')
    return p.parse_args()


def print_shapes(d: dict, indent: int = 2):
    pad = ' ' * indent
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f'{pad}{k}: {v.shape}  dtype={v.dtype}')
        elif isinstance(v, str):
            print(f'{pad}{k}: "{v}"')
        else:
            print(f'{pad}{k}: {type(v).__name__}')


def visualize_window(processed: dict, save_path: Path):
    """Save a simple 3D scatter of hand wrist + object translations over time."""
    x0 = processed['x0']       # (T, 73)
    T  = x0.shape[0]
    t  = np.arange(T)

    # Object translation: x0[..., 6:9] (last 3 of 9D repr = translation)
    obj_t  = x0[:, 6:9]        # (T, 3)
    # Left hand translation: x0[..., 11:14] (offset 9+2 = 11, transl at +3)
    left_t = x0[:, 12:15]      # (T, 3)  [9 obj + 2 contact + 3 orient + start]
    # Right hand translation: left block = 9+2+31=42, transl at +3
    right_t = x0[:, 45:48]     # (T, 3)

    fig = plt.figure(figsize=(10, 5))

    # 3D trajectory plot
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(*obj_t.T,   label='object',     color='steelblue')
    ax.plot(*left_t.T,  label='left wrist',  color='coral')
    ax.plot(*right_t.T, label='right wrist', color='seagreen')
    ax.scatter(*obj_t[0],   s=50, color='steelblue', marker='o')
    ax.scatter(*left_t[0],  s=50, color='coral',     marker='o')
    ax.scatter(*right_t[0], s=50, color='seagreen',  marker='o')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"Gravity-aligned trajectories\n{processed['clip_id']}")
    ax.legend(fontsize=8)

    # Per-axis time series
    ax2 = fig.add_subplot(122)
    ax2.plot(t, obj_t[:, 2],   label='obj z',        color='steelblue')
    ax2.plot(t, left_t[:, 2],  label='left wrist z',  color='coral')
    ax2.plot(t, right_t[:, 2], label='right wrist z', color='seagreen')
    ax2.set_xlabel('frame'); ax2.set_ylabel('z (m)')
    ax2.set_title('Height over time (gravity-aligned)')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'  Saved: {save_path}')


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    print('=== HOT3D Data Pipeline Verification ===')
    print(f'tar_dir  : {args.tar_dir}')
    print(f'max_clips: {args.max_clips}')
    print()

    # ---- 1. Dataset construction ----
    t0 = time.time()
    ds = HOT3DDataset(
        tar_dir    = args.tar_dir,
        max_clips  = args.max_clips,
        load_images= not args.no_images,
        stride     = 30,
    )
    print(f'Dataset: {len(ds)} windows from {args.max_clips} clips  ({time.time()-t0:.1f}s)')

    # ---- 2. Single-item access ----
    print('\n--- Raw window dict ---')
    t0 = time.time()
    item = ds[0]
    print(f'Load time: {time.time()-t0:.2f}s')
    print_shapes(item)

    # ---- 3. Preprocessing ----
    print('\n--- After preprocess_window ---')
    processed = preprocess_window(item, augment=True)
    print_shapes(processed)

    x0      = processed['x0']
    H_tilde = processed['H_tilde']
    valid   = processed['frame_valid']

    assert x0.shape      == (120, 73), f"x0 shape mismatch: {x0.shape}"
    assert H_tilde.shape == (120, 62), f"H_tilde shape mismatch: {H_tilde.shape}"
    assert valid.shape   == (120,),    f"frame_valid shape mismatch: {valid.shape}"

    print(f'\n  x0 range:       [{x0.min():.3f}, {x0.max():.3f}]')
    print(f'  H_tilde range:  [{H_tilde.min():.3f}, {H_tilde.max():.3f}]')
    print(f'  valid frames:   {valid.sum()} / {len(valid)}')

    # ---- 4. Visualization ----
    visualize_window(processed, out_dir / f'{item["clip_id"]}_trajectories.png')

    # ---- 5. DataLoader throughput ----
    print('\n--- DataLoader throughput (batch_size=4, num_workers=2) ---')
    loader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate_fn)
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i == 0:
            print(f'  First batch keys: {list(batch.keys())}')
            print(f'  left_thetas shape: {batch["left_thetas"].shape}')
        if i >= 3:
            break
    dt = time.time() - t0
    print(f'  4 batches in {dt:.1f}s  (~{4*4/dt:.1f} windows/s)')

    print('\n=== All checks passed ===')


if __name__ == '__main__':
    main()
