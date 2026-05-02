"""Pre-process all HOT3D clips into cached .npz files for fast training.

Reads each clip tar once, runs gravity alignment + MANO FK + proximity
contact labels, and saves all arrays to a single .npz per clip.
Training then does `np.load()` instead of reading tars + parsing JSON.

Expected speedup: ~8-10× (from ~2.5s/step → ~0.3s/step).

Usage:
    python scripts/preprocess_cache.py --tar_dir .../train_aria  --out_dir .../cache/train_aria
    python scripts/preprocess_cache.py --tar_dir .../train_quest3 --out_dir .../cache/train_quest3

Runtime: ~30-60 min per split (parallelised with --workers).
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def process_clip(args):
    tar_path, out_dir = args
    out_path = out_dir / (tar_path.stem + '.npz')
    if out_path.exists():
        return str(tar_path.stem), 'skip'

    try:
        from data.hot3d_loader import _load_tar
        from data.preprocessing import gravity_align_window, _compute_proximity_contact
        from data.mano_converter import mano_forward, build_hand_feature
        from utils.rotation import se3_to_9d_repr
        import torch

        clip = _load_tar(tar_path, load_images=False)
        T    = len(clip.cameras)
        obj_id   = next(iter(clip.objects))
        obj_info = clip.objects[obj_id]

        # Build window_dict
        window_dict = {
            'obj_T_world':           obj_info['T_world_from_object'],  # (T,4,4)
            'left_thetas':           clip.hands['left']['thetas'],
            'left_wrist':            clip.hands['left']['wrist_xform'],
            'left_betas':            clip.hand_shapes.get('left',  np.zeros(10, np.float32)),
            'left_valid':            clip.hands['left']['valid'],
            'right_thetas':          clip.hands['right']['thetas'],
            'right_wrist':           clip.hands['right']['wrist_xform'],
            'right_betas':           clip.hand_shapes.get('right', np.zeros(10, np.float32)),
            'right_valid':           clip.hands['right']['valid'],
            'frame_valid':           clip.frame_valid,
            'T_world_from_ref_cam0': (clip.cameras[0]['214-1']['T_world_from_camera']
                                      if clip.device == 'Aria'
                                      else clip.cameras[0]['1201-1']['T_world_from_camera']),
            'clip_id':  clip.clip_id,
            'device':   clip.device,
            'obj_name': obj_info['name'],
            'obj_bop_id': obj_info['bop_id'],
        }

        # Gravity alignment
        from data.preprocessing import gravity_align_window
        window_dict = gravity_align_window(window_dict)

        # Proximity contact labels
        contact = _compute_proximity_contact(window_dict, threshold_m=0.30)

        # 9D object repr
        obj_T  = torch.tensor(window_dict['obj_T_world'], dtype=torch.float32)
        obj_9d = se3_to_9d_repr(obj_T).numpy()

        # Hand features
        left_feat  = build_hand_feature(
            window_dict['left_thetas'],
            window_dict['left_wrist'],
            window_dict['left_betas'],
        )
        right_feat = build_hand_feature(
            window_dict['right_thetas'],
            window_dict['right_wrist'],
            window_dict['right_betas'],
        )

        # x0: (T, 73)
        x0 = np.concatenate([obj_9d, contact, left_feat, right_feat], axis=-1).astype(np.float32)

        # MANO FK for GT joint positions (used by L_const)
        try:
            left_joints = mano_forward(
                window_dict['left_thetas'],
                window_dict['left_wrist'],
                window_dict['left_betas'],
                side='left',
            )['joints'].astype(np.float32)   # (T, 21, 3)
            right_joints = mano_forward(
                window_dict['right_thetas'],
                window_dict['right_wrist'],
                window_dict['right_betas'],
                side='right',
            )['joints'].astype(np.float32)
        except Exception:
            left_joints  = window_dict['left_wrist'][:, 3:].reshape(T, 1, 3)
            right_joints = window_dict['right_wrist'][:, 3:].reshape(T, 1, 3)
            left_joints  = np.broadcast_to(left_joints,  (T, 21, 3)).copy().astype(np.float32)
            right_joints = np.broadcast_to(right_joints, (T, 21, 3)).copy().astype(np.float32)

        np.savez_compressed(
            str(out_path),
            x0          = x0,
            H_left      = left_feat.astype(np.float32),
            H_right     = right_feat.astype(np.float32),
            left_joints = left_joints,
            right_joints= right_joints,
            obj_T_world = window_dict['obj_T_world'].astype(np.float32),
            frame_valid = window_dict['frame_valid'],
            obj_bop_id  = np.array([obj_info['bop_id']], dtype=object),
            R_gravity   = window_dict['R_gravity_align'].astype(np.float32),
        )
        return str(tar_path.stem), 'ok'

    except Exception as e:
        return str(tar_path.stem), f'ERROR: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    tar_dir = Path(args.tar_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tars = sorted(tar_dir.glob('clip-*.tar'))
    print(f'Processing {len(tars)} clips → {out_dir}  (workers={args.workers})')

    tasks = [(t, out_dir) for t in tars]

    ok = skip = err = 0
    with mp.Pool(args.workers) as pool:
        for clip_id, status in tqdm(pool.imap_unordered(process_clip, tasks),
                                     total=len(tasks)):
            if status == 'ok':   ok   += 1
            elif status == 'skip': skip += 1
            else:                err  += 1
            if 'ERROR' in status:
                tqdm.write(f'  {clip_id}: {status}')

    print(f'\nDone: {ok} processed, {skip} skipped, {err} errors')
    print(f'Cache: {out_dir}  ({sum(1 for _ in out_dir.glob("*.npz"))} files)')


if __name__ == '__main__':
    main()
