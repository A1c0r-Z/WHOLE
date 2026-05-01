"""HOT3D-Clip dataset loader.

Reads clip tar files from bop-benchmark/hot3d and returns fixed-length
windows of (hands, objects, cameras, images) ready for downstream processing.

Tar file structure per clip:
    __hand_shapes.json__        MANO beta shape params (clip-level)
    {idx:06d}.cameras.json      Camera SLAM poses and calibration (per-frame)
    {idx:06d}.hands.json        MANO pose: thetas (15D) + wrist_xform (6D) (per-frame)
    {idx:06d}.objects.json      Object SE(3) poses keyed by object_id (per-frame)
    {idx:06d}.image_214-1.jpg   Main RGB camera (Aria: 1408x1408, Quest3: varies)
    {idx:06d}.image_1201-1.jpg  SLAM camera left (640x480)
    {idx:06d}.image_1201-2.jpg  SLAM camera right (640x480)
    {idx:06d}.info.json         Metadata: device, sequence_id, timestamps
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Camera stream IDs present in each device type
ARIA_CAMS   = ('214-1', '1201-1', '1201-2')
QUEST3_CAMS = ('1201-1', '1201-2')

# HOT3D clips have 150 frames; we train on windows of T=120
CLIP_LEN    = 150
WINDOW_LEN  = 120


class HOT3DClip:
    """In-memory representation of one parsed clip window."""

    __slots__ = (
        'clip_id', 'device', 'sequence_id',
        'cameras',        # list[dict]  length T, keys: cam_id -> {T_world_from_camera, calibration}
        'hands',          # dict: 'left'/'right' -> {thetas (T,15), wrist_xform (T,6), boxes (T,4)}
        'hand_shapes',    # dict: 'left'/'right' -> ndarray (10,)
        'objects',        # dict: obj_id -> {T_world_from_object (T,4,4), name, bop_id}
        'images',         # dict: cam_id -> ndarray (T,H,W,3)  or None if load_images=False
        'frame_valid',    # ndarray bool (T,) — False for occluded/truncated frames
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _parse_se3(d: dict) -> np.ndarray:
    """Parse HOT3D SE(3) dict {quaternion_wxyz, translation_xyz} into 4x4 numpy array."""
    from utils.rotation import quat_wxyz_to_matrix_np
    q = np.array(d['quaternion_wxyz'], dtype=np.float32)
    t = np.array(d['translation_xyz'], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_matrix_np(q)
    T[:3, 3]  = t
    return T


def _load_tar(tar_path: str | Path, load_images: bool = True) -> HOT3DClip:
    """Parse all frames of a clip tar into a HOT3DClip object.

    Args:
        tar_path:    Path to the .tar file.
        load_images: Whether to decode JPEG images (slow; disable for faster debugging).
    Returns:
        HOT3DClip with all frames (length = CLIP_LEN).
    """
    tar_path = Path(tar_path)
    clip_id  = tar_path.stem  # e.g. 'clip-001849'

    members: dict[str, bytes] = {}
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            members[member.name] = tf.extractfile(member).read()

    # ---- clip-level hand shapes ----
    shapes_raw = json.loads(members['__hand_shapes.json__'])
    hand_shapes = {
        side: np.array(shapes_raw[side]['mano'], dtype=np.float32)
        for side in ('left', 'right')
        if side in shapes_raw
    }

    # ---- discover frame count from keys ----
    frame_indices = sorted({
        int(name.split('.')[0])
        for name in members
        if name[0].isdigit()
    })
    T = len(frame_indices)

    # ---- per-frame parsing ----
    cameras: list[dict] = []
    hands_thetas   = {s: [] for s in ('left', 'right')}
    hands_wrist    = {s: [] for s in ('left', 'right')}
    hands_boxes    = {s: [] for s in ('left', 'right')}  # amodal 2D boxes, main cam
    hands_valid    = {s: [] for s in ('left', 'right')}
    objects_raw: dict[str, list[np.ndarray]] = {}
    objects_meta: dict[str, dict] = {}
    images_raw:   dict[str, list] = {}
    device = 'Aria'
    sequence_id = ''

    for idx in frame_indices:
        prefix = f'{idx:06d}'

        # info
        info = json.loads(members[f'{prefix}.info.json'])
        device      = info['device']
        sequence_id = info['sequence_id']

        # cameras
        cam_data = json.loads(members[f'{prefix}.cameras.json'])
        frame_cams = {}
        for cam_id, cam_info in cam_data.items():
            frame_cams[cam_id] = {
                'T_world_from_camera': _parse_se3(cam_info['T_world_from_camera']),
                'calibration': cam_info.get('calibration', {}),
            }
        cameras.append(frame_cams)

        # hands — may be absent if hand is not visible
        hands_data = json.loads(members[f'{prefix}.hands.json'])
        for side in ('left', 'right'):
            if side in hands_data and 'mano_pose' in hands_data[side]:
                mp = hands_data[side]['mano_pose']
                hands_thetas[side].append(np.array(mp['thetas'],      dtype=np.float32))
                hands_wrist[side].append( np.array(mp['wrist_xform'], dtype=np.float32))
                # Use the first available camera's amodal box as 2D observation
                boxes = hands_data[side].get('boxes_amodal', {})
                main_cam = next(iter(boxes), None)
                if main_cam:
                    hands_boxes[side].append(np.array(boxes[main_cam], dtype=np.float32))
                else:
                    hands_boxes[side].append(np.zeros(4, dtype=np.float32))
                hands_valid[side].append(True)
            else:
                # Pad with zeros for occluded/absent frames
                hands_thetas[side].append(np.zeros(15, dtype=np.float32))
                hands_wrist[side].append( np.zeros(6,  dtype=np.float32))
                hands_boxes[side].append( np.zeros(4,  dtype=np.float32))
                hands_valid[side].append(False)

        # objects
        obj_data = json.loads(members[f'{prefix}.objects.json'])
        for obj_id, obj_list in obj_data.items():
            obj = obj_list[0]  # HOT3D wraps each object in a list
            T_wo = _parse_se3(obj['T_world_from_object'])
            if obj_id not in objects_raw:
                objects_raw[obj_id] = []
                objects_meta[obj_id] = {
                    'name':   obj.get('object_name', ''),
                    'bop_id': obj.get('object_bop_id', obj_id),
                }
            objects_raw[obj_id].append(T_wo)

        # images
        if load_images:
            cam_ids = ARIA_CAMS if device == 'Aria' else QUEST3_CAMS
            for cam_id in cam_ids:
                key = f'{prefix}.image_{cam_id}.jpg'
                if key in members:
                    arr = np.frombuffer(members[key], dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if cam_id not in images_raw:
                        images_raw[cam_id] = []
                    images_raw[cam_id].append(img)

    # ---- stack per-frame lists into arrays ----
    hands = {}
    frame_valid = np.ones(T, dtype=bool)
    for side in ('left', 'right'):
        valid = np.array(hands_valid[side])
        hands[side] = {
            'thetas':      np.stack(hands_thetas[side]),   # (T, 15)
            'wrist_xform': np.stack(hands_wrist[side]),    # (T, 6)
            'boxes':       np.stack(hands_boxes[side]),    # (T, 4)
            'valid':       valid,                          # (T,)
        }
        frame_valid &= valid

    objects = {}
    for obj_id, pose_list in objects_raw.items():
        # Pad missing frames with identity (object not in view)
        if len(pose_list) == T:
            poses = np.stack(pose_list)
        else:
            poses = np.stack(pose_list + [np.eye(4, dtype=np.float32)] * (T - len(pose_list)))
        objects[obj_id] = {
            'T_world_from_object': poses,  # (T, 4, 4)
            **objects_meta[obj_id],
        }

    images = None
    if load_images:
        images = {cam_id: np.stack(frames) for cam_id, frames in images_raw.items()}

    return HOT3DClip(
        clip_id=clip_id,
        device=device,
        sequence_id=sequence_id,
        cameras=cameras,
        hands=hands,
        hand_shapes=hand_shapes,
        objects=objects,
        images=images,
        frame_valid=frame_valid,
    )


def _sample_window(clip: HOT3DClip, start: int, length: int = WINDOW_LEN) -> HOT3DClip:
    """Slice a fixed-length window from a parsed clip.

    Args:
        clip:   Parsed HOT3DClip (full length).
        start:  Start frame index.
        length: Window length (default WINDOW_LEN=120).
    Returns:
        A new HOT3DClip spanning frames [start, start+length).
    """
    end = start + length

    def _slice_dict_of_arrays(d):
        return {k: (v[start:end] if isinstance(v, np.ndarray) else v)
                for k, v in d.items()}

    hands = {side: _slice_dict_of_arrays(clip.hands[side]) for side in clip.hands}
    objects = {oid: _slice_dict_of_arrays(clip.objects[oid]) for oid in clip.objects}
    cameras = clip.cameras[start:end]
    images  = ({cam: imgs[start:end] for cam, imgs in clip.images.items()}
               if clip.images else None)

    return HOT3DClip(
        clip_id=clip.clip_id,
        device=clip.device,
        sequence_id=clip.sequence_id,
        cameras=cameras,
        hands=hands,
        hand_shapes=clip.hand_shapes,
        objects=objects,
        images=images,
        frame_valid=clip.frame_valid[start:end],
    )


class HOT3DDataset(Dataset):
    """PyTorch Dataset over HOT3D-Clip tar files.

    Each item is one WINDOW_LEN=120 frame window. Multiple windows are
    sampled from each 150-frame clip by sliding with a configurable stride.

    Args:
        tar_dir:       Directory containing clip-XXXXXX.tar files.
        window_len:    Frames per training window (default 120).
        stride:        Stride between windows within one clip (default 30).
        load_images:   Decode JPEG images (disable for faster prototyping).
        min_obj_disp:  Minimum object displacement (m) to include the clip
                       — filters static scenes that add no training signal.
        max_clips:     Cap the dataset size for debugging.
    """

    def __init__(
        self,
        tar_dir: str | Path,
        window_len: int = WINDOW_LEN,
        stride: int = 30,
        load_images: bool = True,
        min_obj_disp: float = 0.0,
        max_clips: int | None = None,
    ):
        self.window_len  = window_len
        self.stride      = stride
        self.load_images = load_images
        self.min_obj_disp = min_obj_disp

        tar_dir  = Path(tar_dir)
        tar_files = sorted(tar_dir.glob('clip-*.tar'))
        if max_clips:
            tar_files = tar_files[:max_clips]

        # Build an index: list of (tar_path, start_frame)
        self._index: list[tuple[Path, int]] = []
        for tf in tar_files:
            n_windows = max(1, (CLIP_LEN - window_len) // stride + 1)
            for i in range(n_windows):
                self._index.append((tf, i * stride))

        # Cache for recently loaded clips (avoid re-reading the same tar for
        # multiple windows from the same clip)
        self._cache: dict[str, HOT3DClip] = {}
        self._cache_max = 32

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tar_path, start = self._index[idx]
        clip_id = tar_path.stem

        if clip_id not in self._cache:
            if len(self._cache) >= self._cache_max:
                # Evict oldest entry
                self._cache.pop(next(iter(self._cache)))
            self._cache[clip_id] = _load_tar(tar_path, self.load_images)

        clip   = self._cache[clip_id]
        window = _sample_window(clip, start, self.window_len)
        return _clip_to_dict(window)


def _clip_to_dict(clip: HOT3DClip) -> dict[str, Any]:
    """Convert a HOT3DClip window to a plain dict suitable for collation."""
    T = clip.window_len if hasattr(clip, 'window_len') else len(clip.cameras)

    # Use the first visible object in the scene as the manipulation target
    obj_id   = next(iter(clip.objects))
    obj_info = clip.objects[obj_id]

    # Gravity vector from the first frame's camera pose (world z-up convention
    # in HOT3D SLAM output means gravity ≈ -z in world frame; we keep raw here
    # and let preprocessing.py apply the alignment)
    cam0_T = clip.cameras[0]
    ref_cam = '214-1' if clip.device == 'Aria' else '1201-1'
    T_wc0 = cam0_T[ref_cam]['T_world_from_camera']  # (4, 4)

    return {
        'clip_id':    clip.clip_id,
        'device':     clip.device,

        # Hand MANO params
        'left_thetas':      clip.hands['left']['thetas'],       # (T, 15)
        'left_wrist':       clip.hands['left']['wrist_xform'],  # (T, 6)
        'left_betas':       clip.hand_shapes.get('left', np.zeros(10, np.float32)),  # (10,)
        'left_valid':       clip.hands['left']['valid'],        # (T,)
        'right_thetas':     clip.hands['right']['thetas'],      # (T, 15)
        'right_wrist':      clip.hands['right']['wrist_xform'], # (T, 6)
        'right_betas':      clip.hand_shapes.get('right', np.zeros(10, np.float32)), # (10,)
        'right_valid':      clip.hands['right']['valid'],       # (T,)

        # Object SE(3) sequence
        'obj_T_world':  obj_info['T_world_from_object'],  # (T, 4, 4)
        'obj_name':     obj_info['name'],
        'obj_bop_id':   obj_info['bop_id'],

        # Reference camera pose at t=0 (for gravity alignment)
        'T_world_from_ref_cam0': T_wc0,  # (4, 4)

        # Frame validity mask
        'frame_valid': clip.frame_valid,  # (T,)
    }


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Stack numpy arrays into torch tensors; keep strings as lists."""
    out: dict[str, Any] = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], np.ndarray):
            out[key] = torch.from_numpy(np.stack(vals))
        elif isinstance(vals[0], (int, float)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals  # strings, etc.
    return out
