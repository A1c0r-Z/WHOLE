"""WHOLE inference: guided generation from a metric-SLAMed egocentric clip.

Usage:
    python inference.py \\
        --checkpoint /path/to/ckpt.pt \\
        --tar_path   /path/to/clip-XXXXXX.tar \\
        --out_dir    /path/to/output \\
        [--use_vlm]  [--use_hawor]

Pipeline (Sec. 3.2):
  1. Load clip (frames, SLAM poses, hand annotations, object template)
  2. Estimate H̃  via HaWoR (or GT fallback)
  3. Label contacts via VLM (or zero / GT fallback)
  4. For each sliding window: interleave diffusion + guidance steps
  5. Blend overlapping windows → full-length reconstruction
  6. Save result as .npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from data.hot3d_loader import _load_tar
from data.preprocessing import gravity_align_window, build_diffusion_variable
from models import build_denoiser, build_diffusion, ObjectBPSCache
from guidance.guidance import GuidanceObs, GuidanceWeights, make_guidance_fn
from guidance.vlm_contact import VLMContactLabeler
from hand_estimator import HaWoRWrapper
from utils.mano_utils import unpack_x0


WINDOW_LEN = 120
OVERLAP    = 30   # frames of overlap between adjacent windows


# ---------------------------------------------------------------------------
# Window blending
# ---------------------------------------------------------------------------

def blend_windows(
    windows:    list[np.ndarray],   # list of (T_win, 73) arrays
    starts:     list[int],
    total_len:  int,
) -> np.ndarray:
    """Blend overlapping windows into a full-length trajectory.

    In overlap regions, MANO shape parameters (betas, dims 32-42 and 63-73)
    are averaged; other parameters use a linear blend.
    """
    out     = np.zeros((total_len, 73), dtype=np.float32)
    weights = np.zeros((total_len, 1),  dtype=np.float32)
    win_len = windows[0].shape[0]

    # Hanning window weights for smooth blending
    hann = np.hanning(win_len).reshape(-1, 1).astype(np.float32)
    hann = np.clip(hann, 1e-3, None)

    for win, start in zip(windows, starts):
        end = start + win_len
        out[start:end]     += win * hann
        weights[start:end] += hann

    weights = np.clip(weights, 1e-6, None)
    return out / weights


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

def build_windows(total_len: int, win_len: int = WINDOW_LEN,
                  overlap: int = OVERLAP) -> list[int]:
    """Return start indices of sliding windows covering [0, total_len)."""
    stride = win_len - overlap
    starts = list(range(0, total_len - win_len + 1, stride))
    if not starts or starts[-1] + win_len < total_len:
        starts.append(max(0, total_len - win_len))
    return starts


# ---------------------------------------------------------------------------
# Per-window guided generation
# ---------------------------------------------------------------------------

def guided_generation_window(
    H_tilde:      torch.Tensor,          # (1, T, 62)
    O:            torch.Tensor,          # (1, K)
    diffusion,
    obs:          GuidanceObs,
    weights:      GuidanceWeights,
    guidance_w:   float = 1.0,
    device:       torch.device = torch.device('cpu'),
    show_progress: bool = False,
) -> np.ndarray:
    """Run guided generation for one window; return (T, 73) numpy array."""
    H = H_tilde.to(device)
    O = O.to(device)
    if obs.T_world_from_cam is not None:
        obs = GuidanceObs(**{
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in obs.__dict__.items()
        })
    if obs.template_verts is not None:
        obs.template_verts = obs.template_verts.to(device)

    guidance_fn = make_guidance_fn(obs, weights) if _has_observations(obs) else None

    with torch.no_grad() if guidance_fn is None else torch.enable_grad():
        x0 = diffusion.sample(
            H_tilde      = H,
            O            = O,
            guidance_fn  = guidance_fn,
            guidance_w   = guidance_w,
            show_progress = show_progress,
        )

    return x0[0].detach().cpu().numpy()   # (T, 73)


def _has_observations(obs: GuidanceObs) -> bool:
    return (obs.obs_hand_masks is not None or
            obs.obs_obj_masks  is not None or
            obs.contact_labels is not None or
            obs.template_verts is not None)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer_clip(
    tar_path:       str | Path,
    checkpoint:     str | Path,
    cfg:            dict,
    out_dir:        str | Path,
    use_vlm:        bool = False,
    use_hawor:      bool = False,
    openai_api_key: Optional[str] = None,
    guidance_w:     float = 1.0,
    device:         Optional[torch.device] = None,
    show_progress:  bool = True,
) -> dict:
    """Run WHOLE inference on a single clip tar file.

    Args:
        tar_path:    path to clip-XXXXXX.tar
        checkpoint:  path to trained model checkpoint (.pt)
        cfg:         config dict (loaded from yaml)
        out_dir:     output directory for results
        use_vlm:     enable VLM contact labeling
        use_hawor:   use HaWoR for H̃ (else GT fallback with noise)
        guidance_w:  guidance weight w
        device:      target device (default: cuda if available)
    Returns:
        dict with 'x0_full' (T, 73) and metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    denoiser  = build_denoiser(cfg).to(device)
    diffusion = build_diffusion(denoiser, cfg).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    denoiser.load_state_dict(ckpt['model'])
    denoiser.eval()

    # ---- Parse clip ----
    print(f'Loading clip: {tar_path}')
    clip = _load_tar(tar_path, load_images=use_vlm or True)

    # Pick the primary camera for this device
    ref_cam = '214-1' if clip.device == 'Aria' else '1201-1'
    T_total = len(clip.cameras)

    # ---- Object BPS ----
    obj_id   = next(iter(clip.objects))
    bop_id   = clip.objects[obj_id]['bop_id']
    bps_cache = ObjectBPSCache(cfg['data']['object_models_dir'],
                               n_points=cfg['model']['bps_n_points'])
    O = bps_cache.get(bop_id).unsqueeze(0)   # (1, K)

    # ---- Load object template vertices ----
    import trimesh
    glb = Path(cfg['data']['object_models_dir']) / f'obj_{int(bop_id):06d}.glb'
    template_verts = None
    if glb.exists():
        mesh = trimesh.load(str(glb), force='mesh')
        verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
        center = verts.mean(0); scale = (verts - center).norm(dim=1).max().clamp(min=1e-6)
        template_verts = (verts - center) / scale   # normalized canonical

    # ---- HaWoR H̃ ----
    hawor = HaWoRWrapper(use_gt_fallback=not use_hawor)
    frames_np = clip.images[ref_cam] if clip.images else None

    slam_poses = np.stack([clip.cameras[t][ref_cam]['T_world_from_camera']
                           for t in range(T_total)])  # (T, 4, 4)

    H_tilde_full = hawor.estimate(
        frames          = frames_np,
        slam_poses      = slam_poses,
        clip_id         = clip.clip_id,
        gt_left_thetas  = clip.hands['left']['thetas'],
        gt_left_wrist   = clip.hands['left']['wrist_xform'],
        gt_left_betas   = clip.hand_shapes.get('left', np.zeros(10)),
        gt_right_thetas = clip.hands['right']['thetas'],
        gt_right_wrist  = clip.hands['right']['wrist_xform'],
        gt_right_betas  = clip.hand_shapes.get('right', np.zeros(10)),
    )   # (1, T, 62)

    # ---- VLM contact labels ----
    contact_full = None
    if use_vlm and frames_np is not None:
        vlm = VLMContactLabeler(api_key=openai_api_key)
        obj_info = clip.objects[obj_id]
        obj_names = [obj_info['name'] or f'obj{bop_id}']

        left_boxes  = clip.hands['left']['boxes']    # (T, 4)
        right_boxes = clip.hands['right']['boxes']   # (T, 4)

        contact_full = vlm.label_clip(
            frames      = frames_np,
            left_boxes  = left_boxes,
            right_boxes = right_boxes,
            obj_masks   = [],
            obj_names   = obj_names,
        ).unsqueeze(0)   # (1, T, 2)

    # ---- SLAM poses as tensor ----
    T_world_from_cam_full = torch.tensor(slam_poses, dtype=torch.float32).unsqueeze(0)

    # Get camera intrinsics from first frame
    cam0_calib = clip.cameras[0][ref_cam].get('calibration', {})
    proj_params = cam0_calib.get('projection_params', [])
    if proj_params:
        intrinsics = {
            'model': 'fisheye624',
            'projection_params': proj_params,
        }
    else:
        intrinsics = {'model': 'pinhole', 'fx': 500, 'fy': 500,
                      'cx': 320, 'cy': 240}

    # ---- Gravity alignment (for display; windows handle their own) ----
    # The gravity rotation is applied inside preprocess_window / per window below

    # ---- Sliding window inference ----
    starts    = build_windows(T_total, WINDOW_LEN, OVERLAP)
    win_results: list[np.ndarray] = []

    for i, start in enumerate(starts):
        end = start + WINDOW_LEN
        print(f'  Window {i+1}/{len(starts)}: frames {start}–{end}')

        H_win = H_tilde_full[:, start:end, :]              # (1, T, 62)
        T_cam_win = T_world_from_cam_full[:, start:end, :, :]

        contact_win = (contact_full[:, start:end, :]
                       if contact_full is not None else None)

        obs = GuidanceObs(
            T_world_from_cam = T_cam_win,
            intrinsics       = intrinsics,
            contact_labels   = contact_win,
            template_verts   = template_verts,
        )
        weights = GuidanceWeights(reproj=0.0, inter=1.0, temp=0.1)

        x0_win = guided_generation_window(
            H_tilde      = H_win,
            O            = O,
            diffusion    = diffusion,
            obs          = obs,
            weights      = weights,
            guidance_w   = guidance_w,
            device       = device,
            show_progress = show_progress and (i == 0),
        )
        win_results.append(x0_win)

    # ---- Blend windows ----
    x0_full = blend_windows(win_results, starts, T_total)  # (T_total, 73)

    # ---- Save ----
    clip_stem = Path(tar_path).stem
    out_path  = out_dir / f'{clip_stem}_reconstruction.npz'
    np.savez(str(out_path),
             x0=x0_full,
             clip_id=clip.clip_id,
             device=clip.device,
             obj_bop_id=bop_id)
    print(f'Saved: {out_path}')

    return {'x0_full': x0_full, 'clip_id': clip.clip_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tar_path',   required=True)
    parser.add_argument('--config',     default='configs/default.yaml')
    parser.add_argument('--out_dir',    default='/tmp/whole_inference')
    parser.add_argument('--use_vlm',    action='store_true')
    parser.add_argument('--use_hawor',  action='store_true')
    parser.add_argument('--guidance_w', type=float, default=1.0)
    parser.add_argument('--openai_key', type=str,   default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    infer_clip(
        tar_path       = args.tar_path,
        checkpoint     = args.checkpoint,
        cfg            = cfg,
        out_dir        = args.out_dir,
        use_vlm        = args.use_vlm,
        use_hawor      = args.use_hawor,
        openai_api_key = args.openai_key,
        guidance_w     = args.guidance_w,
    )


if __name__ == '__main__':
    main()
