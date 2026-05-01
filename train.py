"""WHOLE training script.

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 train.py --config configs/default.yaml

Usage (debug, 1 GPU, small steps):
    python train.py --config configs/default.yaml --debug

Curriculum (Sec. 3.1, Appendix A):
    Steps       0 – aux_loss_start:   L_DDPM only
    Steps aux_loss_start – end:       L_DDPM + λ_inter·L_inter
                                             + λ_const·L_const
                                             + λ_smooth·L_smooth
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from data.hot3d_loader import HOT3DDataset, collate_fn
from data.preprocessing import preprocess_window
from models import build_denoiser, build_diffusion, ObjectBPSCache
from losses import loss_smooth, loss_consistency, loss_interaction
import trimesh
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str):
    if is_main():
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def get_lr(step: int, warmup: int, max_lr: float, total: int) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def load_object_template(models_dir: str, bop_id: str) -> torch.Tensor | None:
    """Load canonical object mesh vertices for a given BOP ID."""
    glb = Path(models_dir) / f'obj_{int(bop_id):06d}.glb'
    if not glb.exists():
        return None
    mesh = trimesh.load(str(glb), force='mesh')
    verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    center = verts.mean(0)
    scale  = (verts - center).norm(dim=1).max().clamp(min=1e-6)
    return (verts - center) / scale


# ---------------------------------------------------------------------------
# Dataset wrapper that runs preprocessing
# ---------------------------------------------------------------------------

class PreprocessedDataset(torch.utils.data.Dataset):
    """Wraps HOT3DDataset and applies preprocess_window on-the-fly."""

    def __init__(self, tar_dir: str, cfg: dict, augment: bool = True):
        m = cfg['data']
        self.ds = HOT3DDataset(
            tar_dir    = tar_dir,
            window_len = m['window_len'],
            stride     = m['stride'],
            load_images = False,   # images not needed for diffusion training
        )
        self.augment = augment
        self.bps_cache = ObjectBPSCache(
            models_dir = m['object_models_dir'],
            n_points   = cfg['model']['bps_n_points'],
            radius     = cfg['model']['bps_radius'],
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        raw = self.ds[idx]
        proc = preprocess_window(raw, augment=self.augment)
        bps  = self.bps_cache.get(raw['obj_bop_id'])

        return {
            'x0':          torch.tensor(proc['x0'],          dtype=torch.float32),
            'H_tilde':     torch.tensor(proc['H_tilde'],     dtype=torch.float32),
            'frame_valid': torch.tensor(proc['frame_valid'], dtype=torch.bool),
            'O':           bps,
            'obj_bop_id':  raw['obj_bop_id'],
        }


def build_datasets(cfg: dict) -> tuple[PreprocessedDataset, ...]:
    d = cfg['data']
    train_dirs = [d['train_aria_dir'], d['train_quest3_dir']]
    datasets = []
    for td in train_dirs:
        if Path(td).exists():
            datasets.append(PreprocessedDataset(td, cfg, augment=True))
            log(f'  Loaded train split: {td}  ({len(datasets[-1])} windows)')
    if not datasets:
        raise RuntimeError('No training data found — check config paths')
    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(
    batch:    dict,
    model:    nn.Module,
    diffusion,
    cfg:      dict,
    step:     int,
    device:   torch.device,
    obj_template_cache: dict,
) -> dict[str, torch.Tensor]:
    """Compute all losses for one batch.

    Args:
        batch:    collated batch from DataLoader
        model:    WHOLEDenoiser
        diffusion: DDPM instance
        cfg:      full config dict
        step:     current training step
        device:   target device
    Returns:
        dict of named scalar losses, plus 'loss' (total)
    """
    x0          = batch['x0'].to(device)           # (B, T, 73)
    H_tilde     = batch['H_tilde'].to(device)
    O           = batch['O'].to(device)
    frame_valid = batch['frame_valid'].to(device)

    # ---- DDPM loss ----
    out = diffusion.compute_loss(x0, H_tilde, O, mask=frame_valid)
    losses = {'loss_ddpm': out['loss_ddpm']}

    # ---- Auxiliary losses (after curriculum warm-up) ----
    aux_start = cfg['training']['aux_loss_start']
    if step >= aux_start:
        x0_pred = out['x0_pred'].detach()   # detach for auxiliary: no double backprop

        losses['loss_smooth'] = loss_smooth(x0_pred, frame_valid)
        losses['loss_const']  = loss_consistency(x0_pred, x0, frame_valid)

        # Interaction loss needs the object template mesh
        bop_ids = batch['obj_bop_id']
        # Use the most common BOP ID in the batch as the template
        # (in practice batches often share the same object for HOT3D clips)
        bop_id = bop_ids[0] if isinstance(bop_ids, list) else bop_ids
        if bop_id not in obj_template_cache:
            verts = load_object_template(cfg['data']['object_models_dir'], bop_id)
            obj_template_cache[bop_id] = verts
        template_verts = obj_template_cache.get(bop_id)

        if template_verts is not None:
            template_verts = template_verts.to(device)
            losses['loss_inter'] = loss_interaction(
                x0_pred, template_verts, frame_valid=frame_valid
            )
        else:
            losses['loss_inter'] = x0_pred.new_zeros(())

    # ---- Combine ----
    tc = cfg['training']
    total = losses['loss_ddpm']
    if step >= aux_start:
        total = (total
                 + tc['lambda_inter']  * losses.get('loss_inter',  torch.zeros(()))
                 + tc['lambda_const']  * losses.get('loss_const',  torch.zeros(()))
                 + tc['lambda_smooth'] * losses.get('loss_smooth', torch.zeros(())))
    losses['loss'] = total
    return losses


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--debug',  action='store_true',
                        help='Run 200 steps with batch_size=2 for smoke test')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # ---- DDP setup ----
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group('nccl')
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    if args.debug:
        cfg['training']['max_iters']      = 200
        cfg['training']['aux_loss_start'] = 100
        cfg['training']['save_every']     = 100
        cfg['training']['val_every']      = 50
        cfg['data']['stride']             = 60   # fewer windows for speed
        log('DEBUG mode: 200 steps, batch_size=2')

    # ---- Model ----
    denoiser  = build_denoiser(cfg).to(device)
    diffusion = build_diffusion(denoiser, cfg).to(device)

    if dist.is_initialized():
        denoiser = DDP(denoiser, device_ids=[local_rank])

    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    log(f'Model: {n_params:,} parameters')

    # ---- Optimizer + LR scheduler ----
    tc = cfg['training']
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr           = tc['lr'],
        weight_decay = tc['weight_decay'],
        betas        = (0.9, 0.999),
    )
    scaler = torch.amp.GradScaler('cuda')   # mixed precision

    # ---- Data ----
    log('Building datasets...')
    train_ds = build_datasets(cfg)

    sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
    batch_size = 2 if args.debug else tc['batch_size']
    loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = 0 if args.debug else tc['num_workers'],
        pin_memory  = True,
        drop_last   = True,
    )
    log(f'Dataset: {len(train_ds)} windows | batch {batch_size} | '
        f'{len(loader)} steps/epoch')

    # ---- Resume ----
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        (denoiser.module if dist.is_initialized() else denoiser).load_state_dict(
            ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step'] + 1
        log(f'Resumed from step {start_step}')

    out_dir = Path(cfg['training']['out_dir'])
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    obj_template_cache: dict = {}
    data_iter  = iter(loader)
    step       = start_step
    max_steps  = tc['max_iters']
    log_every  = tc['log_every']
    save_every = tc['save_every']
    t0         = time.time()
    loss_accum: dict[str, float] = {}

    log(f'Training for {max_steps} steps...')

    while step < max_steps:
        # Refresh iterator at epoch boundary
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            data_iter = iter(loader)
            batch = next(data_iter)

        # Learning rate
        lr = get_lr(step, tc['warmup_iters'], tc['lr'], max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            losses = training_step(batch, denoiser, diffusion, cfg,
                                   step, device, obj_template_cache)

        scaler.scale(losses['loss']).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate for logging
        for k, v in losses.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v.item()

        # Logging
        if step % log_every == 0 and is_main():
            avg = {k: v / log_every for k, v in loss_accum.items()}
            parts = [f'{k}={avg[k]:.4f}' for k in sorted(avg)]
            elapsed = time.time() - t0
            log(f'step={step:7d}  lr={lr:.2e}  {" | ".join(parts)}  '
                f'({elapsed/max(step-start_step,1)*1000:.0f}ms/step)')
            loss_accum = {}

        # Checkpointing
        if step % save_every == 0 and step > start_step and is_main():
            m = denoiser.module if dist.is_initialized() else denoiser
            ckpt_path = out_dir / f'ckpt_{step:07d}.pt'
            torch.save({
                'step':      step,
                'model':     m.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg':       cfg,
            }, ckpt_path)
            log(f'Saved checkpoint: {ckpt_path}')

        step += 1

    log('Training complete.')
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
