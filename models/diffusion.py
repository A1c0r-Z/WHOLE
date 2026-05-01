"""DDPM forward/reverse process for WHOLE.

Implements the conditional DDPM framework from Sec. 3.1:
  - Cosine noise schedule (default) or linear
  - Weighted training loss L_DDPM = E[w_n * ||x̃_0 - D_ψ(x_n | n, H̃, O)||²]
  - Sampling: iterative denoising x_N → x_0, with optional guidance hook
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021.

    Returns:
        (N,) beta tensor
    """
    t = torch.linspace(0, n_steps, n_steps + 1)
    alpha_bar = torch.cos((t / n_steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


def linear_beta_schedule(n_steps: int, beta_min: float = 1e-4,
                          beta_max: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_min, beta_max, n_steps)


# ---------------------------------------------------------------------------
# DDPM core
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """Wraps the denoiser with the DDPM forward/reverse process.

    Args:
        denoiser:  WHOLEDenoiser instance
        n_steps:   total diffusion steps N (default 1000)
        schedule:  'cosine' or 'linear'
        beta_min, beta_max: used only for linear schedule
    """

    def __init__(
        self,
        denoiser:   nn.Module,
        n_steps:    int   = 1000,
        schedule:   str   = 'cosine',
        beta_min:   float = 1e-4,
        beta_max:   float = 0.02,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.N = n_steps

        if schedule == 'cosine':
            betas = cosine_beta_schedule(n_steps)
        else:
            betas = linear_beta_schedule(n_steps, beta_min, beta_max)

        alphas      = 1.0 - betas
        alpha_bar   = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        # Register as buffers so they move with .to(device)
        self.register_buffer('betas',           betas)
        self.register_buffer('alphas',          alphas)
        self.register_buffer('alpha_bar',       alpha_bar)
        self.register_buffer('alpha_bar_prev',  alpha_bar_prev)
        self.register_buffer('sqrt_alpha_bar',         alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alpha_bar', (1 - alpha_bar).sqrt())
        # Variance schedule weight w_n (used to weight DDPM loss per step)
        # w_n = 1 / (1 - alpha_bar_n) following the original DDPM paper
        self.register_buffer('loss_weight', 1.0 / (1.0 - alpha_bar).clamp(min=1e-5))

    # ------------------------------------------------------------------
    # Forward (noising)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,   # (B, T, D) clean trajectory
        t:  torch.Tensor,   # (B,)     integer step indices
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy trajectory x_t ~ q(x_t | x_0).

        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Args:
            x0:    (B, T, D) clean trajectory
            t:     (B,) step indices
            noise: optional pre-sampled Gaussian noise (for testing)
        Returns:
            x_t:   (B, T, D) noisy trajectory
            noise: (B, T, D) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1ab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0:      torch.Tensor,    # (B, T, 73)
        H_tilde: torch.Tensor,    # (B, T, 62)
        O:       torch.Tensor,    # (B, K)
        ambient: Optional[torch.Tensor] = None,   # (B, T, J*3)
        mask:    Optional[torch.Tensor] = None,   # (B, T) bool valid frames
        t:       Optional[torch.Tensor] = None,   # override sampled steps
    ) -> dict[str, torch.Tensor]:
        """Compute weighted DDPM loss.

        L_DDPM = E_{n,ε}[ w_n * ||x̃_0 - D_ψ(x_n | n, H̃, O)||² ]

        Args:
            x0:      ground-truth clean trajectory
            H_tilde: noisy hand conditioning
            O:       object BPS descriptor
            ambient: ambient sensor feature
            mask:    True = valid frame (False = occluded/padded, excluded)
            t:       if None, sample uniformly from [0, N-1]
        Returns:
            dict with 'loss_ddpm' and 'x0_pred' for auxiliary loss computation
        """
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.N, (B,), device=x0.device)

        x_t, _ = self.q_sample(x0, t)
        x0_pred = self.denoiser(x_t, t, H_tilde, O, ambient)

        # Per-frame squared error
        err = (x0 - x0_pred) ** 2    # (B, T, D)

        if mask is not None:
            # Only compute loss on valid frames
            err = err * mask.float().unsqueeze(-1)
            n_valid = mask.float().sum().clamp(min=1)
            mse = err.sum() / (n_valid * x0.shape[-1])
        else:
            mse = err.mean()

        # Variance schedule weighting w_n (B,) → scalar via mean
        w_n = self.loss_weight[t].mean()
        loss_ddpm = w_n * mse

        return {
            'loss_ddpm': loss_ddpm,
            'x0_pred':   x0_pred,
            't':         t,
        }

    # ------------------------------------------------------------------
    # Sampling (reverse process)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self,
        x_t:     torch.Tensor,   # (B, T, D)
        t:       torch.Tensor,   # (B,)
        H_tilde: torch.Tensor,
        O:       torch.Tensor,
        ambient: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One reverse diffusion step x_t → x_{t-1} (DDPM posterior).

        Returns:
            x_{t-1}: (B, T, D)
            x0_pred: (B, T, D) predicted clean trajectory (for guidance)
        """
        x0_pred = self.denoiser(x_t, t, H_tilde, O, ambient)

        ab     = self.alpha_bar[t].view(-1, 1, 1)
        ab_p   = self.alpha_bar_prev[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)

        # Posterior mean: μ̃_t(x_t, x_0)
        coeff1 = beta_t * ab_p.sqrt() / (1 - ab)
        coeff2 = (1 - ab_p) * alpha_t.sqrt() / (1 - ab)
        mean   = coeff1 * x0_pred + coeff2 * x_t

        # Posterior variance (zero at t=0)
        var    = beta_t * (1 - ab_p) / (1 - ab)
        noise  = torch.randn_like(x_t)
        nonzero = (t > 0).float().view(-1, 1, 1)
        x_prev = mean + nonzero * var.sqrt() * noise

        return x_prev, x0_pred

    @torch.no_grad()
    def sample(
        self,
        H_tilde:  torch.Tensor,          # (B, T, 62)
        O:        torch.Tensor,          # (B, K)
        guidance_fn: Optional[Callable] = None,
        guidance_w:  float = 1.0,
        ambient_fn:  Optional[Callable] = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Full reverse diffusion: x_N → x_0.

        Args:
            H_tilde:      noisy hand conditioning
            O:            object BPS descriptor
            guidance_fn:  callable(x0_pred, t) → gradient to subtract;
                          implements the guidance step g from Sec. 3.2
            guidance_w:   guidance weight w
            ambient_fn:   callable(x0_pred) → ambient sensor (B, T, J*3)
            show_progress: show tqdm progress bar
        Returns:
            (B, T, 73) sampled clean trajectory
        """
        B, T, _ = H_tilde.shape
        device   = H_tilde.device

        x = torch.randn(B, T, self.denoiser.x_dim, device=device)

        steps = range(self.N - 1, -1, -1)
        if show_progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc='Sampling')

        for n in steps:
            t = torch.full((B,), n, device=device, dtype=torch.long)

            # Ambient sensor from current estimate (optional)
            ambient = ambient_fn(x) if ambient_fn is not None else None

            if guidance_fn is not None:
                # Guidance step: update x using gradient of guidance objective
                x = x.detach().requires_grad_(True)
                x0_pred_g = self.denoiser(x, t, H_tilde, O, ambient)
                grad = guidance_fn(x0_pred_g, t)
                x = (x - guidance_w * grad).detach()

            x, _ = self.p_sample_step(x, t, H_tilde, O, ambient)

        return x


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------

def build_diffusion(denoiser: nn.Module, cfg: dict) -> DDPM:
    d = cfg.get('diffusion', cfg)
    return DDPM(
        denoiser  = denoiser,
        n_steps   = d.get('n_steps',  1000),
        schedule  = d.get('schedule', 'cosine'),
        beta_min  = d.get('beta_min', 1e-4),
        beta_max  = d.get('beta_max', 0.02),
    )
