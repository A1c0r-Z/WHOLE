"""Combined guidance function g = g_reproj + g_inter + g_temp (Sec. 3.2).

At each diffusion step n, the guidance modifies the score:
    ∇_{x_n} log p(x_n | ŷ) = ∇_{x_n} log p(x_n) - w · ∇_{x_n} g(ŷ, x_n)

g is computed by differentiating through D_ψ(x_n), so gradients flow
through the denoiser's prediction x̂_0 back to x_n.

Usage (called inside DDPM.sample via guidance_fn callback):
    def guidance_fn(x0_pred, t):
        g = compute_guidance(x0_pred, obs)
        grad = torch.autograd.grad(g, x_n)[0]  # x_n is in outer scope
        return grad
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from guidance.reprojection import guidance_reproj
from losses.interaction    import loss_interaction
from losses.smoothness     import loss_smooth


@dataclass
class GuidanceObs:
    """All observations needed for test-time guidance."""
    # Camera
    T_world_from_cam:  Optional[torch.Tensor] = None  # (B, T, 4, 4) SLAM poses
    intrinsics:        Optional[dict]          = None  # camera intrinsics dict

    # 2D observations
    obs_hand_masks:    Optional[torch.Tensor] = None  # (B, T, H, W) binary
    obs_obj_masks:     Optional[torch.Tensor] = None  # (B, T, H, W) binary

    # Contact labels from VLM
    contact_labels:    Optional[torch.Tensor] = None  # (B, T, 2)

    # Object geometry
    template_verts:    Optional[torch.Tensor] = None  # (V, 3) canonical vertices

    # Frame validity
    frame_valid:       Optional[torch.Tensor] = None  # (B, T) bool


@dataclass
class GuidanceWeights:
    """Per-term guidance weights (tunable at inference time)."""
    reproj: float = 1.0
    inter:  float = 1.0
    temp:   float = 0.1


def compute_guidance(
    x0_pred: torch.Tensor,      # (B, T, 73)  — output of denoiser, on computation graph
    obs:     GuidanceObs,
    weights: GuidanceWeights = GuidanceWeights(),
) -> torch.Tensor:
    """Compute the combined guidance cost g(ŷ, x̂_0).

    This is the function whose gradient steers the diffusion sampling toward
    observations consistent with the input video.

    Args:
        x0_pred: predicted clean trajectory (must retain grad for backprop to x_n)
        obs:     all available video observations
        weights: per-term weights
    Returns:
        scalar guidance cost
    """
    g = x0_pred.new_zeros(())

    # ---- g_reproj ----
    if (obs.T_world_from_cam is not None and obs.intrinsics is not None and
            (obs.obs_hand_masks is not None or obs.obs_obj_masks is not None)):
        g_r = guidance_reproj(
            x0_pred          = x0_pred,
            T_world_from_cam = obs.T_world_from_cam,
            intrinsics       = obs.intrinsics,
            obs_hand_masks   = obs.obs_hand_masks,
            obs_obj_masks    = obs.obs_obj_masks,
            contact_labels   = obs.contact_labels,
            template_verts   = obs.template_verts,
            frame_valid      = obs.frame_valid,
        )
        g = g + weights.reproj * g_r

    # ---- g_inter ----
    if obs.template_verts is not None:
        g_i = loss_interaction(
            x0_pred        = x0_pred,
            template_verts = obs.template_verts,
            contact_gt     = obs.contact_labels,
            frame_valid    = obs.frame_valid,
        )
        g = g + weights.inter * g_i

    # ---- g_temp ----
    g_t = loss_smooth(x0_pred, obs.frame_valid)
    g   = g + weights.temp * g_t

    return g


def make_guidance_fn(
    obs:     GuidanceObs,
    weights: GuidanceWeights = GuidanceWeights(),
):
    """Return a guidance_fn(x0_pred, t) → scalar for use in DDPM.sample."""
    def guidance_fn(x0_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return compute_guidance(x0_pred, obs, weights)
    return guidance_fn
