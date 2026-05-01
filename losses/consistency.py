"""Consistency loss L_const.

Promotes agreement between predicted joint positions and MANO FK.

From Appendix A: ||J_ψ - MANO(Γ_ψ, Λ_ψ, Θ_ψ)||_2

In practice we compare the FK joints derived from the predicted MANO
parameters (x0_pred) against those from the ground-truth parameters (x0).
This adds a joint-space supervision signal on top of the parameter-space
DDPM loss, which is particularly useful for wrist position accuracy.

When MANO FK is unavailable (no chumpy), falls back to a simple
wrist-translation consistency check using only the predicted transl vs GT.
"""

import torch
import torch.nn.functional as F

from utils.mano_utils import fk_from_x0, unpack_x0


def loss_consistency(
    x0_pred:     torch.Tensor,              # (B, T, 73) predicted
    x0_gt:       torch.Tensor,              # (B, T, 73) ground truth
    frame_valid: torch.Tensor | None = None,  # (B, T) bool
) -> torch.Tensor:
    """Joint-space consistency loss between predicted and GT MANO FK.

    Tries full 21-joint FK first; falls back to wrist-only if MANO is
    unavailable.

    Args:
        x0_pred:     predicted clean trajectory
        x0_gt:       ground-truth clean trajectory
        frame_valid: optional validity mask
    Returns:
        scalar loss
    """
    left_pred  = fk_from_x0(x0_pred, 'left')
    right_pred = fk_from_x0(x0_pred, 'right')

    if left_pred is None or right_pred is None:
        # MANO unavailable — fall back to wrist translation only
        return _wrist_consistency(x0_pred, x0_gt, frame_valid)

    left_gt   = fk_from_x0(x0_gt,   'left')
    right_gt  = fk_from_x0(x0_gt,   'right')

    return _joint_consistency(
        torch.cat([left_pred, right_pred], dim=2),   # (B, T, 42, 3)
        torch.cat([left_gt,   right_gt],   dim=2),
        frame_valid,
    )


def _joint_consistency(
    joints_pred: torch.Tensor,   # (B, T, J, 3)
    joints_gt:   torch.Tensor,   # (B, T, J, 3)
    frame_valid: torch.Tensor | None,
) -> torch.Tensor:
    err = (joints_pred - joints_gt) ** 2   # (B, T, J, 3)
    if frame_valid is not None:
        mask = frame_valid.float()[:, :, None, None]
        err  = err * mask
        n    = frame_valid.float().sum().clamp(min=1)
        return err.sum() / (n * joints_pred.shape[2] * 3)
    return err.mean()


def _wrist_consistency(
    x0_pred:     torch.Tensor,
    x0_gt:       torch.Tensor,
    frame_valid: torch.Tensor | None,
) -> torch.Tensor:
    """Fallback: compare wrist translations from predicted vs GT."""
    p  = unpack_x0(x0_pred)
    gt = unpack_x0(x0_gt)
    err = (
        (p['left_transl']  - gt['left_transl'])  ** 2 +
        (p['right_transl'] - gt['right_transl']) ** 2
    )   # (B, T, 3)
    if frame_valid is not None:
        err = err * frame_valid.float().unsqueeze(-1)
        return err.sum() / (frame_valid.float().sum().clamp(min=1) * 3)
    return err.mean()
