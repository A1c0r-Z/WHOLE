"""Temporal smoothness loss L_smooth.

Penalizes large accelerations in the predicted trajectory by applying
second-order finite differences along the time axis.

Applied to the full 73D trajectory x̂_0 (both hand and object parameters),
which implicitly enforces smooth wrist translations, joint angles, and
object motion simultaneously.
"""

import torch


def loss_smooth(
    x0_pred:    torch.Tensor,     # (B, T, 73)
    frame_valid: torch.Tensor | None = None,  # (B, T) bool
) -> torch.Tensor:
    """Penalize squared acceleration in the predicted trajectory.

    acc[t] = x[t+2] - 2*x[t+1] + x[t]   (second-order finite difference)

    Args:
        x0_pred:     (B, T, 73) predicted clean trajectory
        frame_valid: optional validity mask; invalid frames contribute zero
    Returns:
        scalar loss
    """
    # acc shape: (B, T-2, 73)
    acc = x0_pred[:, 2:] - 2 * x0_pred[:, 1:-1] + x0_pred[:, :-2]

    if frame_valid is not None:
        # A frame's acceleration is valid only if all three frames are valid
        valid = (frame_valid[:, 2:] & frame_valid[:, 1:-1] & frame_valid[:, :-2])
        acc = acc * valid.float().unsqueeze(-1)
        n = valid.float().sum().clamp(min=1)
        return (acc ** 2).sum() / (n * x0_pred.shape[-1])

    return (acc ** 2).mean()
