"""Interaction loss L_inter.

From Appendix A:
  "Encourages realistic contact between predicted hand-object motions and
  contact labels. Penalizes hand-object distances when contact is predicted
  and enforces near-rigid transport of contact points across consecutive
  contact frames. Specifically, for each hand joint, we find its nearest
  object point p^i, rotate it by the object's relative motion, and penalize
  deviation from its counterpart p^{i+1},
  i.e. ||R^{i+1}(R^i)^T p^i - p^{i+1}||"

Two terms:
  1. Contact distance: when contact[t]=1, penalize hand-joint to object distance
  2. Near-rigid transport: consecutive contact frames must obey object motion
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.mano_utils import fk_from_x0, get_obj_transform, apply_obj_transform, unpack_x0


def _nearest_obj_point(
    joints:    torch.Tensor,   # (B, T, J, 3)
    obj_verts: torch.Tensor,   # (B, T, V, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each joint, find nearest object vertex.

    Returns:
        nn_verts: (B, T, J, 3)  nearest vertex positions
        nn_idx:   (B, T, J)     indices into V dim
    """
    # (B, T, J, V)
    diff  = obj_verts.unsqueeze(2) - joints.unsqueeze(3)
    sq_d  = (diff ** 2).sum(-1)
    nn_idx = sq_d.argmin(-1)                          # (B, T, J)

    idx_exp = nn_idx[..., None, None].expand(*nn_idx.shape, 1, 3)
    nn_verts = obj_verts.unsqueeze(2).expand(
        *obj_verts.shape[:2], joints.shape[2], *obj_verts.shape[2:]
    ).gather(3, idx_exp).squeeze(3)                   # (B, T, J, 3)
    return nn_verts, nn_idx


def loss_interaction(
    x0_pred:        torch.Tensor,       # (B, T, 73)
    template_verts: torch.Tensor,       # (V, 3) canonical object vertices
    contact_gt:     torch.Tensor | None = None,  # (B, T, 2) or None → use predicted
    frame_valid:    torch.Tensor | None = None,  # (B, T) bool
) -> torch.Tensor:
    """Compute L_inter = contact_distance + near_rigid_transport.

    Falls back to a wrist-proximity version when MANO FK is unavailable.

    Args:
        x0_pred:        predicted clean trajectory (B, T, 73)
        template_verts: canonical object mesh vertices (V, 3)
        contact_gt:     GT contact labels (B, T, 2); if None, uses x0_pred
        frame_valid:    validity mask (B, T)
    Returns:
        scalar interaction loss
    """
    # ---- object world vertices ----
    T_obj     = get_obj_transform(x0_pred)                    # (B, T, 4, 4)
    obj_verts = apply_obj_transform(T_obj, template_verts)    # (B, T, V, 3)

    # ---- contact labels ----
    if contact_gt is not None:
        contact = contact_gt.float()
    else:
        contact = x0_pred[..., 9:11].sigmoid()   # (B, T, 2) soft labels

    # ---- hand joints ----
    left_j  = fk_from_x0(x0_pred, 'left')    # (B, T, 21, 3) or None
    right_j = fk_from_x0(x0_pred, 'right')

    if left_j is None or right_j is None:
        return _wrist_interaction(x0_pred, obj_verts, contact, frame_valid)

    joints = torch.cat([left_j, right_j], dim=2)   # (B, T, 42, 3)
    # Replicate contact per joint: (B, T, 2) → (B, T, 42)
    contact_per_joint = contact.repeat_interleave(21, dim=-1)  # (B, T, 42)

    l_dist  = _contact_distance(joints, obj_verts, contact_per_joint, frame_valid)
    l_rigid = _near_rigid_transport(joints, obj_verts, T_obj,
                                    contact_per_joint, frame_valid)
    return l_dist + l_rigid


# ---------------------------------------------------------------------------
# Term 1: contact distance
# ---------------------------------------------------------------------------

def _contact_distance(
    joints:           torch.Tensor,   # (B, T, J, 3)
    obj_verts:        torch.Tensor,   # (B, T, V, 3)
    contact_per_joint: torch.Tensor,  # (B, T, J)  0/1
    frame_valid:      torch.Tensor | None,
) -> torch.Tensor:
    """Penalize distance between contacting joints and nearest object point."""
    nn_verts, _ = _nearest_obj_point(joints, obj_verts)       # (B, T, J, 3)
    dist = ((joints - nn_verts) ** 2).sum(-1)                  # (B, T, J)

    # Only penalize when in contact
    weighted = dist * contact_per_joint

    if frame_valid is not None:
        weighted = weighted * frame_valid.float().unsqueeze(-1)
        n = (contact_per_joint * frame_valid.float().unsqueeze(-1)).sum().clamp(min=1)
        return weighted.sum() / n
    n = contact_per_joint.sum().clamp(min=1)
    return weighted.sum() / n


# ---------------------------------------------------------------------------
# Term 2: near-rigid transport
# ---------------------------------------------------------------------------

def _near_rigid_transport(
    joints:           torch.Tensor,   # (B, T, J, 3)
    obj_verts:        torch.Tensor,   # (B, T, V, 3)
    T_obj:            torch.Tensor,   # (B, T, 4, 4)
    contact_per_joint: torch.Tensor,  # (B, T, J)
    frame_valid:      torch.Tensor | None,
) -> torch.Tensor:
    """Enforce that contact points follow the object's rigid motion.

    For consecutive contact frames (t, t+1):
        p_predicted = R^{t+1}(R^t)^T * p^t
        loss = ||p_predicted - p^{t+1}||^2

    where p^t is the nearest object point at frame t.
    """
    B, T, J, _ = joints.shape
    if T < 2:
        return joints.new_zeros(())

    # Nearest object point at every frame
    nn_t, _ = _nearest_obj_point(joints, obj_verts)             # (B, T, J, 3)

    # Relative object rotation: R^{t+1} (R^t)^T
    R_t   = T_obj[:, :-1, :3, :3]   # (B, T-1, 3, 3)
    R_tp1 = T_obj[:, 1:,  :3, :3]   # (B, T-1, 3, 3)
    R_rel = R_tp1 @ R_t.transpose(-1, -2)   # (B, T-1, 3, 3)

    # Predicted position at t+1: R_rel @ p^t  (broadcast over J)
    p_t   = nn_t[:, :-1]                   # (B, T-1, J, 3)
    p_tp1 = nn_t[:, 1:]                    # (B, T-1, J, 3)

    p_pred = (R_rel.unsqueeze(2) @ p_t.unsqueeze(-1)).squeeze(-1)  # (B, T-1, J, 3)

    err = ((p_pred - p_tp1) ** 2).sum(-1)   # (B, T-1, J)

    # Active only when both frames have contact
    both_contact = contact_per_joint[:, :-1] * contact_per_joint[:, 1:]

    weighted = err * both_contact
    if frame_valid is not None:
        valid_pair = (frame_valid[:, :-1] & frame_valid[:, 1:]).float()
        weighted   = weighted * valid_pair.unsqueeze(-1)
        n = (both_contact * valid_pair.unsqueeze(-1)).sum().clamp(min=1)
        return weighted.sum() / n
    n = both_contact.sum().clamp(min=1)
    return weighted.sum() / n


# ---------------------------------------------------------------------------
# Fallback: wrist-proximity version (no FK)
# ---------------------------------------------------------------------------

def _wrist_interaction(
    x0_pred:     torch.Tensor,
    obj_verts:   torch.Tensor,
    contact:     torch.Tensor,
    frame_valid: torch.Tensor | None,
) -> torch.Tensor:
    """Simplified interaction loss using wrist positions as hand proxy."""
    p = unpack_x0(x0_pred)
    # (B, T, 2, 3): left and right wrist positions
    wrists = torch.stack([p['left_transl'], p['right_transl']], dim=2)

    # Mean object vertex as proxy for nearest point
    obj_center = obj_verts.mean(dim=2, keepdim=True)   # (B, T, 1, 3)
    dist = ((wrists - obj_center) ** 2).sum(-1)        # (B, T, 2)

    weighted = dist * contact
    if frame_valid is not None:
        weighted = weighted * frame_valid.float().unsqueeze(-1)
        n = (contact * frame_valid.float().unsqueeze(-1)).sum().clamp(min=1)
        return weighted.sum() / n
    return weighted.sum() / contact.sum().clamp(min=1)
