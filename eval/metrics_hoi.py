"""HOI interaction quality metrics (Table 3).

From the supplementary:
  "For HOI ADD/ADD-S, we align predictions globally (as in WA-MPJPE)
   and then evaluate object error in this aligned space."

Procedure:
  1. Estimate global similarity transform from predicted → GT HAND joints
     (using all frames, i.e. WA-MPJPE style)
  2. Apply that same transform to the predicted OBJECT trajectory
  3. Compute ADD and ADD-S between transformed predicted object and GT object

This measures whether the predicted hand-object RELATIVE motion is correct,
independent of global coordinate frame drift.
"""

from __future__ import annotations

import numpy as np

from eval.alignment     import global_align, umeyama, apply_similarity
from eval.metrics_object import add_per_frame, add_s_per_frame, compute_auc


def compute_hoi_metrics(
    pred_joints:  np.ndarray,   # (T, J, 3)  predicted hand joints (metres)
    gt_joints:    np.ndarray,   # (T, J, 3)  GT hand joints
    T_pred_obj:   np.ndarray,   # (T, 4, 4)  predicted object poses
    T_gt_obj:     np.ndarray,   # (T, 4, 4)  GT object poses
    verts:        np.ndarray,   # (V, 3)     canonical mesh vertices
    auc_threshold: float = 0.3,
) -> dict[str, float]:
    """Compute HOI-aligned ADD / ADD-S and AUC.

    Args:
        pred_joints:  predicted hand joints (bimanual, any J)
        gt_joints:    GT hand joints
        T_pred_obj:   predicted object SE(3) per frame
        T_gt_obj:     GT object SE(3) per frame
        verts:        canonical object mesh vertices (normalised, metres)
        auc_threshold: AUC upper limit
    Returns:
        dict with 'HOI_ADD', 'HOI_ADD-S', 'HOI_AUC_ADD', 'HOI_AUC_ADD-S'
    """
    # 1. Estimate global hand alignment (WA-MPJPE style: all frames)
    src = pred_joints.reshape(-1, 3)
    dst = gt_joints.reshape(-1, 3)
    s, R, t = umeyama(src, dst)

    # 2. Apply same transform to predicted object poses
    #    For SE(3): T_aligned = T_hand_align @ T_pred_obj
    #    where T_hand_align = [[sR, t], [0, 1]]
    T_align = np.eye(4)
    T_align[:3, :3] = s * R
    T_align[:3,  3] = t

    T_pred_obj_aligned = np.einsum('ij,tjk->tik', T_align, T_pred_obj)

    # 3. Compute ADD / ADD-S in aligned space
    T = T_pred_obj.shape[0]
    add_vals  = np.empty(T)
    adds_vals = np.empty(T)

    for i in range(T):
        add_vals[i]  = add_per_frame( T_pred_obj_aligned[i], T_gt_obj[i], verts)
        adds_vals[i] = add_s_per_frame(T_pred_obj_aligned[i], T_gt_obj[i], verts)

    return {
        'HOI_ADD':      float(add_vals.mean()),
        'HOI_ADD-S':    float(adds_vals.mean()),
        'HOI_AUC_ADD':  compute_auc(add_vals,  auc_threshold),
        'HOI_AUC_ADD-S': compute_auc(adds_vals, auc_threshold),
    }
