"""pytest configuration and shared fixtures."""

import pytest


def _can_load_mano_pkl() -> bool:
    """Return True if smplx can load MANO .pkl files (requires chumpy)."""
    try:
        import smplx, torch
        from pathlib import Path
        pkl = Path('/scr/cezhao/workspace/HOI_recon/hamer/_DATA/data/mano/MANO_RIGHT.pkl')
        if not pkl.exists():
            return False
        smplx.MANOLayer(str(pkl), use_pca=True, num_pca_comps=15,
                         is_rhand=True, flat_hand_mean=True)
        return True
    except Exception:
        return False


# Evaluated once at collection time
CAN_LOAD_MANO_PKL = _can_load_mano_pkl()

requires_chumpy = pytest.mark.skipif(
    not CAN_LOAD_MANO_PKL,
    reason='smplx cannot load MANO pkl (chumpy missing — run in hawor_h200 env)',
)
