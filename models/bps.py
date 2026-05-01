"""Basis Point Set (BPS) encoding for object templates.

BPS [Prokudin et al. ICCV 2019] represents a 3D point cloud as a fixed-length
vector of squared distances from N fixed basis points to the nearest surface
point.  The basis points are sampled once on a unit sphere and reused across
all objects.

Two encodings are implemented:
  - encode_object: standard BPS, encodes a mesh template into a (N,) vector
  - encode_ambient: per-joint nearest-point displacements (Ambient Sensor),
    used as an auxiliary input to the denoiser to reason about hand-object
    contact proximity
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basis point sampling (deterministic, cached)
# ---------------------------------------------------------------------------

_BASIS: dict[int, torch.Tensor] = {}   # n_points -> (N, 3) on unit sphere


def get_basis_points(n_points: int = 1024, seed: int = 0) -> torch.Tensor:
    """Return N points uniformly sampled on the unit sphere (cached).

    Args:
        n_points: number of basis points
        seed:     RNG seed for reproducibility
    Returns:
        (N, 3) float32 tensor on CPU
    """
    if n_points in _BASIS:
        return _BASIS[n_points]

    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_points, 3)).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    basis = torch.tensor(pts)
    _BASIS[n_points] = basis
    return basis


# ---------------------------------------------------------------------------
# Object template BPS encoding
# ---------------------------------------------------------------------------

def encode_object(
    vertices:   torch.Tensor,   # (V, 3) mesh vertices in canonical frame
    n_points:   int = 1024,
    radius:     float = 0.2,    # scale basis sphere to object size
) -> torch.Tensor:
    """Encode an object mesh as a BPS descriptor.

    Computes the squared distance from each scaled basis point to the nearest
    mesh vertex.

    Args:
        vertices:  (V, 3) mesh vertices in canonical object frame
        n_points:  number of basis points
        radius:    radius of the basis sphere (should match object scale)
    Returns:
        (N,) float32 BPS descriptor — squared distances
    """
    basis = get_basis_points(n_points).to(vertices.device) * radius  # (N, 3)
    # Pairwise squared distances: (N, V)
    diff  = basis.unsqueeze(1) - vertices.unsqueeze(0)               # (N, V, 3)
    sq_d  = (diff ** 2).sum(-1)                                       # (N, V)
    return sq_d.min(dim=1).values                                     # (N,)


@torch.no_grad()
def encode_object_from_glb(
    glb_path:   str | Path,
    n_points:   int = 1024,
    radius:     float = 0.2,
    device:     str = 'cpu',
) -> torch.Tensor:
    """Load a .glb mesh and compute its BPS descriptor.

    Args:
        glb_path:  path to a HOT3D object model .glb file
        n_points:  BPS resolution
        radius:    basis sphere radius
        device:    target device
    Returns:
        (N,) BPS descriptor
    """
    import trimesh
    mesh = trimesh.load(str(glb_path), force='mesh')
    verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32).to(device)
    # Normalize to unit scale so radius is meaningful across objects
    center = verts.mean(0, keepdim=True)
    scale  = (verts - center).norm(dim=1).max().clamp(min=1e-6)
    verts_norm = (verts - center) / scale
    return encode_object(verts_norm, n_points=n_points, radius=radius)


# ---------------------------------------------------------------------------
# BPS descriptor cache (object_bop_id -> descriptor)
# ---------------------------------------------------------------------------

class ObjectBPSCache:
    """Lazy-loading cache of BPS descriptors for HOT3D objects.

    Args:
        models_dir:  directory containing obj_XXXXXX.glb files
        n_points:    BPS resolution
        radius:      basis sphere radius
        device:      where to store tensors
    """

    def __init__(
        self,
        models_dir: str | Path,
        n_points:   int = 1024,
        radius:     float = 0.2,
        device:     str = 'cpu',
    ):
        self.models_dir = Path(models_dir)
        self.n_points   = n_points
        self.radius     = radius
        self.device     = device
        self._cache:    dict[str, torch.Tensor] = {}

    def get(self, bop_id: str) -> torch.Tensor:
        """Return the (N,) BPS descriptor for the given object BOP ID.

        Args:
            bop_id: HOT3D object BOP ID (e.g. '1', '5')
        Returns:
            (N,) float32 BPS descriptor
        """
        if bop_id in self._cache:
            return self._cache[bop_id]

        glb = self.models_dir / f'obj_{int(bop_id):06d}.glb'
        if not glb.exists():
            # Zero descriptor as fallback; logged once
            import warnings
            warnings.warn(f'Object model not found: {glb}')
            desc = torch.zeros(self.n_points, dtype=torch.float32, device=self.device)
        else:
            desc = encode_object_from_glb(glb, self.n_points, self.radius, self.device)

        self._cache[bop_id] = desc
        return desc

    def get_batch(self, bop_ids: list[str]) -> torch.Tensor:
        """Return (B, N) BPS descriptors for a list of BOP IDs."""
        return torch.stack([self.get(bid) for bid in bop_ids])


# ---------------------------------------------------------------------------
# Ambient Sensor: hand-joint to object nearest-point displacements
# ---------------------------------------------------------------------------

def compute_ambient_sensor(
    joints:    torch.Tensor,   # (B, T, J, 3) hand joint positions in local frame
    obj_verts: torch.Tensor,   # (B, T, V, 3) posed object vertices in local frame
) -> torch.Tensor:
    """Compute per-joint displacement to the nearest object surface point.

    For each hand joint j at each frame t, finds the nearest vertex on the
    posed object mesh and returns the displacement vector.

    This is the "Ambient Sensor" feature from Sec. 3.1, equivalent to
    BPS_J(T_i[O]) using hand joints as basis.

    Args:
        joints:    (B, T, J, 3) joint world positions
        obj_verts: (B, T, V, 3) posed object vertex positions
    Returns:
        (B, T, J, 3) displacement vectors (nearest_vertex - joint)
    """
    B, T, J, _ = joints.shape
    V = obj_verts.shape[2]

    # Expand for broadcast: (B, T, J, 1, 3) vs (B, T, 1, V, 3)
    diff = obj_verts.unsqueeze(2) - joints.unsqueeze(3)   # (B, T, J, V, 3)
    sq_d = (diff ** 2).sum(-1)                             # (B, T, J, V)
    nn_idx = sq_d.argmin(-1)                               # (B, T, J)

    # Gather nearest vertex positions
    idx_exp = nn_idx.unsqueeze(-1).unsqueeze(-1).expand(B, T, J, 1, 3)
    nn_verts = obj_verts.unsqueeze(2).expand(B, T, J, V, 3).gather(3, idx_exp).squeeze(3)
    return nn_verts - joints                               # (B, T, J, 3)


def compute_ambient_sensor_approx(
    joints:    torch.Tensor,  # (B, T, J, 3)
    obj_verts: torch.Tensor,  # (B, T, V, 3)
    subsample: int = 128,
) -> torch.Tensor:
    """Memory-efficient approximate Ambient Sensor using vertex subsampling.

    Randomly subsamples object vertices to keep memory under control
    for large V (778 HOT3D vertices × T=120 frames × B samples).

    Args:
        joints:    (B, T, J, 3)
        obj_verts: (B, T, V, 3)
        subsample: number of vertices to subsample
    Returns:
        (B, T, J, 3)
    """
    V = obj_verts.shape[2]
    if V <= subsample:
        return compute_ambient_sensor(joints, obj_verts)

    # Use the same random subset for all (B, T) to keep gradients consistent
    idx = torch.randperm(V, device=obj_verts.device)[:subsample]
    return compute_ambient_sensor(joints, obj_verts[:, :, idx, :])
