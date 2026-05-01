# External Dependencies

Assets and tools that live **outside** this repository and must be set up manually.

---

## 1. MANO Hand Model

**What:** Parametric hand model used for forward kinematics.  
**License:** Requires free registration at https://mano.is.tue.mpg.de  
**Files needed:**
```
MANO_RIGHT.pkl   # right hand model
MANO_LEFT.pkl    # left hand model (register separately)
```

**Current status:**
- `MANO_RIGHT.pkl` available at:
  `/scr/cezhao/workspace/HOI_recon/hamer/_DATA/data/mano/MANO_RIGHT.pkl`
- `MANO_LEFT.pkl` — **not yet obtained** (needs registration/download)

**Setup:**
Because `smplx` requires `chumpy` which is incompatible with Python 3.11+,
we convert the `.pkl` files to a chumpy-free `.npz` format once:

```bash
# Run in hawor_h200 env (Python 3.10, has chumpy)
/scr/cezhao/workspace/HOI_recon/.conda_envs/hawor_h200/bin/python \
    scripts/convert_mano_pkl.py \
    --mano_dir /scr/cezhao/workspace/HOI_recon/hamer/_DATA/data/mano \
    --out_dir   /scr/cezhao/workspace/HOI_recon/_DATA/mano
```

Output `.npz` files are stored at `/scr/cezhao/workspace/HOI_recon/_DATA/mano/`
and are referenced by `data/mano_converter.py` (`MANO_NPZ_DIR`).

**Current status of npz:**
- `MANO_RIGHT.npz` — ✅ converted
- `MANO_LEFT.npz`  — ⏳ pending (need MANO_LEFT.pkl first)

---

## 2. HOT3D Dataset

**What:** Training and evaluation data (egocentric HOI videos).  
**Source:** https://huggingface.co/datasets/bop-benchmark/hot3d  
**License:** HOT3D dataset license (see `_DATA/hot3d/hot3d_dataset_license_agreement.pdf`)

**Download status** (`/scr/cezhao/workspace/HOI_recon/_DATA/hot3d/`):

| Split | Clips | Size | Status |
|-------|-------|------|--------|
| `test_aria`    | 467   | 44 GB  | ✅ Done |
| `train_aria`   | 1,516 | 163 GB | ✅ Done |
| `test_quest3`  | 561   | 57 GB  | ✅ Done |
| `train_quest3` | 1,288 | 131 GB | ✅ Done |
| `object_models` + metadata | — | ~0.3 GB | ✅ Done |

**Download script:** `_DATA/download_hot3d.sh`

---

## 3. HOT3D Toolkit

**What:** Python utilities for reading HOT3D data (MANO FK, data loaders).  
**Source:** https://github.com/facebookresearch/hot3d  
**Note:** Not a pip-installable package; cloned locally.

**Setup:**
```bash
git clone --depth 1 https://github.com/facebookresearch/hot3d.git \
    /scr/cezhao/workspace/HOI_recon/hot3d_toolkit
```

**Location:** `/scr/cezhao/workspace/HOI_recon/hot3d_toolkit/`  
**Usage in code:** `sys.path.insert(0, '.../hot3d_toolkit')` — see `data/mano_converter.py`

---

## 4. HaWoR (World-Grounded Hand Motion Estimator)

**What:** Off-the-shelf estimator used at test time to produce the noisy
hand conditioning H̃. Reference: Zhang et al., CVPR 2025 [73].  
**Source:** https://github.com/ThunderVVV/HaWoR (check paper project page)  
**Conda env:** `/scr/cezhao/workspace/HOI_recon/.conda_envs/hawor_h200`

**Status:** Environment set up; wrapper `hand_estimator/hawor_wrapper.py` — ⏳ pending

---

## 5. Python Environment

Main development uses the **base** conda environment (Python 3.13):
```bash
pip install webdataset smplx einops trimesh opencv-python openai
```

For tasks requiring chumpy (MANO pkl conversion):
```bash
/scr/cezhao/workspace/HOI_recon/.conda_envs/hawor_h200/bin/python ...
```

---

## 6. Object Templates

**What:** 3D mesh files (.glb) for the 33 HOT3D objects — used for BPS encoding.  
**Location:** `/scr/cezhao/workspace/HOI_recon/_DATA/hot3d/object_models/`  
**Format:** `.glb` (loadable with `trimesh.load`)  
**Status:** ✅ Downloaded (34 files, ~175 MB)
