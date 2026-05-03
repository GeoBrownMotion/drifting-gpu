"""Global paths for the public Drift release."""

from __future__ import annotations

import os

IMAGENET_PATH = "/datasets/imagenet"
IMAGENET_CACHE_PATH = "/home/om/latent_cache"
IMAGENET_FID_NPZ = "/nfs_share5/code/om/drifting/extras/imagenet_256_fid_stats.npz"
IMAGENET_PR_NPZ = "/nfs_share5/code/om/drifting/extras/imagenet_val_prc_arr0.npz"

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("HF_ROOT", "/home/om/.cache/huggingface/")
