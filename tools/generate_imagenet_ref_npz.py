"""Generate ImageNet reference files used by Drift FID/PR evaluation.

This writes:
  - FID stats: npz with keys `mu` and `sigma`
  - PR reference: npz with default key `arr_0` containing uint8 NHWC images

Example:
  python tools/generate_imagenet_ref_npz.py \
    --imagenet-path /data/imagenet \
    --fid-out /data/refs/imagenet_256_fid_stats.npz \
    --pr-out /data/refs/imagenet_val_prc_arr0.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer all visible CUDA devices for this offline reference generation tool.
# This must be set before importing JAX to avoid slow TPU initialization probes.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset.dataset import center_crop_arr
from utils.fid_util import _build_jax_inception, _revert_pmap_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-path", required=True, help="ImageNet root containing val/.")
    parser.add_argument("--fid-out", required=True, help="Output .npz path for FID mu/sigma.")
    parser.add_argument("--pr-out", required=True, help="Output .npz path for PR reference images.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--pr-count", type=int, default=10000)
    parser.add_argument("--max-images", type=int, default=0, help="Limit val images for a smoke test; 0 uses all images.")
    parser.add_argument("--loader-batch-size", type=int, default=512)
    parser.add_argument("--inception-per-device-batch", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


class CenterCropUint8:
    def __init__(self, resolution: int):
        self.resolution = resolution

    def __call__(self, image: Image.Image) -> np.ndarray:
        image = center_crop_arr(image.convert("RGB"), self.resolution)
        return np.asarray(image, dtype=np.uint8).copy()


def build_loader(imagenet_path: str, resolution: int, batch_size: int, num_workers: int, pin_memory: bool):
    dataset = ImageFolder(
        root=str(Path(imagenet_path) / "val"),
        transform=CenterCropUint8(resolution),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )


def inception_features(images_nhwc: np.ndarray, inception_net: dict, per_device_batch: int) -> np.ndarray:
    """Run the repo's FID Inception model on uint8 NHWC images."""
    local_devices = jax.local_device_count()
    full_batch = local_devices * per_device_batch
    valid = images_nhwc.shape[0]
    pad = int(np.ceil(valid / full_batch)) * full_batch - valid
    if pad:
        images_nhwc = np.concatenate(
            [images_nhwc, np.zeros((pad, *images_nhwc.shape[1:]), dtype=np.uint8)],
            axis=0,
        )

    feats = []
    for start in range(0, len(images_nhwc), full_batch):
        batch = images_nhwc[start : start + full_batch].astype(np.float32)
        # resize.forward consumes BCHW and returns BCHW.
        from utils.jax_fid import resize

        batch = torch.from_numpy(batch.transpose(0, 3, 1, 2))
        batch = resize.forward(batch).numpy().transpose(0, 2, 3, 1)
        batch = batch.reshape((local_devices, per_device_batch, *batch.shape[1:]))
        pooled, _, _ = inception_net["fn"](inception_net["params"], jax.lax.stop_gradient(batch))
        feats.append(np.asarray(_revert_pmap_shape(pooled)))

    return np.concatenate(feats, axis=0)[:valid]


def main() -> None:
    args = parse_args()
    fid_out = Path(args.fid_out)
    pr_out = Path(args.pr_out)
    fid_out.parent.mkdir(parents=True, exist_ok=True)
    pr_out.parent.mkdir(parents=True, exist_ok=True)

    loader = build_loader(
        args.imagenet_path,
        args.resolution,
        args.loader_batch_size,
        args.num_workers,
        args.pin_memory,
    )
    total = len(loader.dataset)
    max_images = total if args.max_images <= 0 else min(args.max_images, total)
    if args.pr_count > max_images:
        raise ValueError(f"--pr-count={args.pr_count} exceeds selected image count {max_images}.")

    inception_net = _build_jax_inception(batch_size=args.inception_per_device_batch)
    devices = jax.local_devices()
    effective_batch = len(devices) * args.inception_per_device_batch
    print(f"JAX devices ({len(devices)}): {[str(device) for device in devices]}", flush=True)
    print(
        f"Inception batch: {len(devices)} devices x {args.inception_per_device_batch} = {effective_batch} images",
        flush=True,
    )

    all_features = []
    pr_images = []
    seen = 0

    pbar = tqdm(total=max_images, desc="ImageNet val refs", unit="img")
    for images, _ in loader:
        images = images.numpy().astype(np.uint8)
        remaining = max_images - seen
        if remaining <= 0:
            break
        images = images[:remaining]

        all_features.append(inception_features(images, inception_net, args.inception_per_device_batch))

        remaining_pr = args.pr_count - seen
        if remaining_pr > 0:
            pr_images.append(images[:remaining_pr])
        seen += images.shape[0]
        pbar.update(images.shape[0])
    pbar.close()

    features = np.concatenate(all_features, axis=0).astype(np.float64)
    np.savez(fid_out, mu=np.mean(features, axis=0), sigma=np.cov(features, rowvar=False))

    pr_arr = np.concatenate(pr_images, axis=0)[: args.pr_count].astype(np.uint8)
    np.savez(pr_out, pr_arr)

    print(f"Wrote FID stats: {fid_out} keys=(mu, sigma), features={features.shape}")
    print(f"Wrote PR refs:   {pr_out} key=arr_0, images={pr_arr.shape} dtype={pr_arr.dtype}")


if __name__ == "__main__":
    main()
