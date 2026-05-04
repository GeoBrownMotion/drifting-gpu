"""Check cached ImageNet latent .pt files for NaN/Inf values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def _check_file(path: Path) -> list[str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    failures = []
    for key in ("moments", "moments_flip"):
        if key not in data:
            failures.append(f"{key}:missing")
            continue
        arr = np.asarray(data[key])
        if not np.all(np.isfinite(arr)):
            bad = np.argwhere(~np.isfinite(arr))
            first = tuple(int(i) for i in bad[0]) if bad.size else ()
            value = arr[first] if first else arr
            failures.append(f"{key}:nonfinite index={first} value={value}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Latent cache root containing train/ and val/.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of files to scan; 0 scans all.")
    parser.add_argument("--max-report", type=int, default=20, help="Maximum bad files to print.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = sorted(root.rglob("*.pt"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print(f"[ERROR] No .pt files found under {root}")
        raise SystemExit(1)

    bad_files = []
    for path in tqdm(paths, desc="latent-cache finite check", unit="file"):
        failures = _check_file(path)
        if failures:
            bad_files.append((path, failures))
            if len(bad_files) <= args.max_report:
                print(f"[BAD] {path}: {'; '.join(failures)}", flush=True)

    print(f"checked={len(paths)} bad={len(bad_files)} root={root}")
    if bad_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
