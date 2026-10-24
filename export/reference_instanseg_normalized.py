"""instanseg-package reference for percentile-normalized InstanSeg input.

Runs the instanseg package itself (not our port) with its defaults: normalise=True,
the model's own thresholds, CPU. Images are already at the model's 0.5 µm/px, so
pixel_size is left unset and nothing is rescaled.

  * every tile PNG in --tiles:  eval_small_image  -> OUT/<tile>/labels_pnorm.bin
  * every PNG in --regions:     eval_medium_image(tile_size=1024)
                                -> OUT/region/<name>/labels_pnorm.bin, centroids_pnorm.json

Each output directory also gets stats_pnorm.json: the per-channel 0.1/99.9 percentiles
percentile_normalize used, in 0–255 units.

The export venv has no instanseg; run with a Python that has instanseg-torch 0.1.1:
  python reference_instanseg_normalized.py \\
      --torchscript .../brightfield_nuclei/0.1.1/instanseg.pt \\
      --tiles ../fixtures/instanseg/tiles --regions ../fixtures/instanseg/region --out ../fixtures/instanseg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import read_rgb, tile_pngs, write_json
from export_instanseg import load_scripted

def load_model(path: Path):
    from instanseg import InstanSeg  # pyright: ignore[reportMissingImports]  (instanseg-torch only)

    return InstanSeg(load_scripted(path), device="cpu", verbosity=0)


def stats(rgb: np.ndarray) -> dict:
    # Same call percentile_normalize makes on the float32 image, per channel.
    img = rgb.astype(np.float32)
    lows, highs = [], []
    for c in range(3):
        lo, hi = np.percentile(img[:, :, c], [0.1, 99.9])
        lows.append(float(lo))
        highs.append(float(hi))
    return {"low": lows, "high": highs}


def centroids(labels: np.ndarray) -> list[list[float]]:
    ids = labels.ravel()
    keep = ids > 0
    flat = np.nonzero(keep)[0]
    ids = ids[keep]
    ys, xs = np.divmod(flat, labels.shape[1])
    order = np.argsort(ids, kind="stable")
    ids, ys, xs = ids[order], ys[order], xs[order]
    _, start, counts = np.unique(ids, return_index=True, return_counts=True)
    sx = np.add.reduceat(xs.astype(np.float64), start)
    sy = np.add.reduceat(ys.astype(np.float64), start)
    return [[float(x), float(y), int(n)] for x, y, n in zip(sx / counts, sy / counts, counts)]


def write(out: Path, labels: np.ndarray, rgb: np.ndarray, with_centroids: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    labels.astype("<i4").tofile(out / "labels_pnorm.bin")
    meta = {"width": int(labels.shape[1]), "height": int(labels.shape[0]), **stats(rgb)}
    meta["instances"] = int(len(np.unique(labels)) - (1 if (labels == 0).any() else 0))
    write_json(out / "stats_pnorm.json", meta)
    if with_centroids:
        write_json(out / "centroids_pnorm.json", centroids(labels), indent=None)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = load_model(args.torchscript)
    for path in tile_pngs(args.tiles):
        rgb = read_rgb(path)
        with torch.no_grad():
            labels = model.eval_small_image(rgb, pixel_size=None, normalise=True, return_image_tensor=False)
        labels = labels.squeeze().numpy().astype(np.int32)
        meta = write(args.out / path.stem, labels, rgb, with_centroids=False)
        print(f"{path.stem}: {meta['instances']} instances, low {meta['low']}, high {meta['high']}", flush=True)

    for path in sorted(args.regions.glob("*.png")) if args.regions else []:
        rgb = read_rgb(path)
        with torch.no_grad():
            labels = model.eval_medium_image(
                rgb, pixel_size=None, normalise=True, tile_size=1024, return_image_tensor=False
            )
        labels = labels.squeeze().numpy().astype(np.int32)
        meta = write(args.out / "region" / path.stem, labels, rgb, with_centroids=True)
        print(f"region {path.stem}: {meta['instances']} instances, low {meta['low']}, high {meta['high']}", flush=True)


if __name__ == "__main__":
    main()
