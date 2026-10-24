"""Torch reference outputs for InstanSeg brightfield_nuclei on 1024x1024 tiles at 0.5 µm/px.

The reference is the ORIGINAL TorchScript model (UNet + its scripted postprocessing) run
on CPU with the fixtures' input contract: rgb_u8 / 255 in [1,3,H,W].

For each tile PNG writes, under OUT/<tile>/ (little-endian, C order):
  input.bin    float32 [1,3,1024,1024]
  raw.bin      float32 [1,5,1024,1024]   UNet output (fields x/y, sigma 0/1, seed)
  labels.bin   int32   [1024,1024]       full TorchScript forward, 0 = background
  seeds.bin    int32   [N,2]             (row, col) local maxima, as the model finds them
  meta.json    counts
plus OUT/timings.json (full model and UNet, CPU and MPS).

Usage (path-2-burn export venv):
  .venv/bin/python reference_instanseg.py --torchscript .../instanseg.pt --tiles ../fixtures/instanseg/tiles --out ../fixtures/instanseg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import read_rgb, tile_pngs, timed, write_json
from export_instanseg import load_scripted


def seeds(raw: torch.Tensor, peak_distance: int, threshold: float) -> torch.Tensor:
    mask_map = raw[0, -1] / 15 + 0.5
    h, w = mask_map.shape
    k = 2 * peak_distance + 1
    pooled, inds = F.max_pool2d(mask_map.view(1, 1, h, w), k, 1, peak_distance, return_indices=True)
    own = torch.arange(h * w).view(1, 1, h, w)
    return torch.nonzero(((inds == own) & (pooled > threshold)).squeeze()).to(torch.int32)


def unit_input(rgb: np.ndarray) -> torch.Tensor:
    """[1, 3, H, W] float32 in [0, 1], the input the fixtures were made with."""
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = load_scripted(args.torchscript)
    tiles = tile_pngs(args.tiles)

    for tile_path in tiles:
        x = unit_input(read_rgb(tile_path))
        with torch.no_grad():
            raw = model.fcn(x.clone())
            labels = model(x.clone())[0, 0].to(torch.int32)
        found = seeds(raw, int(model.default_peak_distance), float(model.default_seed_threshold))
        out = args.out / tile_path.stem
        out.mkdir(parents=True, exist_ok=True)
        x.numpy().astype("<f4").tofile(out / "input.bin")
        raw.numpy().astype("<f4").tofile(out / "raw.bin")
        labels.numpy().astype("<i4").tofile(out / "labels.bin")
        found.numpy().astype("<i4").tofile(out / "seeds.bin")
        ids = torch.unique(labels)
        meta = {"seeds": int(found.shape[0]), "instances": int((ids > 0).sum()), "foreground_px": int((labels > 0).sum())}
        write_json(out / "meta.json", meta, indent=None)
        print(f"{tile_path.stem}: {meta}", flush=True)

    x = unit_input(read_rgb(tiles[0]))
    timings = {}
    with torch.no_grad():
        timings["full_cpu_s"] = timed(lambda: model(x.clone()))
        timings["unet_cpu_s"] = timed(lambda: model.fcn(x.clone()))
        if torch.backends.mps.is_available():
            sync = torch.mps.synchronize
            xm = x.to("mps")
            fcn_mps = model.fcn.to("mps")
            timings["unet_mps_s"] = timed(lambda: fcn_mps(xm.clone()), sync)
            try:
                model_mps = model.to("mps")
                timings["full_mps_s"] = timed(lambda: model_mps(xm.clone()), sync)
            except Exception as exc:  # the scripted postprocessing is not guaranteed to run on MPS
                timings["full_mps_error"] = f"{type(exc).__name__}: {exc}"[:300]
    write_json(args.out / "timings.json", timings)
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
