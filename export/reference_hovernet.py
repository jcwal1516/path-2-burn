"""Torch references for HoVer-Net's published checkpoints.

The checkpoints come from vqdang/hover_net's Google Drive (README, "Inference"). Download
one, list it in WEIGHTS/manifest.json under its name below with its sha256, then run
`reference`. It builds hover_net's own network (vendored in hovernet_ref/), loads the
checkpoint strictly, and writes under OUT/<name>/:

  rgb.bin      [N, S, S, 3] u8: two crops of each fixtures/tiles PNG at 0.25 um/px (40x),
               S = 256 (fast mode) or 270 (original mode)
  np.bin       [N, 2, O, O] float32 nucleus logits, O = 164 (fast) or 80 (original)
  hv.bin       [N, 2, O, O] float32 horizontal/vertical distances
  tp.bin       [N, T, O, O] float32 type logits, for checkpoints with types
  meta.json    shapes, versions and the checkpoint file

References run in float32 on the CPU.

Usage:
  .venv/bin/python reference_hovernet.py --weights ../weights --out ../fixtures hovernet_pannuke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import checkpoint, fixture_crops, write_json
from hovernet_ref import HoVerNet

# name: (mode, nr_types), matching src/hovernet.rs.
VARIANTS = {
    "hovernet_pannuke": ("fast", 6),
    "hovernet_monusac": ("fast", 5),
    "hovernet_consep": ("original", 5),
    "hovernet_consep_notype": ("original", None),
    "hovernet_kumar": ("original", None),
    "hovernet_cpm17": ("original", None),
}
INPUT = {"fast": 256, "original": 270}


def load_network(path: Path, mode: str, types: int | None) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)["desc"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    net = HoVerNet(nr_types=types, mode=mode)
    net.load_state_dict(state, strict=True)
    return net.eval()


@torch.no_grad()
def reference(weights: Path, out: Path, name: str) -> None:
    mode, types = VARIANTS[name]
    path = checkpoint(weights, name)
    net = load_network(path, mode, types)
    size = INPUT[mode]
    rgb, provenance = fixture_crops(size, ((0, 0), (512, 512)))
    outputs: dict[str, list[np.ndarray]] = {"np": [], "hv": [], "tp": []}
    for tile in rgb:
        x = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)[None])).float()
        for branch, value in net(x).items():
            outputs[branch].append(value.numpy())
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    rgb.tofile(folder / "rgb.bin")
    shapes = {}
    for branch, values in outputs.items():
        if values:
            array = np.concatenate(values).astype("<f4")
            array.tofile(folder / f"{branch}.bin")
            shapes[branch] = list(array.shape)
    meta = {"mode": mode, "types": types, "tiles": provenance, "shapes": shapes, "checkpoint": path.name, "torch": torch.__version__}
    write_json(folder / "meta.json", meta)
    print(f"{name}: {len(rgb)} tiles, {shapes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("names", nargs="+", choices=sorted(VARIANTS))
    args = parser.parse_args()
    for name in args.names:
        reference(args.weights, args.out, name)


if __name__ == "__main__":
    main()
