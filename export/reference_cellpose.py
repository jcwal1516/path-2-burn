"""Download Cellpose-SAM's checkpoint and write torch references for its network.

`download` fetches `cpsam` unchanged from Hugging Face (mouseland/cellpose-sam, BSD-3) at a
pinned revision into WEIGHTS/cellpose_sam/ and lists it in WEIGHTS/manifest.json.

`reference` builds the network with the cellpose package, loads the checkpoint strictly
(Cellpose itself loads non-strictly), and writes under OUT/cellpose_sam/:

  rgb.bin      [N, 256, 256, 3] u8: two crops of each fixtures/tiles PNG at 0.5 um/px
  input.bin    [N, 3, 256, 256] float32: each tile through cellpose's normalize_img
  flows.bin    [N, 3, 256, 256] float32: the network's dY, dX and cell-probability logit
  meta.json    shapes, library versions and the checkpoint file

References run in float32 on the CPU.

Usage:
  .venv/bin/python reference_cellpose.py download --weights ../weights
  .venv/bin/python reference_cellpose.py reference --weights ../weights --out ../fixtures
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from cellpose import models, transforms

from common import checkpoint, download, fixture_crops, write_json

NAME = "cellpose_sam"
REPO = "mouseland/cellpose-sam"
REVISION = "7c61431b5fbb078f3296754bd15d9f51b320f837"
FILE = "cpsam"
SHA256 = "e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2"
TILE = 256
CORNERS = ((0, 0), (256, 256))


def load_network(path: Path) -> torch.nn.Module:
    # cellpose builds the network in bfloat16 by default; loading the float32
    # checkpoint into it would round every weight, so convert it first.
    net = models.CellposeModel(gpu=False, pretrained_model=str(path)).net.float()
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    result = net.load_state_dict(state, strict=False)
    # cellpose 4.2's diameter buffers, which cpsam does not carry and the network does not use.
    if result.unexpected_keys or set(result.missing_keys) - {"diam_labels", "diam_mean"}:
        raise SystemExit(f"cpsam does not match cellpose's network: {result}")
    return net.eval().float()


@torch.no_grad()
def reference(weights: Path, out: Path) -> None:
    path = checkpoint(weights, NAME)
    net = load_network(path)
    rgb, provenance = fixture_crops(TILE, CORNERS, reduce=2)
    inputs = np.stack([transforms.normalize_img(tile.astype(np.float32)) for tile in rgb])
    inputs = np.ascontiguousarray(inputs.transpose(0, 3, 1, 2))
    flows = np.concatenate([net(torch.from_numpy(inputs[i : i + 4]))[0].numpy() for i in range(0, len(inputs), 4)])
    folder = out / NAME
    folder.mkdir(parents=True, exist_ok=True)
    rgb.tofile(folder / "rgb.bin")
    inputs.astype("<f4").tofile(folder / "input.bin")
    flows.astype("<f4").tofile(folder / "flows.bin")
    meta = {
        "tiles": provenance,
        "input_shape": list(inputs.shape),
        "flows_shape": list(flows.shape),
        "checkpoint": path.name,
        "torch": torch.__version__,
        "cellpose": version("cellpose"),
    }
    write_json(folder / "meta.json", meta)
    print(f"{NAME}: {len(rgb)} tiles, flows {flows.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "reference"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        download(args.weights, NAME, REPO, REVISION, FILE, expected_sha256=SHA256)
    elif args.out is None:
        raise SystemExit("reference needs --out")
    else:
        reference(args.weights, args.out)


if __name__ == "__main__":
    main()
