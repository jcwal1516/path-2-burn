"""Download Cellpose-SAM's checkpoint and write torch references for its network.

`download` fetches `cpsam` unchanged from Hugging Face (mouseland/cellpose-sam, BSD-3) at a
pinned revision into WEIGHTS/cellpose_sam/ and lists it in WEIGHTS/manifest.json.

`reference` builds the network with the cellpose package, loads the checkpoint strictly
(Cellpose itself loads non-strictly), and writes under OUT/cellpose_sam/:

  rgb.bin      [N, 256, 256, 3] u8: two crops of each fixtures/tiles PNG at 0.5 um/px
  input.bin    [N, 3, 256, 256] float32: each tile through cellpose's normalize_img
  flows.bin    [N, 3, 256, 256] float32: the network's dY, dX and cell-probability logit
  meta.json    shapes, library versions and the checkpoint file

`masks --out OUT` reuses those flows and writes CPU mask references under
OUT/cellpose_sam/masks/, including synthetic and empty cases. --limit samples
the network tiles during development without rerunning the network.

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
from cellpose import dynamics, models, transforms

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


def mask_references(out: Path, limit: int | None) -> None:
    """Reuse the saved network flows; also exercise rectangular and empty inputs."""
    folder = out / NAME
    flows = np.fromfile(folder / "flows.bin", dtype="<f4").reshape(-1, 3, TILE, TILE)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        flows = flows[:limit]
    cases = [(f"tile_{i:02d}", flow) for i, flow in enumerate(flows)]
    labels = np.zeros((96, 112), dtype=np.uint16)
    y, x = np.indices(labels.shape)
    for label, (cy, cx, radius) in enumerate([(24, 24, 12), (52, 70, 15), (80, 90, 2), (62, 2, 9)], 1):
        labels[(y-cy)**2 + (x-cx)**2 < radius**2] = label
    labels[(y-52)**2 + (x-70)**2 < 4**2] = 0
    dP, _ = dynamics.masks_to_flows_gpu(labels, device=torch.device("cpu"))
    synthetic = np.concatenate([dP * 5, np.where(labels > 0, 5, -5)[None]]).astype(np.float32)
    cases.extend([("synthetic", synthetic), ("empty", np.zeros((3, 32, 48), np.float32))])
    for name, flow in cases:
        target = folder / "masks" / name
        target.mkdir(parents=True, exist_ok=True)
        unfiltered = dynamics.compute_masks(flow[:2], flow[2], flow_threshold=0, min_size=-1,
                                             device=torch.device("cpu"))
        masks = dynamics.compute_masks(flow[:2], flow[2], min_size=15, device=torch.device("cpu"))
        flow.astype("<f4").tofile(target / "flows.bin")
        unfiltered.astype("<u4").tofile(target / "unfiltered.bin")
        masks.astype("<u4").tofile(target / "masks.bin")
        write_json(target / "meta.json", {"height": flow.shape[1], "width": flow.shape[2],
                   "cellpose": version("cellpose"), "iterations": 200, "min_size": 15,
                   "cellprob_threshold": 0.0, "flow_threshold": 0.4, "max_size_fraction": 0.4})
        print(f"{name}: {int(unfiltered.max())} candidates -> {int(masks.max())} masks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "reference", "masks"])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, help="number of saved flow tiles for mask references")
    args = parser.parse_args()
    if args.command != "masks" and args.weights is None:
        parser.error("download and reference need --weights")
    if args.command == "download":
        download(args.weights, NAME, REPO, REVISION, FILE, expected_sha256=SHA256)
    elif args.out is None:
        parser.error("reference and masks need --out")
    elif args.command == "masks":
        mask_references(args.out, args.limit)
    else:
        reference(args.weights, args.out)


if __name__ == "__main__":
    main()
