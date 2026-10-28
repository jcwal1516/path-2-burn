"""Download TITAN and compare its slide encoder using CONCH v1.5 patch features.

References use the pinned upstream vision tower, loaded strictly in float32. Pass
an HDF5 file with features, coords, and coords.attrs['patch_size_level0']; the
upstream demo file has this format. --limit uses a small subset while developing.
Weights, input features and outputs stay in the ignored weights/ and fixtures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from safetensors.torch import load_file

from common import checkpoint, download, load_upstream_module, write_json

REPO = "MahmoodLab/TITAN"
REVISION = "dac6773d9961cfc75503440676ff157a2c6e8d2e"
EXTRA_FILES = ("config.json", "configuration_titan.py", "vision_transformer.py")


def build(folder: Path, weights: Path) -> torch.nn.Module:
    # The upstream vision module imports its configuration by relative name.
    upstream = load_upstream_module(folder, "vision_transformer")
    cfg = json.loads((folder / "config.json").read_text())["vision_config"]
    model = upstream.build_vision_tower(SimpleNamespace(**cfg)).float()
    state = load_file(weights)
    model.load_state_dict({k.removeprefix("vision_encoder."): v for k, v in state.items()
                           if k.startswith("vision_encoder.")}, strict=True)
    return model.eval()


@torch.inference_mode()
def reference(weights: Path, source: Path, out: Path, limit: int | None, device: str) -> None:
    path = checkpoint(weights, "titan")
    model = build(path.parent, path).to(device)
    with h5py.File(source) as h5:
        features = np.asarray(h5["features"], dtype=np.float32).reshape(-1, 768)
        coords = np.asarray(h5["coords"], dtype=np.int64).reshape(-1, 2)
        patch_size = int(h5["coords"].attrs["patch_size_level0"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        features, coords = features[:limit], coords[:limit]
    # A rectangular sparse grid, shuffled cells, a duplicate cell and an empty
    # feature explicitly exercise upstream's scatter-add and background removal.
    synthetic = features[:5].copy()
    if len(synthetic) != 5:
        raise ValueError("reference input needs at least five patches")
    synthetic[-1] = 0
    synthetic_coords = np.array([[2, 1], [0, 0], [2, 1], [1, 4], [3, 2]], dtype=np.int64)
    synthetic_coords = synthetic_coords * patch_size + np.array([17, 31])
    cases = [
        ("patches", features, coords),
        ("sparse_duplicates", synthetic, synthetic_coords),
        # A zero cell keeps the grid 2x2: upstream's NumPy indexing of a torch
        # bool mask is ambiguous for a 1x1 grid in recent NumPy versions.
        ("single_patch", np.concatenate([features[:1], np.zeros((1, 768), dtype=np.float32)]),
         np.array([[0, 0], [patch_size, patch_size]], dtype=np.int64)),
    ]
    for name, features, coords in cases:
        embedding = model(torch.from_numpy(features).to(device), torch.from_numpy(coords).to(device),
                          patch_size, no_proj=True).cpu().numpy()
        folder = out / "titan" / name
        folder.mkdir(parents=True, exist_ok=True)
        features.astype("<f4").tofile(folder / "features.bin")
        coords.astype("<i8").tofile(folder / "coords.bin")
        embedding.astype("<f4").tofile(folder / "embedding.bin")
        write_json(folder / "meta.json", {
            "repo": REPO, "revision": REVISION, "patch_size_lv0": patch_size,
            "patches": len(features), "embedding_shape": list(embedding.shape),
            "device": device, "torch": torch.__version__,
        })
        print(f"{name}: {len(features)} patches -> {embedding.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--weights", type=Path, required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--weights", type=Path, required=True)
    ref.add_argument("--input", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    ref.add_argument("--limit", type=int)
    ref.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.command == "download":
        download(args.weights, "titan", REPO, REVISION, "model.safetensors", EXTRA_FILES)
    else:
        reference(args.weights, args.input, args.out, args.limit, args.device)


if __name__ == "__main__":
    main()
