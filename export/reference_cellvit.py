"""Torch reference outputs for CellViT (SAM-H or ViT-256 backbone) on fixture tiles.

For each tile PNG writes, under OUT/<tile>/, with N the variant's input size (the tile's
top-left N x N pixels when the PNG is larger):
  input.bin               float32 [1,3,N,N] normalized input
  nuclei_binary_map.bin   float32 [1,2,N,N]
  hv_map.bin              float32 [1,2,N,N]
  nuclei_type_map.bin     float32 [1,6,N,N]
  tissue_types.bin        float32 [1,19]
  instance_map.bin        int32   [N,N]  HoVer-Net watershed result
  cells.json              per-instance centroid/bbox/contour/type/type_prob
All .bin files are little-endian, C order. Also records CPU and MPS forward timings.

Usage:
  .venv/bin/python reference_cellvit.py --checkpoint CKPT --tiles ../fixtures/tiles --out ../fixtures/cellvit
  .venv/bin/python reference_cellvit.py --variant vit256 --checkpoint CKPT --tiles ../fixtures/tiles --out ../fixtures/cellvit_256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cellvit_ref.cellvit_model import CellViT, CellViTSAM
from common import read_rgb, sha256, tile_pngs, timed, write_json
from cellvit_ref.postprocess import DetectionCellPostProcessor

DEFAULTS = Path(__file__).parent / "cellvit_ref" / "cellvit_model" / "defaults"
CONFIGS = {"sam_h": DEFAULTS / "cellvit_sam_h.json", "vit256": DEFAULTS / "cellvit_256.json"}
OUTPUT_NAMES = ["nuclei_binary_map", "hv_map", "nuclei_type_map", "tissue_types"]


class OutputTuple(torch.nn.Module):
    """Returns the four PanNuke outputs in OUTPUT_NAMES order."""

    def __init__(self, model: CellViT) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return tuple(out[name] for name in OUTPUT_NAMES)


def load_model(checkpoint: Path, config: dict) -> CellViT:
    """The variant's network with the checkpoint's weights, loaded strictly. The checkpoint
    must hash to the config's checkpoint_sha256."""
    actual = sha256(checkpoint)
    expected = config["checkpoint_sha256"]
    if expected is None:
        raise SystemExit(f"record this checkpoint's sha256 ({actual}) as checkpoint_sha256 in the variant's defaults JSON")
    if actual != expected:
        raise SystemExit(f"checkpoint sha256 {actual} is not {expected}")
    # weights_only=True refuses arbitrary pickled objects; mmap skips optimizer state.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state_dict = state.get("model_state_dict", state)
    classes = {"num_nuclei_classes": config["num_nuclei_classes"], "num_tissue_classes": config["num_tissue_classes"]}
    if config["vit_structure"] == "ViT256":
        # CellViT256: HIPT's ViT-S/16.
        model = CellViT(
            **classes, embed_dim=384, input_channels=3, depth=12, num_heads=6, extract_layers=[3, 6, 9, 12]
        )
    else:
        model = CellViTSAM(model_path="unused", **classes, vit_structure=config["vit_structure"])
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def normalize(tile: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    x = tile.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(binary: np.ndarray, hv: np.ndarray, types: np.ndarray, nr_types: int):
    type_pred = np.argmax(types, axis=0).astype(np.int32)
    shifted = binary - binary.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    nuclei_prob = exp[1] / exp.sum(axis=0)
    pred_map = np.stack([type_pred, nuclei_prob, hv[0], hv[1]], axis=-1)
    processor = DetectionCellPostProcessor(nr_types=nr_types, magnification=40)
    instance_map, info = processor.post_process_cell_segmentation(pred_map)
    cells = [
        {
            "id": int(inst_id),
            "centroid": [float(v) for v in item["centroid"]],
            "bbox": [[int(v) for v in row] for row in item["bbox"]],
            "contour": [[int(p[0]), int(p[1])] for p in item["contour"]],
            "type": int(item["type"]),
            "type_prob": float(item["type_prob"]),
        }
        for inst_id, item in info.items()
    ]
    return instance_map.astype(np.int32), cells


@torch.no_grad()
def time_forward(model: torch.nn.Module, x: torch.Tensor, device: str, repeats: int) -> float:
    model, x = model.to(device), x.to(device)
    return timed(lambda: model(x), torch.mps.synchronize if device == "mps" else None, repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(CONFIGS), default="sam_h")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIGS[args.variant].read_text())
    size = config["input_size"]
    model = OutputTuple(load_model(args.checkpoint, config)).eval()
    tiles = tile_pngs(args.tiles)

    for tile_path in tiles:
        tile = read_rgb(tile_path)[:size, :size]
        x = normalize(tile, config["normalize_mean"], config["normalize_std"])
        with torch.no_grad():
            outputs = [t.numpy() for t in model(torch.from_numpy(x))]
        out_dir = args.out / tile_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        x.astype("<f4").tofile(out_dir / "input.bin")
        for name, value in zip(OUTPUT_NAMES, outputs):
            value.astype("<f4").tofile(out_dir / f"{name}.bin")
        instance_map, cells = postprocess(outputs[0][0], outputs[1][0], outputs[2][0], config["num_nuclei_classes"])
        instance_map.astype("<i4").tofile(out_dir / "instance_map.bin")
        write_json(out_dir / "cells.json", cells, indent=None)
        print(f"{tile_path.stem}: {len(cells)} cells, tissue argmax {int(outputs[3].argmax())}", flush=True)

    first = read_rgb(tiles[0])[:size, :size]
    sample = torch.from_numpy(normalize(first, config["normalize_mean"], config["normalize_std"]))
    timings = {"cpu_fp32_s": time_forward(model, sample, "cpu", 1)}
    if torch.backends.mps.is_available():
        timings["mps_fp32_s"] = time_forward(model, sample, "mps", 5)
    write_json(args.out / "timings.json", timings)
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
