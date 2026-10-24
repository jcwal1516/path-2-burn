"""Extract InstanSeg brightfield_nuclei's weights for path-2-burn.

The published model is TorchScript, a UNet plus data-dependent postprocessing. path-2-burn
runs the UNet in Burn and the postprocessing in Rust, so this script, run on your own copy of
instanseg.pt, writes the two parts they read and lists both in WEIGHTS/manifest.json:

  WEIGHTS/instanseg_net.safetensors     the UNet's state dict (InstanSeg_UNet names),
                                        checked to reproduce the scripted fcn first
  WEIGHTS/instanseg_postprocess.json    thresholds + the 4->5->5->1 pixel classifier

Usage (path-2-burn export venv):
  .venv/bin/python export_instanseg.py --torchscript .../brightfield_nuclei/0.1.1/instanseg.pt --out ../weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from common import register, sha256, write_json
from instanseg_ref.instanseg_unet import InstanSeg_UNet

EXPECTED_SHA256 = "0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0"  # brightfield_nuclei 0.1.1
UNET_NAME = "instanseg_net"
POSTPROCESS_NAME = "instanseg_postprocess"


def load_scripted(path: Path) -> torch.jit.ScriptModule:
    actual = sha256(path)
    if actual != EXPECTED_SHA256:
        raise SystemExit(f"instanseg.pt sha256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
    return torch.jit.load(str(path), map_location="cpu").eval()


def eager_unet(scripted: torch.jit.ScriptModule) -> InstanSeg_UNet:
    if scripted.cells_and_nuclei or scripted.dim_seeds != 1:
        raise SystemExit("only single-output brightfield models (dim_seeds=1, nuclei only) are supported")
    out_channels = [[scripted.dim_coords, scripted.n_sigma, scripted.dim_seeds]]
    unet = InstanSeg_UNet(in_channels=3, out_channels=out_channels, layers=[256, 128, 64, 32], norm="BATCH", act="ReLU")
    # The scripted fcn also carries a copy of the pixel classifier; forward() uses the
    # top-level one, so make sure they agree and keep only the UNet weights.
    state = scripted.fcn.state_dict()
    top_level = scripted.pixel_classifier.state_dict()
    for key in [k for k in state if k.startswith("pixel_classifier.")]:
        if not torch.equal(state[key], top_level[key.removeprefix("pixel_classifier.")]):
            raise SystemExit(f"fcn.{key} differs from the top-level pixel classifier")
        del state[key]
    unet.load_state_dict(state, strict=True)
    unet.eval()
    probe = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        diff = (unet(probe) - scripted.fcn(probe)).abs().max().item()
    if diff > 1e-5:
        raise SystemExit(f"eager UNet does not reproduce the scripted fcn (max|d| {diff})")
    return unet


def linear(layer: torch.nn.Module) -> dict:
    to_list = lambda t: t.detach().to(torch.float32).numpy().astype(np.float32).tolist()
    return {"weight": to_list(layer.weight), "bias": to_list(layer.bias)}


def postprocess_params(scripted: torch.jit.ScriptModule, source: Path) -> dict:
    classifier = scripted.pixel_classifier
    return {
        "model": "brightfield_nuclei",
        "version": "0.1.1",
        "source_torchscript_sha256": EXPECTED_SHA256,
        "source_file": source.name,
        "pixel_size_um": float(scripted.pixel_size),
        "dim_coords": int(scripted.dim_coords),
        "n_sigma": int(scripted.n_sigma),
        "dim_seeds": int(scripted.dim_seeds),
        "mask_threshold": float(scripted.default_mask_threshold),
        "peak_distance": int(scripted.default_peak_distance),
        "seed_threshold": float(scripted.default_seed_threshold),
        "overlap_threshold": float(scripted.default_overlap_threshold),
        "window_size": int(scripted.default_window_size),
        "cleanup_fragments": bool(scripted.default_cleanup_fragments),
        "pixel_classifier": {
            "fc1": linear(classifier.fc1),
            "fc2": linear(classifier.fc2),
            "fc3": linear(classifier.fc3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="path-2-burn weights directory")
    args = parser.parse_args()

    scripted = load_scripted(args.torchscript)
    unet = eager_unet(scripted)

    unet_path = args.out / f"{UNET_NAME}.safetensors"
    save_file({key: value.contiguous() for key, value in unet.state_dict().items()}, unet_path)
    register(args.out, UNET_NAME, unet_path)

    params_path = args.out / f"{POSTPROCESS_NAME}.json"
    write_json(params_path, postprocess_params(scripted, args.torchscript))
    register(args.out, POSTPROCESS_NAME, params_path)
    print(f"wrote {unet_path} and {params_path}")


if __name__ == "__main__":
    main()
