"""Download the tile encoders' checkpoints and write torch references for them.

The models match `src/encoders.rs`. Weights are never redistributed: `download` fetches
each checkpoint unchanged from Hugging Face at a pinned revision, with your own access
(`hf auth login`, and the model's terms accepted on its page for gated repos), into
WEIGHTS/<name>/, and adds it with its sha256 to WEIGHTS/manifest.json.

`reference` builds each downloaded model the way its model card does (timm or
transformers) and writes, under OUT/encoders/:

  tiles.bin               N RGB u8 tiles, 224x224 at 0.5 um/px: two crops of each
                          fixtures/tiles PNG (1024 px at 0.25 um/px) after 2x box reduction.
                          Written once and reused.
  <name>/tokens.bin       every output token after the final norm, [N, tokens, width] float32
  <name>/embedding.bin    the pooled embedding the card documents, [N, dim] float32
  <name>/meta.json        shapes, library versions and the checkpoint file

The tiles are already at the model's size, so the cards' resize and crop steps are left
out; inputs are normalized as (x / 255 - mean) / std in float32, as the Rust wrapper does.
References run in float32 on the CPU.

Usage:
  .venv/bin/python reference_encoders.py download --weights ../weights [NAME ...]
  .venv/bin/python reference_encoders.py reference --weights ../weights --out ../fixtures [NAME ...]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import timm
import torch
import transformers
from safetensors.torch import load_file
from timm.layers import SwiGLUPacked
from torch import nn
from transformers import AutoModel

from common import IMAGENET, checkpoint, download, fixture_crops, write_json

# Bioptimus' H-optimus normalization.
BIOPTIMUS = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))
TILE = 224
CROPS = ((0, 0), (288, 288))  # top-left corners in the 512 px reduced tile
BATCH = 4


@dataclass(frozen=True)
class Spec:
    repo: str
    revision: str
    file: str
    build: Callable[[Path, Path], nn.Module]
    norm: tuple[tuple[float, float, float], tuple[float, float, float]] = IMAGENET
    pooling: str = "cls"  # or "cls+mean": class token, then the mean of the patch tokens
    # Published sha256 of `file`, when the repo is public; gated repos hide it.
    sha256: str | None = None
    extra_files: tuple[str, ...] = field(default=("config.json",))


def load_state(weights: Path) -> dict[str, torch.Tensor]:
    if weights.suffix == ".safetensors":
        return load_file(weights)
    return torch.load(weights, map_location="cpu", weights_only=True)


def timm_card(model_name: str, **kwargs: object) -> Callable[[Path, Path], nn.Module]:
    """The card's explicit `timm.create_model` call, then a strict state-dict load."""

    def build(_folder: Path, weights: Path) -> nn.Module:
        model = timm.create_model(model_name, pretrained=False, **kwargs)
        model.load_state_dict(load_state(weights), strict=True)
        return model

    return build


def timm_hub(**kwargs: object) -> Callable[[Path, Path], nn.Module]:
    """What `timm.create_model("hf-hub:...", **kwargs)` builds from the repo's config.json."""

    def build(folder: Path, weights: Path) -> nn.Module:
        config = json.loads((folder / "config.json").read_text())
        args = {**config.get("model_args", {}), **kwargs, "num_classes": 0}
        # As timm does for hf-hub models: a fixed input size sets img_size.
        pretrained = config.get("pretrained_cfg", {})
        if pretrained.get("fixed_input_size"):
            args.setdefault("img_size", tuple(pretrained["input_size"][-2:]))
        model = timm.create_model(config["architecture"], pretrained=False, **args)
        model.load_state_dict(load_state(weights), strict=True)
        return model

    return build


# Virchow and Virchow2: their card's packed SwiGLU on the repo's timm config.
VIRCHOW_BUILD = timm_hub(mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)


def transformers_auto(folder: Path, _weights: Path) -> nn.Module:
    return AutoModel.from_pretrained(folder, torch_dtype=torch.float32)


ENCODERS: dict[str, Spec] = {
    "uni": Spec(
        "MahmoodLab/UNI",
        "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0",
        "pytorch_model.bin",
        timm_card(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        ),
    ),
    "uni2_h": Spec(
        "MahmoodLab/UNI2-h",
        "d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
        "pytorch_model.bin",
        timm_card(
            "vit_giant_patch14_224",
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        ),
    ),
    "virchow": Spec(
        "paige-ai/Virchow",
        "19eebc84ae33e79f1b2d866e6ff90ae50e522f9a",
        "model.safetensors",
        VIRCHOW_BUILD,
        pooling="cls+mean",
    ),
    "virchow2": Spec(
        "paige-ai/Virchow2",
        "3158645804b69e3f3bc4439d4116edddf0840a72",
        "model.safetensors",
        VIRCHOW_BUILD,
        pooling="cls+mean",
    ),
    "h_optimus_0": Spec(
        "bioptimus/H-optimus-0",
        "b145cc1e6c6b30d3251aa8b1f844e6974188a743",
        "pytorch_model.bin",
        timm_hub(init_values=1e-5, dynamic_img_size=False),
        norm=BIOPTIMUS,
    ),
    "h_optimus_1": Spec(
        "bioptimus/H-optimus-1",
        "50170265f1240b9b21ad65c6781f5432e438fa68",
        "model.safetensors",
        timm_hub(init_values=1e-5, dynamic_img_size=False),
        norm=BIOPTIMUS,
    ),
    "prov_gigapath": Spec(
        "prov-gigapath/prov-gigapath",
        "64f9e26c15019f2d4f6d9113c6822f88bb16b01b",
        "pytorch_model.bin",
        timm_hub(),
    ),
    "phikon": Spec(
        "owkin/phikon",
        "057cc0295895c2df3dd7681a89680da6015cbefe",
        "model.safetensors",
        transformers_auto,
        sha256="d01642bfeb683c6b039622eca273e0a8e4208065a4595b63edba178c1c4cfa68",
    ),
    "phikon_v2": Spec(
        "owkin/phikon-v2",
        "2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd",
        "model.safetensors",
        transformers_auto,
        sha256="261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569",
    ),
    "midnight_12k": Spec(
        "kaiko-ai/midnight",
        "adc6b15679c981cce6f9b018bbad09d16eeeda9f",
        "model.safetensors",
        transformers_auto,
        norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        pooling="cls+mean",
        sha256="52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1",
    ),
}


def selected(names: list[str]) -> list[str]:
    unknown = sorted(set(names) - ENCODERS.keys())
    if unknown:
        raise SystemExit(f"unknown encoders {unknown}; known: {sorted(ENCODERS)}")
    return names or list(ENCODERS)


def download_all(weights: Path, names: list[str]) -> None:
    for name in selected(names):
        spec = ENCODERS[name]
        download(weights, name, spec.repo, spec.revision, spec.file, spec.extra_files, spec.sha256)


def fixture_tiles(out: Path) -> np.ndarray:
    """[N, 224, 224, 3] u8, written once to OUT/encoders/tiles.bin and reused."""
    path = out / "tiles.bin"
    if path.exists():
        return np.fromfile(path, dtype=np.uint8).reshape(-1, TILE, TILE, 3)
    tiles, provenance = fixture_crops(TILE, CROPS, reduce=2)
    out.mkdir(parents=True, exist_ok=True)
    tiles.tofile(path)
    write_json(out / "tiles.json", provenance)
    return tiles


def tokens_of(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, transformers.PreTrainedModel):
        return model(pixel_values=x).last_hidden_state
    forward_features: Callable[[torch.Tensor], torch.Tensor] = getattr(model, "forward_features")
    return forward_features(x)


def pool(tokens: torch.Tensor, pooling: str, prefix: int) -> torch.Tensor:
    cls = tokens[:, 0]
    if pooling == "cls":
        return cls
    return torch.cat([cls, tokens[:, prefix:].mean(1)], dim=-1)


def prefix_tokens(model: nn.Module) -> int:
    return int(getattr(model, "num_prefix_tokens", 1))  # transformers ViT/DINOv2: class token only


@torch.no_grad()
def reference(weights: Path, out: Path, names: list[str]) -> None:
    out = out / "encoders"
    tiles = fixture_tiles(out)
    for name in selected(names):
        spec = ENCODERS[name]
        path = checkpoint(weights, name)
        model = spec.build(path.parent, path).eval().float()
        mean, std = (torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1) for v in spec.norm)
        all_tokens, embeddings = [], []
        for start in range(0, len(tiles), BATCH):
            x = torch.from_numpy(tiles[start : start + BATCH]).permute(0, 3, 1, 2).float() / 255.0
            tokens = tokens_of(model, (x - mean) / std)
            all_tokens.append(tokens)
            embeddings.append(pool(tokens, spec.pooling, prefix_tokens(model)))
        tokens = torch.cat(all_tokens).contiguous()
        embedding = torch.cat(embeddings).contiguous()
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        tokens.numpy().astype("<f4").tofile(folder / "tokens.bin")
        embedding.numpy().astype("<f4").tofile(folder / "embedding.bin")
        meta = {
            "repo": spec.repo,
            "revision": spec.revision,
            "checkpoint": path.name,
            "tokens_shape": list(tokens.shape),
            "embedding_shape": list(embedding.shape),
            "prefix_tokens": prefix_tokens(model),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "transformers": transformers.__version__,
        }
        write_json(folder / "meta.json", meta)
        print(f"{name}: tokens {tuple(tokens.shape)} embedding {tuple(embedding.shape)}")
        del model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--weights", type=Path, required=True)
    fetch.add_argument("names", nargs="*")
    ref = commands.add_parser("reference")
    ref.add_argument("--weights", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    ref.add_argument("names", nargs="*")
    args = parser.parse_args()
    if args.command == "download":
        download_all(args.weights, args.names)
    else:
        reference(args.weights, args.out, args.names)


if __name__ == "__main__":
    main()
