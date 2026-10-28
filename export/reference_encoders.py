"""Download the tile encoders' checkpoints and write torch references for them.

The models match `src/encoders.rs`. Weights are never redistributed: `download` fetches
each checkpoint unchanged from Hugging Face at a pinned revision, with your own access
(`hf auth login`, and the model's terms accepted on its page for gated repos), into
WEIGHTS/<name>/, and adds it with its sha256 to WEIGHTS/manifest.json.

`reference` builds each downloaded model the way its model card does (timm,
transformers, or downloaded upstream code) and writes, under OUT/encoders/:

  tiles.bin               N RGB u8 tiles, 224x224 at 0.5 um/px: two crops of each
                          fixtures/tiles PNG (1024 px at 0.25 um/px) after 2x box reduction.
                          Written once and reused.
  tiles_448.bin           CONCH v1/v1.5's 448x448 crops, also at 0.5 um/px.
  <name>/tokens.bin       every output token, [N, tokens, width] float32. PLIP stores
                          last_hidden_state before the class token's post_layernorm;
                          other entries include final normalization.
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
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import numpy as np
import timm
import torch
import transformers
from safetensors.torch import load_file
from timm.layers import SwiGLUPacked
from torch import nn
from transformers import AutoModel, CLIPVisionConfig, CLIPVisionModelWithProjection

from common import IMAGENET, checkpoint, download, fixture_crops, load_upstream_module, write_json

# Bioptimus' H-optimus normalization.
BIOPTIMUS = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
CONCH_SOURCE_REVISION = "141cc09c7d4ff33d8eda562bd75169b457f71a62"
BATCH = 4


@dataclass(frozen=True)
class Spec:
    repo: str
    revision: str
    file: str
    build: Callable[[Path, Path], nn.Module]
    norm: tuple[tuple[float, float, float], tuple[float, float, float]] = IMAGENET
    pooling: str = "cls"  # also "cls+mean", "conch" attention pooling, or "plip" projection
    tile: int = 224
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


def conch_v1_5(folder: Path, weights: Path) -> nn.Module:
    """The upstream build_conch architecture, with a strict local checkpoint load."""
    source = importlib.util.spec_from_file_location("conch_v1_5", folder / "conch_v1_5.py")
    assert source is not None and source.loader is not None
    upstream = importlib.util.module_from_spec(source)
    source.loader.exec_module(upstream)
    cfg = json.loads((folder / "config.json").read_text())["conch_config"]
    trunk = upstream.VisionTransformer(
        patch_size=cfg["patch_size"], embed_dim=cfg["context_dim"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
        qkv_bias=cfg["qkv_bias"], init_values=cfg["init_values"],
    )
    pooler = upstream.AttentionalPooler(
        d_model=cfg["embed_dim"], context_dim=cfg["context_dim"],
        n_queries=cfg["pooler_n_queries_contrast"],
    )
    model = upstream.EncoderWithAttentionalPooler(trunk, pooler, cfg["embed_dim"]).float()
    model.load_state_dict(load_state(weights), strict=True)
    return model


def download_conch_source(folder: Path) -> None:
    for name in ("vision_tower.py", "transformer.py", "utils.py", "model_configs/conch_ViT-B-16.json"):
        target = folder / "upstream" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(f"https://raw.githubusercontent.com/mahmoodlab/CONCH/{CONCH_SOURCE_REVISION}/conch/open_clip_custom/{name}") as response:
                target.write_bytes(response.read())


def conch_v1(folder: Path, weights: Path) -> nn.Module:
    upstream = load_upstream_module(folder / "upstream", "vision_tower")
    cfg = json.loads((folder / "upstream/model_configs/conch_ViT-B-16.json").read_text())
    vision = cfg["vision_cfg"]
    trunk = timm.models.vision_transformer.VisionTransformer(
        img_size=vision["image_size"], patch_size=vision["patch_size"], embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4, num_classes=0, dynamic_img_size=True,
    )
    trunk.forward = trunk.forward_features
    model = upstream.VisualModel(
        cfg["embed_dim"], cfg["embed_dim_caption"], trunk, image_size=vision["image_size"],
        use_attentional_pool_contrast=vision["attentional_pool_contrast"],
        use_attentional_pool_caption=vision["attentional_pool_caption"],
        n_queries_contrast=vision["n_queries_contrast"], n_queries_caption=vision["n_queries_caption"],
        attn_pooler_heads=vision["attn_pooler_heads"], output_tokens=vision["output_tokens"],
    ).float()
    model.load_state_dict({k.removeprefix("visual."): v for k, v in load_state(weights).items()
                           if k.startswith("visual.")}, strict=True)
    return model


def plip(folder: Path, weights: Path) -> nn.Module:
    cfg = json.loads((folder / "config.json").read_text())
    model = CLIPVisionModelWithProjection(CLIPVisionConfig(**cfg["vision_config"])).float()
    state = {k: v for k, v in load_state(weights).items()
             if k.startswith(("vision_model.", "visual_projection."))}
    # transformers 4.x persisted this constant; current versions recreate it.
    position_ids = state.pop("vision_model.embeddings.position_ids")
    expected = torch.arange(model.vision_model.embeddings.num_positions).view(1, -1)
    if not torch.equal(position_ids, expected):
        raise ValueError("PLIP has unexpected position indices")
    model.load_state_dict(state, strict=True)
    return model


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
    "conch_v1_5": Spec(
        "MahmoodLab/TITAN",
        "dac6773d9961cfc75503440676ff157a2c6e8d2e",
        "conch_v1_5_pytorch_model.bin",
        conch_v1_5,
        pooling="conch",
        tile=448,
        extra_files=("config.json", "conch_v1_5.py"),
    ),
    "conch_v1": Spec(
        "MahmoodLab/CONCH", "f9ca9f877171a28ade80228fb195ac5d79003357", "pytorch_model.bin",
        conch_v1, norm=CLIP_NORM, pooling="conch", tile=448, extra_files=(),
        sha256="40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c",
    ),
    "plip": Spec(
        "vinid/plip", "67ade53ddd32195868f422585f72698ef5d15094", "pytorch_model.bin",
        plip, norm=CLIP_NORM, pooling="plip",
        sha256="98a7f8d2a1f4a8fc8f6dedb3a16ff7efbe02a7ef67c93904c80bca9767c69630",
        extra_files=("config.json", "preprocessor_config.json"),
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
        if name == "conch_v1":
            download_conch_source(weights / name)


def fixture_tiles(out: Path, tile: int = 224) -> np.ndarray:
    """[N, tile, tile, 3] u8, cached separately for each input size."""
    path = out / ("tiles.bin" if tile == 224 else f"tiles_{tile}.bin")
    if path.exists():
        return np.fromfile(path, dtype=np.uint8).reshape(-1, tile, tile, 3)
    tiles, provenance = fixture_crops(tile, ((0, 0), (512 - tile, 512 - tile)), reduce=2)
    out.mkdir(parents=True, exist_ok=True)
    tiles.tofile(path)
    write_json(path.with_suffix(".json"), provenance)
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
    for name in selected(names):
        spec = ENCODERS[name]
        tiles = fixture_tiles(out, spec.tile)
        path = checkpoint(weights, name)
        model = spec.build(path.parent, path).eval().float()
        trunk = model.trunk if spec.pooling == "conch" else model
        mean, std = (torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1) for v in spec.norm)
        all_tokens, embeddings = [], []
        for start in range(0, len(tiles), BATCH):
            x = torch.from_numpy(tiles[start : start + BATCH]).permute(0, 3, 1, 2).float() / 255.0
            tokens = tokens_of(trunk, (x - mean) / std)
            all_tokens.append(tokens)
            if spec.pooling == "conch":
                embeddings.append(model.ln_contrast(model.attn_pool_contrast(tokens)[:, 0]))
            elif spec.pooling == "plip":
                embeddings.append(model.visual_projection(model.vision_model.post_layernorm(tokens[:, 0])))
            else:
                embeddings.append(pool(tokens, spec.pooling, prefix_tokens(trunk)))
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
            "prefix_tokens": prefix_tokens(trunk),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "transformers": transformers.__version__,
        }
        if name == "conch_v1":
            meta["source_revision"] = CONCH_SOURCE_REVISION
        write_json(folder / "meta.json", meta)
        print(f"{name}: tokens {tuple(tokens.shape)} embedding {tuple(embedding.shape)}")
        del model, trunk


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
