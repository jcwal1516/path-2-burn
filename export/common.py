"""Helpers shared by the reference scripts: checkpoints (hashing, downloading, the weights
manifest), fixture tiles and timing."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

# Fixture tiles: fixtures/tiles holds 1024 px PNGs at 0.25 um/px.
TILES = Path(__file__).resolve().parent.parent / "fixtures" / "tiles"
# ImageNet RGB mean and std, for inputs in [0, 1].
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    weights: Path,
    name: str,
    repo: str,
    revision: str,
    file: str,
    extra_files: tuple[str, ...] = (),
    expected_sha256: str | None = None,
) -> str:
    """Fetches `file` (and `extra_files`) unchanged from a Hugging Face repo at `revision`
    into WEIGHTS/<name>/, using the caller's own login for gated repos, and lists it with
    its sha256 under `name` in WEIGHTS/manifest.json. Returns the sha256."""
    from huggingface_hub import hf_hub_download

    folder = weights / name
    for item in (*extra_files, file):
        hf_hub_download(repo, item, revision=revision, local_dir=folder)
    digest = register(weights, name, folder / file)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SystemExit(f"{name}: sha256 {digest} differs from the published {expected_sha256}")
    print(f"{name}: {repo}@{revision[:8]} {file} sha256 {digest}")
    return digest


def register(weights: Path, name: str, path: Path) -> str:
    """Lists `path` (inside WEIGHTS) under `name` in WEIGHTS/manifest.json with its sha256,
    which it returns."""
    digest = sha256(path)
    manifest_path = weights / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name] = {"file": path.relative_to(weights).as_posix(), "sha256": digest}
    write_json(manifest_path, manifest)
    return digest


def checkpoint(weights: Path, name: str) -> Path:
    """WEIGHTS' manifest entry for `name`, after checking the file's sha256."""
    entry = json.loads((weights / "manifest.json").read_text())[name]
    path = weights / entry["file"]
    digest = sha256(path)
    if digest != entry["sha256"]:
        raise SystemExit(f"{name}: {path} sha256 {digest} differs from the manifest")
    return path


def fixture_crops(
    size: int, corners: tuple[tuple[int, int], ...], reduce: int = 1
) -> tuple[np.ndarray, list[dict]]:
    """[N, size, size, 3] u8 tiles: each fixtures/tiles PNG (0.25 um/px) reduced `reduce`
    times (box filter; 2 gives 0.5 um/px), then cropped at each top-left corner. Also
    returns where each crop came from."""
    tiles, provenance = [], []
    for source in tile_pngs(TILES):
        image = Image.open(source).convert("RGB")
        image = image.reduce(reduce) if reduce > 1 else image
        for x, y in corners:
            tiles.append(np.asarray(image.crop((x, y, x + size, y + size))))
            provenance.append({"source": source.name, "reduce": reduce, "crop_xy": [x, y]})
    return np.stack(tiles).astype(np.uint8), provenance


def read_rgb(path: Path) -> np.ndarray:
    """[H, W, 3] u8 of an image file."""
    return np.asarray(Image.open(path).convert("RGB"))


def write_json(path: Path, value: object, indent: int | None = 2) -> None:
    path.write_text(json.dumps(value, indent=indent) + "\n")


def tile_pngs(directory: Path) -> list[Path]:
    """The PNG tiles in `directory`, sorted; exits when there are none."""
    tiles = sorted(directory.glob("*.png"))
    if not tiles:
        raise SystemExit(f"no PNG tiles in {directory}")
    return tiles


def timed(run: Callable[[], object], sync: Callable[[], None] | None = None, repeats: int = 3) -> float:
    """Mean seconds per call of `run` after one warm-up call; `sync` waits for the device."""
    run()
    if sync:
        sync()
    start = time.perf_counter()
    for _ in range(repeats):
        run()
    if sync:
        sync()
    return (time.perf_counter() - start) / repeats
