"""Cut deterministic square RGB fixture tiles (1024 px by default) at a target resolution from local slides.

Tiles are picked from tissue-rich, hematoxylin-dense areas of the thumbnail with a fixed seed,
read from level 0 and resampled to the target microns-per-pixel.

Usage:
  .venv/bin/python make_tiles.py --out ../fixtures/tiles --mpp 0.25 --per-slide 2 SLIDE [SLIDE ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openslide
from PIL import Image

from common import write_json

TILE = 1024
THUMB = 2048


def candidate_centers(slide: openslide.OpenSlide, count: int, seed: int) -> list[tuple[int, int]]:
    thumb = np.asarray(slide.get_thumbnail((THUMB, THUMB)).convert("RGB"), dtype=np.float32)
    scale_x = slide.dimensions[0] / thumb.shape[1]
    scale_y = slide.dimensions[1] / thumb.shape[0]
    saturation = thumb.max(axis=2) - thumb.min(axis=2)
    darkness = 255.0 - thumb.mean(axis=2)
    score = np.where(saturation > 25, darkness, 0.0)
    threshold = np.quantile(score[score > 0], 0.8) if np.any(score > 0) else 0.0
    ys, xs = np.nonzero(score >= max(threshold, 1.0))
    if len(xs) == 0:
        raise SystemExit("no tissue found in thumbnail")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(xs), size=min(count, len(xs)), replace=False)
    return [(int(xs[i] * scale_x), int(ys[i] * scale_y)) for i in picks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mpp", type=float, default=0.25)
    parser.add_argument("--per-slide", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefix", default="tile")
    parser.add_argument("--size", type=int, default=TILE, help="output side in pixels")
    parser.add_argument("slides", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for slide_index, slide_path in enumerate(args.slides):
        slide = openslide.OpenSlide(str(slide_path))
        slide_mpp = float(slide.properties["openslide.mpp-x"])
        span = round(args.size * args.mpp / slide_mpp)
        for tile_index, (cx, cy) in enumerate(candidate_centers(slide, args.per_slide, args.seed + slide_index)):
            x = min(max(cx - span // 2, 0), slide.dimensions[0] - span)
            y = min(max(cy - span // 2, 0), slide.dimensions[1] - span)
            region = slide.read_region((x, y), 0, (span, span)).convert("RGB")
            resized = region.resize((args.size, args.size), Image.Resampling.BILINEAR)
            # Plain sRGB pixels only: slide ICC profiles exceed PIL's PNG chunk limit on reload.
            tile = Image.frombytes("RGB", resized.size, resized.tobytes())
            name = f"{args.prefix}{slide_index:02d}_{tile_index}"
            tile.save(args.out / f"{name}.png")
            manifest.append({
                "name": name,
                "slide": slide_path.name,
                "slide_mpp": slide_mpp,
                "target_mpp": args.mpp,
                "level0_region": [x, y, span, span],
            })
            print(f"{name}: {slide_path.name} at {x},{y} span {span}")
    write_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
