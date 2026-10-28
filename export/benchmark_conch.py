"""Tile encoder float32 CPU-input-to-CPU-embedding latency on PyTorch MPS.

Matches examples/benchmark_conch.rs: same cached RGB tiles, normalization,
input copy/upload, embedding readback and warmup/repeat counts. Model loading,
file I/O and normalization are excluded. Run the processes sequentially on AC
power; use alternating process order to check for temperature/load drift.
Defaults to CONCH v1.5; select another model with --encoder virchow2.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from common import checkpoint
from reference_encoders import ENCODERS, fixture_tiles, pool, prefix_tokens, tokens_of


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=ENCODERS, default="conch_v1_5")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.batch, args.warmups, args.repeats) <= 0:
        parser.error("batch, warmups and repeats must be positive")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; this benchmark requires the Apple GPU")
    root = Path(__file__).resolve().parent.parent
    spec = ENCODERS[args.encoder]
    weights = checkpoint(root / "weights", args.encoder)
    model = spec.build(weights.parent, weights).eval().float().to("mps")
    trunk = model.trunk if spec.pooling == "conch" else model
    reference_dir = root / "fixtures/encoders" / args.encoder
    meta = json.loads((reference_dir / "meta.json").read_text())
    tiles = fixture_tiles(root / "fixtures/encoders", spec.tile)
    if args.batch > len(tiles):
        parser.error("batch exceeds fixture count")
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in spec.norm)
    x = torch.from_numpy(tiles[:args.batch]).permute(0, 3, 1, 2).float() / 255.0
    x = ((x - mean) / std).contiguous()

    # Match the Rust benchmark's token check outside the timed embedding calls.
    tokens = tokens_of(trunk, x.clone().to("mps")).cpu().contiguous().numpy()
    token_reference = np.fromfile(reference_dir / "tokens.bin", dtype="<f4").reshape(meta["tokens_shape"])[:args.batch]
    np.testing.assert_allclose(tokens, token_reference, atol=2e-4 * max(1, float(np.abs(token_reference).max())), rtol=1e-3)

    def run() -> np.ndarray:
        tokens = tokens_of(trunk, x.clone().to("mps"))
        if spec.pooling == "conch":
            embedding = model.ln_contrast(model.attn_pool_contrast(tokens)[:, 0])
        elif spec.pooling == "plip":
            embedding = model.visual_projection(model.vision_model.post_layernorm(tokens[:, 0]))
        else:
            embedding = pool(tokens, spec.pooling, prefix_tokens(trunk))
        # Own the CPU output, matching Rust's returned Vec<f32>.
        return embedding.cpu().contiguous().numpy().copy()

    for _ in range(args.warmups):
        run()
    torch.mps.synchronize()
    seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        last = run()
        torch.mps.synchronize()
        seconds.append(time.perf_counter() - start)
    reference = np.fromfile(reference_dir / "embedding.bin", dtype="<f4").reshape(meta["embedding_shape"])[:args.batch]
    np.testing.assert_allclose(last, reference, atol=2e-4 * max(1, float(np.abs(reference).max())), rtol=1e-3)
    median = statistics.median(seconds)
    print(json.dumps({
        "framework": "pytorch", "version": torch.__version__, "device": "mps",
        "model": args.encoder, "dtype": "float32",
        "timm_fused_attention": sorted({m.fused_attn for m in model.modules() if hasattr(m, "fused_attn")}),
        "scope": "normalized_cpu_input_to_cpu_embedding", "batch": args.batch,
        "warmups": args.warmups, "repeats": args.repeats, "seconds": seconds,
        "median_seconds": median, "images_per_second": args.batch / median,
    }))


if __name__ == "__main__":
    main()
