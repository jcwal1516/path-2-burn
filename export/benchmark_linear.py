"""Resident CONCH dense-layer shapes matching examples/benchmark_linear.rs.

Initialization and readback are excluded. Each timed call includes dispatch
and GPU synchronization. This isolates linear layers, not a whole-model run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

TOKENS, WARMUPS, REPEATS = 785, 5, 21


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[1, 4], default=1)
    batch = parser.parse_args().batch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required")
    for name, k, n in [("qkv", 1024, 3072), ("projection", 1024, 1024),
                       ("mlp_expand", 1024, 4096), ("mlp_contract", 4096, 1024)]:
        x_cpu = (((np.arange(batch * TOKENS * k, dtype=np.int64) * 13 + 7) % 127 - 63)
                 .astype(np.float32) / 256).reshape(batch, TOKENS, k)
        w_cpu = (((np.arange(n * k, dtype=np.int64) * 17 + 3) % 127 - 63)
                 .astype(np.float32) / 256).reshape(n, k)
        b_cpu = ((np.arange(n, dtype=np.int64) % 17 - 8).astype(np.float32) / 256)
        x = torch.from_numpy(x_cpu).to("mps")
        bias = torch.from_numpy(b_cpu).to("mps")
        column = torch.from_numpy(w_cpu).to("mps")
        row = torch.from_numpy(w_cpu.T.copy()).to("mps").T
        for flatten in ([False] if batch == 1 else [False, True]):
            for layout, weight in [("row", row), ("column", column)]:
                # Layout labels describe the logical RHS [k, n], as in Rust.
                def run() -> torch.Tensor:
                    values = x.reshape(batch * TOKENS, k) if flatten else x
                    return F.linear(values, weight, bias).reshape(batch, TOKENS, n)

                for _ in range(WARMUPS):
                    last = run()
                    torch.mps.synchronize()
                seconds = []
                for _ in range(REPEATS):
                    start = time.perf_counter()
                    out = run()
                    torch.mps.synchronize()
                    seconds.append(time.perf_counter() - start)
                    last = out
                values = last.cpu().numpy()
                assert np.isfinite(values).all()
                max_error = 0.0
                for b in range(batch):
                    for t in [0, TOKENS // 2, TOKENS - 1]:
                        for j in [0, n // 2, n - 1]:
                            expected = float(b_cpu[j]) + np.dot(
                                x_cpu[b, t].astype(np.float64), w_cpu[j].astype(np.float64))
                            max_error = max(max_error, abs(float(values[b, t, j]) - expected))
                assert max_error <= 1e-5, (name, max_error)
                print(json.dumps({
                    "framework": "pytorch", "version": torch.__version__, "operation": name,
                    "batch": batch, "tokens": TOKENS, "k": k, "n": n,
                    "weight_layout": layout, "flatten": flatten,
                    "warmups": WARMUPS, "repeats": REPEATS,
                    "scope": "resident_linear_dispatch_and_sync", "dtype": "float32",
                    "seconds": seconds, "median_seconds": statistics.median(seconds),
                    "sample_max_error": max_error,
                }), flush=True)


if __name__ == "__main__":
    main()
