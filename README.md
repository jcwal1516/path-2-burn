# path-2-burn

A reusable pathology-model library for Rust applications. It owns model input
requirements, preprocessing, named outputs, model-specific postprocessing,
weight packaging and reference validation. CellViT is the current validation
focus; the library also supports the models below.

Models run on Burn on Metal (macOS), wgpu (Windows/Linux) or the CPU, without
Python or ONNX Runtime. Every model is written directly in Burn and loads its
published PyTorch checkpoint by the original parameter names (InstanSeg's
TorchScript is first split into its UNet weights and postprocessing parameters
by `export/export_instanseg.py`).

| Model | Rust API | Notes |
|---|---|---|
| CellViT-SAM-H (PanNuke) | `cellvit::CellVit` with `cellvit::SAM_H_X40`, `hover::decode` | Native Burn on `sam-burn`; 1024x1024 tiles at 0.25 µm/px; HoVer-Net watershed decoding in Rust |
| CellViT-256 (PanNuke) | `cellvit::CellVit` with `cellvit::VIT256_X40` | Native Burn on `vit-burn` (HIPT ViT-S/16); 256x256 tiles at 0.25 µm/px |
| HoVer-Net: PanNuke, MoNuSAC, CoNSeP (with and without types), Kumar, CPM17 | `hovernet::HoverNet` with a `hovernet::HoverNetWeights` entry, `hover::decode` | Native Burn; fast mode 256 px in / 164 px out, original mode 270 / 80, at 0.25 µm/px; the same watershed decoding as CellViT |
| Cellpose-SAM | `cellpose::CellposeSam` | Native Burn on `sam-burn` (ViT-L, 8 px patches, global attention); 256x256 tiles; flows and cell probability (flow-to-mask dynamics not ported yet) |
| InstanSeg brightfield nuclei | `instanseg::InstanSeg` | Native Burn UNet, postprocessing in Rust; 1024 px tiles at 0.5 µm/px |
| Tile encoders: UNI, UNI2-h, Virchow, Virchow2, H-optimus-0, H-optimus-1, Prov-GigaPath, Phikon, Phikon-v2, Midnight-12k | `encoders::TileEncoder` with an `encoders::Encoder` entry | Native Burn on `vit-burn`; 224x224 tiles at 0.5 µm/px; one embedding per tile, pooled as each card documents |

Weights are not in git. `weights/manifest.json` lists every weights file with
its SHA-256, and `ModelStore` checks the hash before first use.

## Layout

- `crates/sam-burn`, `crates/vit-burn`: generic Burn backbones (the SAM image encoder;
  timm/DINOv2 Vision Transformers) with the original checkpoint parameter names.
- `src/cellvit/`: CellViT's decoders on either backbone (`native.rs`), the
  checkpoint entries and wrapper (`model.rs`).
- `src/hover.rs`, `src/hover/`: shared HoVer-Net instance decoding for CellViT and HoVer-Net.
- `src/encoders.rs`: one entry per tile encoder (upstream, architecture, input,
  pooling) and the `TileEncoder` that runs it.
- `src/{cellvit,hovernet,cellpose,instanseg}`: typed wrappers and model-specific decoding.
- `export/`: offline Python (uv) that downloads checkpoints with your own access
  and writes torch reference fixtures. Never shipped and never run by the app.
- `weights/`, `fixtures/`: local, git-ignored.

## Weights from checkpoints

CellViT needs no conversion. Put the official `CellViT-SAM-H-x40-AMP.pth`
(sha256 `356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4`) in
`weights/` and list it in `weights/manifest.json`:

```json
"cellvit_sam_h": {
  "file": "CellViT-SAM-H-x40-AMP.pth",
  "sha256": "356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4"
}
```

Loading is strict: every Burn parameter must come from the checkpoint and every
checkpoint tensor must be used (BatchNorm's `num_batches_tracked` aside).

InstanSeg's weights come from your copy of the brightfield_nuclei 0.1.1
TorchScript (`instanseg.pt`):

```sh
cd export && uv sync
.venv/bin/python export_instanseg.py --torchscript <instanseg.pt> --out ../weights
```

The runtime uses Burn main from GitHub (0.23.0-pre.1), pinned by git rev in
`Cargo.toml`. Burn's workspace pins CubeCL (0.12.0-pre.1) and CubeK
(0.4.0-pre.1) by git rev; the direct `cubecl-runtime` and `cubecl-wgpu`
dependencies must use the same CubeCL rev, or Cargo builds a second CubeCL and
the device types no longer match. When bumping Burn, copy the CubeCL rev from
Burn's root `Cargo.toml`.

## Tile encoders

No encoder weights ship with this crate. Log in to Hugging Face (`hf auth login`),
accept the terms on each gated model's page (see `MODELS-NOTICE.md`), then:

```sh
cd export && uv sync
.venv/bin/python reference_encoders.py download --weights ../weights            # all, or list names
.venv/bin/python reference_encoders.py reference --weights ../weights --out ../fixtures
cd ..
cargo test --release --test encoder_parity -- --ignored
```

`download` fetches each checkpoint unchanged at a pinned revision into
`weights/<name>/` and records it, with its sha256, in `weights/manifest.json`.
`reference` writes torch float32 CPU references for N tiles: two crops of each
`fixtures/tiles` PNG. The parity tests compare every output token and the pooled
embedding; they are ignored by default because they need the downloaded weights.

## HoVer-Net

Download a checkpoint from vqdang/hover_net's README ("Inference"; PanNuke and
MoNuSAC use fast mode, the others original mode), list it in
`weights/manifest.json` under its entry's name (for example `hovernet_pannuke`)
with its sha256, then:

```sh
cd export
.venv/bin/python reference_hovernet.py --weights ../weights --out ../fixtures hovernet_pannuke
cd ..
cargo test --release --test hovernet_parity -- --ignored pannuke
```

## Cellpose-SAM

```sh
cd export && uv sync
.venv/bin/python reference_cellpose.py download --weights ../weights
.venv/bin/python reference_cellpose.py reference --weights ../weights --out ../fixtures
cd ..
cargo test --release --test cellpose_parity -- --ignored
```

`download` fetches the public `cpsam` checkpoint (BSD-3) at a pinned revision;
`reference` runs it with the cellpose package on N tiles of 256x256: two crops of each
`fixtures/tiles` PNG at 0.5 µm/px. The test checks `CellposeSam::normalize_image`
against cellpose's `normalize_img`, then the network's flows.

## CellViT validation

Run these with the existing weights and reference fixtures present:

```sh
cargo +1.99.0 test --locked --release -p path-2-burn --example parity_cellvit
cargo +1.99.0 run --locked --release -p path-2-burn --example parity_cellvit
cargo +1.99.0 test --locked --release -p path-2-burn --test cellvit_postprocess
```

CellViT-256 is checked the same way on 256x256 crops of `fixtures/tiles`. Download
`CellViT-256-x40.pth` from the link in TIO-IKIM/CellViT's README (its sha256 is in
`export/cellvit_ref/cellvit_model/defaults/cellvit_256.json`), list it in
`weights/manifest.json` as `cellvit_256`, then:

```sh
cd export
.venv/bin/python reference_cellvit.py --variant vit256 --checkpoint ../weights/cellvit_256/CellViT-256-x40.pth \
  --tiles ../fixtures/tiles --out ../fixtures/cellvit_256
cd ..
cargo test --release --test cellvit_parity -- --ignored vit256
```

The example checks every value in all four raw outputs against CPU float32
references. It fails on missing fixtures, wrong lengths, non-finite values or
errors above `atol + 1e-4 * abs(reference)`. Optional positional arguments are
`[weights_dir] [fixtures_dir] [max_tiles]`; `max_tiles` must be positive.

| Output | Absolute tolerance (`atol`) |
|---|---:|
| Nucleus binary logits | 0.005 |
| Nucleus type logits | 0.0005 |
| Horizontal/vertical distances | 0.0001 |
| Tissue logits | 0.0001 |

These are float32 backend tolerances, not bitwise equality. On the ten existing
fixtures, PyTorch 2.14.1/MPS itself differs from the CPU references by up to
0.00292 binary-logit units, 0.000158 type-logit units, 0.000042 HV units and
0.000017 tissue-logit units. The example reports integer counts of changed pixel
argmax decisions and requires the tissue class to agree.

The integration tests separately require bit-exact CPU postprocessing of fixed
reference maps. Two dense tiles also run through Burn prediction and cell
decoding: cell counts and types must match, every reference cell must have a
centroid match within one pixel, and relabelled foreground-pixel agreement must
be at least 99.99%. This checks the application-facing result in addition to
small raw-tensor errors.

Other existing model comparisons:

```sh
cargo run --release --example parity_instanseg
```

The first run on a new tile shape autotunes GPU kernels; the results are cached
on disk.

CellViT timing comparisons must use the same input/output scope. `forward_raw`
includes input upload and readback of four full outputs; `predict` reduces
probabilities and classes on the GPU before readback. Neither includes slide I/O
or cell postprocessing. The validation example's per-tile elapsed times are
diagnostics, not a warmed benchmark.

The measurements below used the ONNX-generated CellViT that `native.rs` replaced;
they have not been repeated for the native model.

Matched measurements on 2026-10-10, M4 Pro MacBook Pro (16 GPU cores, 48 GB),
batch-one float32 on `tile04_0`, with five warmups and five measured calls per
process:

| Backend | Round A median | Round B median |
|---|---:|---:|
| Burn 0.22, wgpu/Metal | 3.778 s/tile | 3.730 s/tile |
| PyTorch 2.8.0, MPS | 3.094 s/tile | 2.981 s/tile |

Process order was Burn A, PyTorch A, PyTorch B, Burn B. Both measured CPU input
copy/upload through four raw contiguous CPU outputs. Model loading and disk I/O
were excluded. The laptop was on AC power with normal desktop activity, so these
are not isolated-system measurements. These baseline runs used the registry
runtime and Burn was slower in both rounds. This single-tile result does not
measure whole-slide throughput.

A subsequent local-runtime comparison used the same timing scope, input and
warmup/repeat counts, in baseline → patched → patched → baseline order. The
patch was a local Burn 0.22 change and is not part of the current Burn main
runtime:

| Local Burn runtime | Round A median | Round B median |
|---|---:|---:|
| Original convolution choices | 4.601 s/tile | 4.925 s/tile |
| Additional 64×64 FP32 convolution choice | 4.236 s/tile | 4.199 s/tile |

The patched runtime was 8–15% faster in these two pairs. Other jobs remained
active and timings drifted; compare within this paired run, not with the earlier
registry measurements. A refreshed PyTorch/MPS run measured 3.861 s/tile, so this
change does not yet make Burn faster than MPS. The 64×64 candidate won 13
convolution autotune keys in this model. All ten raw-output fixtures passed the
existing tolerances, and the two dense tiles retained all 1,000 and 1,540
reference cells with identical types,
centroids and foreground masks. On the timed tile, the patched and baseline raw
outputs were also exactly equal. The run outputs are under
`target/cellvit-kernel-fix/benchmark/`. A warmed GPU profile measured the 34
large decoder convolutions at 0.910 s total, versus 1.403 s in the baseline
profile. The three main encoder dense-matrix kernel families still took 1.651 s
in the patched profile.

The previously measured InstanSeg network takes 0.7 s per tile on an M4 Pro, and
its label maps match the TorchScript model.

See `MODELS-NOTICE.md` for weight licenses and citations.
