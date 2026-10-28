# path-2-burn

A reusable pathology-model library for Rust applications. It owns model input
requirements, preprocessing, named outputs, model-specific postprocessing,
weight packaging and reference validation. CellViT is the current validation
focus; the library also supports the models below.

Models run on Burn on Metal (macOS), wgpu (Windows/Linux) or the CPU, without
Python or ONNX Runtime. Every model is written directly in Burn and loads its
published checkpoint by its parameter names. InstanSeg's TorchScript is first
split into its UNet weights and postprocessing parameters by
`export/export_instanseg.py`; StarDist's frozen TensorFlow graphs are read by
OpenCV to extract their convolution tensors into safetensors.

| Model | Rust API | Notes |
|---|---|---|
| CellViT-SAM-H (PanNuke) | `cellvit::CellVit` with `cellvit::SAM_H_X40`, `hover::decode` | Native Burn on `sam-burn`; 1024x1024 tiles at 0.25 µm/px; HoVer-Net watershed decoding in Rust |
| CellViT-256 (PanNuke) | `cellvit::CellVit` with `cellvit::VIT256_X40` | Native Burn on `vit-burn` (HIPT ViT-S/16); 256x256 tiles at 0.25 µm/px |
| HoVer-Net: PanNuke, MoNuSAC, CoNSeP (with and without types), Kumar, CPM17 | `hovernet::HoverNet` with a `hovernet::HoverNetWeights` entry, `hover::decode` | Native Burn; fast mode 256 px in / 164 px out, original mode 270 / 80, at 0.25 µm/px; the same watershed decoding as CellViT |
| Cellpose-SAM | `cellpose::CellposeSam` | Native Burn on `sam-burn` (ViT-L, 8 px patches, global attention); 256x256 tiles; flows, cell probability and 2D instance masks |
| InstanSeg brightfield nuclei | `instanseg::InstanSeg` | Native Burn UNet, postprocessing in Rust; 1024 px tiles at 0.5 µm/px |
| Tile encoders: UNI, UNI2-h, Virchow, Virchow2, H-optimus-0, H-optimus-1, Prov-GigaPath, Phikon, Phikon-v2, Midnight-12k | `encoders::TileEncoder` with an `encoders::Encoder` entry | Native Burn on `vit-burn`; 224x224 tiles at 0.5 µm/px; one embedding per tile, pooled as each card documents |
| CONCH v1.5 | `encoders::TileEncoder` with `encoders::CONCH_V1_5` | Native Burn ViT-L/16 with attention pooling; 448x448 inputs; 768-dimensional embedding |
| CONCH v1 | `encoders::TileEncoder` with `encoders::CONCH_V1` | ViT-B/16 at 448x448; 512-dimensional features before contrastive projection and normalization |
| PLIP | `encoders::TileEncoder` with `encoders::PLIP` | CLIP ViT-B/32 at 224x224; 512-dimensional projected image features without L2 normalization |
| TITAN (slide encoder) | `titan::Titan` | CONCH v1.5 patch features and level-0 coordinates to a 768-dimensional slide embedding; six transformer blocks with 2D ALiBi and attention pooling |
| StarDist2D (H&E, fluorescence heavy augmentation, fluorescence paper) | `stardist::StarDist` with `HE`, `FLUORESCENCE`, or `FLUORESCENCE_PAPER` | U-Net probabilities and 32 ray distances; polygon NMS in Rust; RGB or single-channel input |

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
- `src/titan/`, `src/stardist/`: slide encoding and star-convex nucleus detection.
- `src/open_clip.rs`: attention pooling and checkpoint key names shared by TITAN and CONCH v1.
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

Loading is strict: every Burn parameter must come from the checkpoint. Every
checkpoint tensor must be loaded or explicitly listed as unused, such as a text
tower omitted from an image-only API (and BatchNorm's `num_batches_tracked`).

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

For CONCH v1.5 alone, use `conch_v1_5` with each command:

```sh
export/.venv/bin/python export/reference_encoders.py download --weights weights conch_v1_5
export/.venv/bin/python export/reference_encoders.py reference --weights weights --out fixtures conch_v1_5
cargo test --release --test encoder_parity conch_v1_5 -- --ignored --nocapture
```

CONCH v1.5 loads the original image checkpoint published with
[TITAN](https://huggingface.co/MahmoodLab/TITAN). The download command fetches the
pinned upstream `conch_v1_5.py` alongside the weights and config; reference
generation loads them locally. The Rust wrapper expects RGB tiles already
resized and cropped to 448x448; upstream uses
bilinear resize of the shorter side to 448 followed by a 448x448 center crop, then
ImageNet normalization. For TITAN features, start with 512x512 patches at 20×
(0.5 µm/px) before resizing. The parity fixtures use 448x448 crops at 0.5 µm/px,
stored separately in `fixtures/encoders/tiles_448.bin`, to compare both runtimes
on identical pixels. Embeddings are returned after the pooling head's LayerNorm,
without L2 normalization.

CONCH v1 and PLIP use the same download/reference commands with names `conch_v1`
and `plip`. Both use CLIP RGB normalization. Prepare CONCH v1 images by bicubic
resize of the shorter side to 448 and a 448x448 center crop; PLIP uses 224.
CONCH v1 matches upstream `encode_image(proj_contrast=False, normalize=False)`
for downstream feature extraction. PLIP matches transformers' projected image
features. Their text towers and tokenizers are not exposed by `TileEncoder`.
CONCH v1's reference code is downloaded at a pinned GitHub revision alongside
its checkpoint. Its one-query attention pooler is shared with TITAN in Rust.

## TITAN slide encoder

TITAN takes flattened `[N, 768]` CONCH v1.5 features and matching `[x, y]`
coordinates in level-0 pixels. `Titan::encode_slide_from_patch_features` also
takes the level-0 patch spacing: 1024 for 512 px patches read at 20× from a 40×
slide, or 512 from a 20× slide. It returns the upstream slide embedding before
the vision-language projection, without L2 normalization.

```sh
export/.venv/bin/python export/reference_titan.py download --weights weights
export/.venv/bin/python export/reference_titan.py reference --weights weights --input <patch_features.h5> --out fixtures
cargo test --release --test titan_parity -- --ignored --nocapture
```

The reference input has `features`, `coords` and the `coords` attribute
`patch_size_level0`, as in the upstream demo. `--limit 64` uses a small subset
during development. References also cover sparse and duplicate grid cells and
a single retained patch. Both implementations sum features that occupy the
same cell and discard all-zero cells. Rust rejects empty or non-finite inputs
and slides with no nonzero cells. Attention memory is quadratic in the number
of retained cells; the Rust preprocessing keeps a sparse grid so coordinate
gaps alone do not allocate image-sized arrays.

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

## StarDist2D

Download the three frozen graphs from [qupath/models](https://github.com/qupath/models/tree/main/stardist)
into a graph directory, then extract weights and generate references:

```sh
export/.venv/bin/python export/reference_stardist.py export --graphs weights/stardist --weights weights
export/.venv/bin/python export/reference_stardist.py reference --graphs weights/stardist --out fixtures
cargo test --release --test stardist_parity -- --ignored --nocapture --test-threads=1
```

`StarDistModel::normalize` applies per-channel 1st / 99.8th percentile
normalization to a CHW float32 image without clipping. `StarDist::predict` takes
one normalized image; each side must be divisible by 16 for the heavy-augmentation
models, or 8 for the paper model. Pad or tile larger images before calling it.
Predictions stay on the native grid (spacing 2 or 1), with distances measured in
input-image pixels. `StarDistRaw::instances` selects polygons by descending
probability using intersection area divided by the smaller polygon's area.
Equal scores use descending grid index as a deterministic tie-breaker.
Each instance provides its center, probability, ray distances, and `[x, y]`
polygon vertices. This API returns polygons rather than rasterized label masks.

References use the public H&E and fluorescence samples included in StarDist,
OpenCV for network outputs, and StarDist's C++ NMS for selected points and scores.
The export environment needs neither TensorFlow nor a new model conversion
framework. Rust uses the `clipper2-rust` polygon-intersection library.

## Cellpose-SAM

```sh
cd export && uv sync
.venv/bin/python reference_cellpose.py download --weights ../weights
.venv/bin/python reference_cellpose.py reference --weights ../weights --out ../fixtures
.venv/bin/python reference_cellpose.py masks --out ../fixtures
cd ..
cargo test --release --test cellpose_parity -- --ignored
cargo test --release --test cellpose_postprocess -- --ignored
```

`download` fetches the public `cpsam` checkpoint (BSD-3) at a pinned revision;
`reference` runs it with the cellpose package on N tiles of 256x256: two crops of each
`fixtures/tiles` PNG at 0.5 µm/px. The test checks `CellposeSam::normalize_image`
against cellpose's `normalize_img`, then the network's flows and resulting masks.

`CellposeSam::masks` returns a label image for each normalized tile, with label IDs
starting at one per tile. `cellpose::compute_masks` decodes an existing `[3, H, W]`
flow/logit array on the CPU. `DynamicsOptions` defaults to 200 integration steps,
cell-logit threshold 0, flow-error threshold 0.4, minimum area 15, and maximum
area fraction 0.4. It includes bilinear flow integration, convergence histogram
clustering, diffusion-based flow filtering, hole filling and size filtering.
This is the 2D path; image resizing and stitching across tiles are caller tasks.

The `masks` reference command reuses the saved network outputs and adds rectangular
synthetic and empty cases. Postprocessing is compared pixel-for-pixel before and
after cleanup. The end-to-end GPU check preserves mask counts and instance matches
while allowing small foreground-boundary differences from float32 arithmetic.

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

For a matched CONCH v1.5 benchmark, first generate its cached encoder fixtures
and CPU references, then run these commands sequentially from the repository
root on AC power. Change both batch arguments to `4` for the batched comparison:

```sh
BURN_DEVICE=metal cargo run --locked --release --example benchmark_conch -- 1 5 7
export/.venv/bin/python export/benchmark_conch.py --batch 1 --warmups 5 --repeats 7
```

Both measure normalized CPU input copy/upload through an owned CPU embedding,
using the same 448×448 tiles and float32 tensors. Loading, file I/O and
normalization are excluded; readback completes inside each timed call. Each
process checks its final embedding against the existing CPU reference tolerance
and prints individual timings, median latency and images per second as JSON.

For Virchow2, set `BURN_BENCH_ENCODER=virchow2` on the Rust command and add
`--encoder virchow2` to the Python command. Other registered encoder names are
also supported. Both use the selected encoder's saved fixtures and check token
outputs outside the timer as well as the final embedding.

On 2026-10-25, an M4 Pro (16 GPU cores, 48 GB, macOS 27.0.1) on AC power gave
the following medians. Each process used five warmups and seven measured calls,
in Burn A → PyTorch A → PyTorch B → Burn B order for each batch size. Burn used
the pinned 0.23 development runtime through wgpu/Metal; PyTorch 2.14.1 used MPS.

| Batch | Burn A | Burn B | PyTorch A | PyTorch B |
|---|---:|---:|---:|---:|
| 1 | 0.207 s | 0.210 s | 0.138 s | 0.138 s |
| 4 | 0.779 s | 0.774 s | 0.503 s | 0.512 s |

Burn's latency was about 51–53% higher using the mean of each pair of medians.
These are warmed embedding measurements with normal desktop activity, not
isolated-system or whole-slide throughput measurements.

A separate batch-one trial replaced explicit attention with Burn's attention
API, using the same timing scope and counts in original → candidate → candidate
→ original order. The candidate measured 0.201 / 0.197 s against 0.194 / 0.195 s
for the original, so it was not retained. Kernel tracing showed the float32
candidate falling back to separate operations. An aligned half-precision
attention trial failed the unchanged 20-tile token tolerance: maximum difference
0.00523, with eight values outside the budget. Neither experiment is enabled;
the original float32 attention remains in use.

The dense layers can also be measured independently, using deterministic
float32 inputs at CONCH's actual matrix dimensions:

```sh
BURN_DEVICE=metal cargo run --locked --release --example benchmark_linear -- 4
export/.venv/bin/python export/benchmark_linear.py --batch 4
```

These calls keep inputs and weights on the GPU and include dispatch and
synchronization, excluding initialization and output readback. They test both
weight orientations and native versus explicitly flattened batch dimensions.
Every output must be finite; selected coordinates are checked against float64
scalar arithmetic. This diagnostic does not replace real-model parity tests.

On the same M4 Pro on AC power, five warmups and 21 calls per case, in
Burn/PyTorch/PyTorch/Burn process order, gave these batch-four times (mean of the two
run medians). The table uses the layouts of the current model implementations:
Burn's contiguous `[input, output]` weights and PyTorch's contiguous
`[output, input]` weights, with native three-dimensional inputs.

| Linear layer | Burn | PyTorch MPS |
|---|---:|---:|
| QKV, 1024 → 3072 | 5.31 ms | 4.17 ms |
| Output projection, 1024 → 1024 | 1.85 ms | 1.51 ms |
| MLP expansion, 1024 → 4096 | 7.15 ms | 5.55 ms |
| MLP contraction, 4096 → 1024 | 7.04 ms | 5.51 ms |

Changing weight orientation or explicitly flattening the batch did not give a
consistent Burn improvement. The remaining difference in these isolated calls
points to backend kernel selection/execution as a further optimization target;
these timings include host dispatch and synchronization, not just GPU execution.

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
