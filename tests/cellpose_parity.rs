//! Burn vs Cellpose for Cellpose-SAM: the input normalization and the network's
//! flows and cell probability on 256 px tiles.
//!
//! `export/reference_cellpose.py` downloads the `cpsam` checkpoint and writes
//! fixtures/cellpose_sam/. Ignored by default because it needs that checkpoint:
//!
//!   cargo test --release --test cellpose_parity -- --ignored

mod support;

use path_2_burn::cellpose::CellposeSam;
use path_2_burn::select_device;
use support::{assert_close, model_store, read_f32, read_tiles, root};

const TILE: usize = CellposeSam::TILE;
/// Float32 through 24 global-attention blocks, on flows of order 1-10.
const FLOW_ATOL: f32 = 1e-3;
const FLOW_RTOL: f32 = 1e-3;
const BATCH: usize = 4;

#[test]
#[ignore = "needs the cpsam checkpoint in weights/ and fixtures from export/reference_cellpose.py"]
fn cellpose_sam_matches_cellpose() {
    let fixtures = root().join("fixtures").join("cellpose_sam");
    let tile_bytes = TILE * TILE * 3;
    let (rgb, count) = read_tiles(&fixtures.join("rgb.bin"), tile_bytes);
    let input = read_f32(&fixtures.join("input.bin"));

    let normalized: Vec<f32> = rgb
        .chunks_exact(tile_bytes)
        .flat_map(|tile| CellposeSam::normalize_image(tile, TILE, TILE).expect("normalize"))
        .collect();
    assert_close(
        "normalized input",
        &normalized,
        &input,
        count * 3 * TILE * TILE,
        1e-5,
        1e-6,
    );

    let (device, _) = select_device();
    let model = CellposeSam::load(&model_store(), &device).expect("load Cellpose-SAM");
    let flows: Vec<f32> = input
        .chunks(BATCH * 3 * TILE * TILE)
        .flat_map(|batch| model.flows(batch.to_vec()).expect("flows"))
        .collect();
    assert_close(
        "flows",
        &flows,
        &read_f32(&fixtures.join("flows.bin")),
        count * 3 * TILE * TILE,
        FLOW_ATOL,
        FLOW_RTOL,
    );
}
