//! Burn vs torch for CellViT checkpoints: all four raw outputs and the nucleus
//! probability on each fixture tile, within the PanNuke example's float32
//! budgets. That example (`examples/parity_cellvit.rs`) covers CellViT-SAM-H.
//!
//! CellViT-256: `export/reference_cellvit.py --variant vit256`, on 256 px crops
//! of `fixtures/tiles`. Ignored by default: it needs `CellViT-256-x40.pth`.

mod support;

use path_2_burn::cellvit::{CellVit, CellVitWeights, VIT256_X40};
use path_2_burn::select_device;
use support::cellvit::{BINARY_ATOL, assert_raw_close, read_reference};
use support::{assert_close, fixture_dirs, model_store, read_f32, root};

// A logit error of BINARY_ATOL moves a two-class softmax by at most half as much.
const PROB_ATOL: f32 = BINARY_ATOL / 2.0;

#[test]
#[ignore = "needs CellViT-256-x40.pth in weights/ and its fixtures"]
fn vit256_outputs_match_torch() {
    check(
        &VIT256_X40,
        "cellvit_256",
        "run export/reference_cellvit.py --variant vit256 --out ../fixtures/cellvit_256",
    );
}

fn check(weights: &CellVitWeights, fixtures: &str, how: &str) {
    let (device, _) = select_device();
    let model = CellVit::load(&model_store(), &device, weights)
        .unwrap_or_else(|e| panic!("load {}: {e}", weights.name));
    let pixels = weights.input_size * weights.input_size;
    let tiles = fixture_dirs(&root().join("fixtures").join(fixtures), "input.bin", how);
    for tile in tiles {
        println!("{}", tile.file_name().unwrap().to_string_lossy());
        let input = read_f32(&tile.join("input.bin"));
        let raw = model.forward_raw(input.clone()).expect("forward");
        let torch = read_reference(&tile);
        assert_raw_close(&raw, &torch, pixels);

        let binary = &torch.nuclei_binary_map;
        let expected: Vec<f32> = (0..pixels)
            .map(|p| 1.0 / (1.0 + (binary[p] - binary[pixels + p]).exp()))
            .collect();
        let prob = model.predict(input).expect("predict").nuclei_prob;
        assert_close("nuclei_prob", &prob, &expected, pixels, PROB_ATOL, 0.0);
    }
}
