//! Burn vs Cellpose for Cellpose-SAM: the input normalization and the network's
//! flows and cell probability on 256 px tiles.
//!
//! `export/reference_cellpose.py` downloads the `cpsam` checkpoint and writes
//! fixtures/cellpose_sam/. Ignored by default because it needs that checkpoint:
//!
//!   cargo test --release --test cellpose_parity -- --ignored

mod support;

use path_2_burn::cellpose::{CellposeSam, DynamicsOptions, compute_masks};
use path_2_burn::select_device;
use support::{assert_close, model_store, read_f32, read_le, read_tiles, root};

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
    for (tile, flow) in flows.chunks_exact(3 * TILE * TILE).enumerate() {
        let actual = compute_masks(flow, TILE, TILE, &DynamicsOptions::default()).expect("masks");
        let reference: Vec<u32> = read_le(
            &fixtures
                .join("masks")
                .join(format!("tile_{tile:02}"))
                .join("masks.bin"),
            u32::from_le_bytes,
        );
        assert_eq!(reference.len(), actual.len());
        let count = *actual.iter().max().unwrap() as usize;
        assert_eq!(
            count,
            *reference.iter().max().unwrap() as usize,
            "tile {tile}: mask count"
        );
        let mut intersections = vec![vec![0usize; count + 1]; count + 1];
        for (&a, &r) in actual.iter().zip(&reference) {
            intersections[a as usize][r as usize] += 1;
        }
        let mut matched = vec![false; count + 1];
        for row in intersections.iter().skip(1) {
            let (label, &overlap) = row
                .iter()
                .enumerate()
                .skip(1)
                .max_by_key(|(_, n)| *n)
                .unwrap();
            assert!(!matched[label], "tile {tile}: merged masks");
            matched[label] = true;
            let actual_area: usize = row.iter().sum();
            let reference_area: usize = intersections.iter().map(|row| row[label]).sum();
            assert!(
                overlap as f64 >= 0.9 * actual_area.max(reference_area) as f64,
                "tile {tile}: mask shape changed"
            );
        }
        // As for CellViT, allow small float32 boundary shifts while preserving
        // every instance. Decoding fixed reference flows is bit-exact separately.
        let boundary_changes = actual
            .iter()
            .zip(&reference)
            .filter(|&(&a, &r)| (a == 0) != (r == 0))
            .count();
        println!("tile {tile}: {count} masks, {boundary_changes} foreground changes");
        assert!(
            boundary_changes as f64 / (TILE * TILE) as f64 <= 1e-4,
            "tile {tile}: foreground changed"
        );
    }
}
