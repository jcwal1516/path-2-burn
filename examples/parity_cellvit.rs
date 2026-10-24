//! Check CellViT-SAM-H outputs against the torch reference fixtures.
//!
//! cargo run --release --example parity_cellvit -- [weights_dir] [fixtures_dir] [max_tiles]

#[path = "../tests/support/mod.rs"]
mod support;

use path_2_burn::cellvit::{CellVit, NUCLEI_CLASSES, SAM_H_X40, TISSUE_CLASSES};
use path_2_burn::{ModelStore, select_device};
use std::path::PathBuf;
use std::time::Instant;
use support::cellvit::{assert_raw_close, read_reference};
use support::{fixture_dirs, read_f32};

fn argmax_disagreements(burn: &[f32], torch: &[f32], classes: usize, pixels: usize) -> usize {
    let argmax = |m: &[f32], p: usize| {
        (1..classes).fold(0, |best, c| {
            if m[c * pixels + p] > m[best * pixels + p] {
                c
            } else {
                best
            }
        })
    };
    (0..pixels)
        .filter(|&p| argmax(burn, p) != argmax(torch, p))
        .count()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights = PathBuf::from(args.first().map(String::as_str).unwrap_or("weights"));
    let fixtures = PathBuf::from(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("fixtures/cellvit"),
    );

    let mut tiles = fixture_dirs(&fixtures, "cells.json", "run export/reference_cellvit.py");
    if let Some(limit) = args.get(2) {
        let limit = limit
            .parse::<usize>()
            .expect("max_tiles must be an integer");
        assert!(limit > 0, "max_tiles must be positive");
        tiles.truncate(limit);
    }

    let (device, choice) = select_device();
    println!("device: {device:?} ({choice})");
    let store = ModelStore::open(&weights).expect("open model store");
    let t = Instant::now();
    let model = CellVit::load(&store, &device, &SAM_H_X40).expect("load cellvit");
    println!("load (incl. sha256): {:.1}s", t.elapsed().as_secs_f32());

    let pixels = SAM_H_X40.input_size * SAM_H_X40.input_size;
    let mut worst = 0f32;
    for tile in &tiles {
        let input = read_f32(&tile.join("input.bin"));
        let t = Instant::now();
        let raw = model.forward_raw(input).expect("forward");
        println!(
            "{} forward+readback {:.2}s",
            tile.file_name().unwrap().to_string_lossy(),
            t.elapsed().as_secs_f32()
        );
        let torch = read_reference(tile);
        worst = worst.max(assert_raw_close(&raw, &torch, pixels));
        let tissue_disagreements = argmax_disagreements(
            &raw.tissue_types,
            &torch.tissue_types,
            TISSUE_CLASSES.len(),
            1,
        );
        println!(
            "    argmax differences: nucleus {}/{pixels}, type {}/{pixels}, tissue {tissue_disagreements}/1",
            argmax_disagreements(&raw.nuclei_binary_map, &torch.nuclei_binary_map, 2, pixels),
            argmax_disagreements(
                &raw.nuclei_type_map,
                &torch.nuclei_type_map,
                NUCLEI_CLASSES.len(),
                pixels
            ),
        );
        assert_eq!(tissue_disagreements, 0, "tissue prediction changed");
    }
    println!(
        "PASS: {} tiles, four outputs each; worst max|d| {worst:.6}",
        tiles.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use support::assert_close;
    use support::cellvit::{BINARY_ATOL, HV_ATOL, RTOL};

    /// [`assert_close`] with CellViT's relative tolerance.
    fn diff(name: &str, burn: &[f32], torch: &[f32], expected: usize, atol: f32) -> f32 {
        assert_close(name, burn, torch, expected, atol, RTOL)
    }

    #[test]
    fn comparison_rejects_invalid_or_inaccurate_outputs() {
        for (label, actual, reference) in [
            ("empty", vec![], vec![]),
            ("nan output", vec![f32::NAN], vec![0.0]),
            ("infinite output", vec![f32::INFINITY], vec![0.0]),
            ("nan reference", vec![0.0], vec![f32::NAN]),
            ("wrong value", vec![0.006], vec![0.0]),
        ] {
            assert!(
                std::panic::catch_unwind(|| diff(
                    label,
                    &actual,
                    &reference,
                    reference.len(),
                    BINARY_ATOL
                ))
                .is_err(),
                "accepted {label}"
            );
        }
        diff(
            "float32 rounding",
            &[100.005, 0.0005],
            &[100.0, 0.0],
            2,
            BINARY_ATOL,
        );
        assert!(
            std::panic::catch_unwind(|| diff("shape", &[0.0], &[0.0], 2, BINARY_ATOL)).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| diff("HV drift", &[0.00015], &[0.0], 1, HV_ATOL)).is_err()
        );
    }

    #[test]
    fn argmax_uses_the_first_class_on_ties() {
        assert_eq!(argmax_disagreements(&[0.0, 1.0], &[1.0, 1.0], 2, 1), 1);
    }
}
