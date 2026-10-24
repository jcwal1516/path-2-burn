//! Rust InstanSeg postprocessing on the torch UNet output vs the full scripted model.
//!
//! Needs local fixtures (export/reference_instanseg.py) and weights/instanseg_postprocess.json.
//! cargo test --release --test instanseg_parity -- --nocapture

#[path = "instanseg_support/compare.rs"]
mod compare;
mod support;

use compare::{SIDE, compare_labels, fixture_tiles};
use path_2_burn::instanseg::{PostprocessParams, decode};
use std::time::Instant;
use support::{read_f32, read_i32};

fn params() -> PostprocessParams {
    PostprocessParams::load(&support::model_store())
        .unwrap_or_else(|e| panic!("{e}; run export/export_instanseg.py"))
}

#[test]
fn postprocess_matches_scripted_model_on_torch_network_output() {
    let params = params();
    println!("tile      ref  rust  matched  meanIoU  fgIoU    identical  decode");
    for tile in fixture_tiles() {
        let raw = read_f32(&tile.join("raw.bin"));
        let reference = read_i32(&tile.join("labels.bin"));
        let start = Instant::now();
        let ours = decode(&raw, SIDE, SIDE, &params).expect("decode");
        let elapsed = start.elapsed().as_secs_f64();
        let parity = compare_labels(&reference, &ours.labels);
        println!(
            "{:<8} {:>5} {:>5}  {:.4}   {:.4}   {:.5}  {:.5}    {:.0} ms",
            tile.file_name().unwrap().to_string_lossy(),
            parity.reference_instances,
            parity.rust_instances,
            parity.matched_fraction(),
            parity.mean_matched_iou,
            parity.foreground_iou,
            parity.identical_pixels,
            elapsed * 1e3,
        );
        assert!(parity.matched_fraction() >= 0.995, "{tile:?}: {parity:?}");
        assert!(parity.mean_matched_iou >= 0.99, "{tile:?}: {parity:?}");
        assert!(parity.foreground_iou >= 0.995, "{tile:?}: {parity:?}");
        // Same input maps: the port is expected to reproduce label values, not just shapes.
        assert!(parity.identical_pixels >= 0.9999, "{tile:?}: {parity:?}");
    }
}

#[test]
fn seeds_match_reference() {
    let params = params();
    for tile in fixture_tiles() {
        let reference = read_i32(&tile.join("seeds.bin"));
        let raw = read_f32(&tile.join("raw.bin"));
        let ours = path_2_burn::instanseg::find_seeds(&raw, SIDE, SIDE, &params).expect("seeds");
        let flat: Vec<i32> = ours
            .iter()
            .flat_map(|&(r, c)| [r as i32, c as i32])
            .collect();
        assert_eq!(
            flat,
            reference,
            "{tile:?}: {} rust vs {} torch seeds",
            ours.len(),
            reference.len() / 2
        );
    }
}
