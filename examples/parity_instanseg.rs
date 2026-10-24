//! InstanSeg on burn (network) + Rust postprocessing vs the scripted torch model, with timings.
//! Then percentile-normalized input vs the instanseg package itself, when its
//! fixtures exist (export/reference_instanseg_normalized.py).
//!
//! cargo run --release --example parity_instanseg

#[path = "../tests/instanseg_support/compare.rs"]
mod compare;
#[path = "../tests/support/mod.rs"]
mod support;

use compare::{SIDE, compare_labels, fixture_tiles};
use path_2_burn::instanseg::{InputNorm, InstanSeg, decode};
use path_2_burn::{ModelStore, select_device};
use std::path::{Path, PathBuf};
use std::time::Instant;
use support::{read_f32, read_i32};

fn main() {
    let weights = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    let (device, choice) = select_device();
    println!("device: {device:?} ({choice})");
    let store = ModelStore::open(&weights).expect("open model store");
    let t = Instant::now();
    let model = InstanSeg::load(&store, &device).expect("load instanseg");
    println!("load (incl. sha256): {:.2}s", t.elapsed().as_secs_f32());

    let tiles = fixture_tiles();
    let fixtures = compare::fixtures_dir();
    let mut densest = (0usize, tiles[0].clone());
    println!(
        "tile     net s   raw max|d| (fx fy s0 s1 seed)                   ref  rust  matched meanIoU  fgIoU   identical  post ms"
    );
    for (i, tile) in tiles.iter().enumerate() {
        let name = tile.file_name().unwrap().to_string_lossy().to_string();
        let rgb = tile_rgb(&fixtures, &name);
        let t = Instant::now();
        let raw = model
            .predict(rgb.as_raw(), SIDE, SIDE, &InputNorm::UNIT)
            .expect("predict");
        let net = t.elapsed().as_secs_f64();
        let reference_raw = read_f32(&tile.join("raw.bin"));
        let plane = SIDE * SIDE;
        let diffs: Vec<f32> = (0..5)
            .map(|c| {
                raw[c * plane..(c + 1) * plane]
                    .iter()
                    .zip(&reference_raw[c * plane..(c + 1) * plane])
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
            })
            .collect();
        let t = Instant::now();
        let labels = decode(&raw, SIDE, SIDE, model.params()).expect("decode");
        let post = t.elapsed().as_secs_f64();
        let p = compare_labels(&read_i32(&tile.join("labels.bin")), &labels.labels);
        if p.reference_instances > densest.0 {
            densest = (p.reference_instances, tile.clone());
        }
        println!(
            "{name:<8} {net:>6.2}{}  {:.1e} {:.1e} {:.1e} {:.1e} {:.1e}  {:>5} {:>5}  {:.4}  {:.4}  {:.5}  {:.5}  {:>5.0}",
            if i == 0 { "*" } else { " " },
            diffs[0],
            diffs[1],
            diffs[2],
            diffs[3],
            diffs[4],
            p.reference_instances,
            p.rust_instances,
            p.matched_fraction(),
            p.mean_matched_iou,
            p.foreground_iou,
            p.identical_pixels,
            post * 1e3,
        );
    }
    println!("* first network call on this shape (includes autotune unless cached)");
    percentile_parity(&model, &tiles, &fixtures);

    let rgb = tile_rgb(&fixtures, &densest.1.file_name().unwrap().to_string_lossy());
    model
        .segment(rgb.as_raw(), SIDE, SIDE, &InputNorm::UNIT)
        .expect("warmup");
    let runs = 5;
    let (mut net, mut post) = (0f64, 0f64);
    for _ in 0..runs {
        let t = Instant::now();
        let raw = model
            .predict(rgb.as_raw(), SIDE, SIDE, &InputNorm::UNIT)
            .expect("predict");
        net += t.elapsed().as_secs_f64();
        let t = Instant::now();
        decode(&raw, SIDE, SIDE, model.params()).expect("decode");
        post += t.elapsed().as_secs_f64();
    }
    println!(
        "steady state on {} ({} instances) over {runs} runs: network+readback {:.3}s, postprocess {:.3}s per 1024x1024 tile",
        densest.1.file_name().unwrap().to_string_lossy(),
        densest.0,
        net / runs as f64,
        post / runs as f64
    );
}

/// The fixture tile `name` as RGB.
fn tile_rgb(fixtures: &Path, name: &str) -> image::RgbImage {
    image::open(fixtures.join("tiles").join(format!("{name}.png")))
        .expect("tile png")
        .to_rgb8()
}

/// Percentile-normalized burn path vs instanseg's `eval_small_image(normalise=True)`,
/// plus how far the old `/255` input (torch fixture) was from that reference.
fn percentile_parity(model: &InstanSeg, tiles: &[PathBuf], fixtures: &Path) {
    println!("\npercentile-normalized input vs instanseg eval_small_image(normalise=True)");
    println!(
        "tile     stats|d|   ref  rust  matched meanIoU  fgIoU    identical   | /255 input: inst  matched"
    );
    for tile in tiles {
        let name = tile.file_name().unwrap().to_string_lossy().to_string();
        let reference_path = tile.join("labels_pnorm.bin");
        if !reference_path.is_file() {
            println!("{name:<8} no labels_pnorm.bin; run export/reference_instanseg_normalized.py");
            continue;
        }
        let rgb = tile_rgb(fixtures, &name);
        let norm = InputNorm::percentile(rgb.as_raw(), SIDE, SIDE).expect("percentile stats");
        let stats: serde_json::Value = serde_json::from_slice(
            &std::fs::read(tile.join("stats_pnorm.json")).expect("stats_pnorm.json"),
        )
        .expect("stats json");
        let stats_diff = (0..3)
            .map(|c| {
                let low = stats["low"][c].as_f64().expect("low") as f32;
                let high = stats["high"][c].as_f64().expect("high") as f32;
                (norm.low()[c] - low)
                    .abs()
                    .max((norm.low()[c] + norm.range()[c] - high).abs())
            })
            .fold(0f32, f32::max);
        let reference = read_i32(&reference_path);
        let ours = model
            .segment(rgb.as_raw(), SIDE, SIDE, &norm)
            .expect("segment");
        let p = compare_labels(&reference, &ours.labels);
        let old: Vec<u32> = read_i32(&tile.join("labels.bin"))
            .into_iter()
            .map(|l| l.max(0) as u32)
            .collect();
        let o = compare_labels(&reference, &old);
        println!(
            "{name:<8} {stats_diff:>8.1e}  {:>5} {:>5}  {:.4}  {:.4}  {:.5}  {:.5}   | {:>5}  {:.4}",
            p.reference_instances,
            p.rust_instances,
            p.matched_fraction(),
            p.mean_matched_iou,
            p.foreground_iou,
            p.identical_pixels,
            o.rust_instances,
            o.matched_fraction(),
        );
    }
}
