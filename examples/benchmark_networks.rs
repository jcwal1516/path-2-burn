//! Float32 host-to-host timings for segmentation networks and TITAN.
//!
//! BURN_DEVICE=metal cargo run --release --example benchmark_networks -- MODEL [warmups=5] [repeats=7]
//! Uses the first reference fixture, checks raw outputs outside the timer, and
//! excludes loading, fixture I/O, normalization and instance postprocessing.
//! Tile encoders use benchmark_conch, which also supports batches.

#[path = "../tests/support/mod.rs"]
mod support;

use path_2_burn::{
    cellpose::CellposeSam, cellvit, hovernet, instanseg::InstanSeg, stardist, titan::Titan,
};
use std::{hint::black_box, path::PathBuf, time::Instant};
use support::{assert_close, fixture_dirs, model_store, read_f32, read_le, root, scale};

struct Benchmark {
    name: String,
    warmups: usize,
    repeats: usize,
}

impl Benchmark {
    fn run<T>(
        &self,
        fixture: &str,
        input_shape: &[usize],
        mut forward: impl FnMut() -> T,
        check: impl Fn(&T),
    ) {
        check(&forward());
        for _ in 0..self.warmups {
            black_box(forward());
        }
        let mut seconds = Vec::with_capacity(self.repeats);
        let mut last = None;
        for _ in 0..self.repeats {
            let start = Instant::now();
            let output = forward();
            seconds.push(start.elapsed().as_secs_f64());
            last = Some(black_box(output));
        }
        check(last.as_ref().unwrap());
        let mut sorted = seconds.clone();
        sorted.sort_by(f64::total_cmp);
        let median = (sorted[(self.repeats - 1) / 2] + sorted[self.repeats / 2]) / 2.0;
        println!(
            "{}",
            serde_json::json!({
                "framework": "burn", "device": "metal", "model": self.name,
                "dtype": "float32", "scope": "prepared_cpu_input_to_raw_cpu_output",
                "fixture": fixture, "input_shape": input_shape, "batch": 1,
                "warmups": self.warmups, "repeats": self.repeats, "seconds": seconds,
                "median_seconds": median,
            })
        );
    }
}

fn first_fixture(directory: &str) -> PathBuf {
    fixture_dirs(
        &root().join("fixtures").join(directory),
        "input.bin",
        "generate reference fixtures",
    )
    .remove(0)
}

fn main() {
    assert_eq!(std::env::var("BURN_DEVICE").as_deref(), Ok("metal"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        (1..=3).contains(&args.len()),
        "expected model, warmups, repeats"
    );
    let number = |index, default| {
        args.get(index).map_or(default, |s: &String| {
            s.parse::<usize>().expect("positive integer")
        })
    };
    let bench = Benchmark {
        name: args[0].clone(),
        warmups: number(1, 5),
        repeats: number(2, 7),
    };
    assert!(bench.warmups > 0 && bench.repeats > 0);
    let (device, choice) = path_2_burn::select_device();
    eprintln!("device: {device:?} ({choice})");
    let store = model_store();
    match bench.name.as_str() {
        "cellvit_sam_h" | "cellvit_256" => {
            let (weights, folder) = if bench.name == "cellvit_sam_h" {
                (&cellvit::SAM_H_X40, "cellvit")
            } else {
                (&cellvit::VIT256_X40, "cellvit_256")
            };
            let fixture = first_fixture(folder);
            let input = read_f32(&fixture.join("input.bin"));
            let reference = support::cellvit::read_reference(&fixture);
            let model = cellvit::CellVit::load(&store, &device, weights).expect("load CellViT");
            let n = weights.input_size;
            bench.run(
                &fixture.display().to_string(),
                &[1, 3, n, n],
                || model.forward_raw(input.clone()).expect("CellViT output"),
                |actual| {
                    support::cellvit::assert_raw_close(actual, &reference, n * n);
                },
            );
        }
        name if name.starts_with("hovernet_") => {
            let weights = [
                hovernet::PANNUKE,
                hovernet::MONUSAC,
                hovernet::CONSEP,
                hovernet::CONSEP_NOTYPE,
                hovernet::KUMAR,
                hovernet::CPM17,
            ]
            .into_iter()
            .find(|w| w.name == name)
            .expect("known HoVer-Net");
            let fixture = root().join("fixtures").join(name);
            let n = weights.mode.input_size();
            let rgb = std::fs::read(fixture.join("rgb.bin")).expect("RGB fixture");
            let model =
                hovernet::HoverNet::load(&store, &device, &weights).expect("load HoVer-Net");
            let input = model.normalize_rgb(&rgb[..3 * n * n]).expect("normalize");
            let np = read_f32(&fixture.join("np.bin"));
            let hv = read_f32(&fixture.join("hv.bin"));
            let tp = weights
                .types
                .map(|_| read_f32(&fixture.join("tp.bin")))
                .unwrap_or_default();
            let atol = 2e-4 * scale(&[&np, &hv, &tp]);
            let pixels = weights.mode.output_size().pow(2);
            bench.run(
                &fixture.display().to_string(),
                &[1, 3, n, n],
                || model.forward_raw(input.clone()).expect("HoVer-Net output"),
                |actual| {
                    assert_close("np", &actual.np, &np[..2 * pixels], 2 * pixels, atol, 1e-4);
                    assert_close("hv", &actual.hv, &hv[..2 * pixels], 2 * pixels, atol, 1e-4);
                    if let Some(types) = weights.types {
                        assert_close(
                            "tp",
                            &actual.tp,
                            &tp[..types * pixels],
                            types * pixels,
                            atol,
                            1e-4,
                        );
                    } else {
                        assert!(actual.tp.is_empty());
                    }
                },
            );
        }
        "instanseg_net" => {
            let fixture = first_fixture("instanseg");
            let input = read_f32(&fixture.join("input.bin"));
            let reference = read_f32(&fixture.join("raw.bin"));
            let model = InstanSeg::load(&store, &device).expect("load InstanSeg");
            bench.run(
                &fixture.display().to_string(),
                &[1, 3, 1024, 1024],
                || model.forward_raw(input.clone()).expect("InstanSeg output"),
                |actual| {
                    assert_close("raw", actual, &reference, 5 * 1024 * 1024, 1e-3, 1e-4);
                },
            );
        }
        "cellpose_sam" => {
            let fixture = root().join("fixtures/cellpose_sam");
            let size = 3 * CellposeSam::TILE.pow(2);
            let input = read_f32(&fixture.join("input.bin"))[..size].to_vec();
            let reference = read_f32(&fixture.join("flows.bin"));
            let model = CellposeSam::load(&store, &device).expect("load Cellpose-SAM");
            bench.run(
                &fixture.display().to_string(),
                &[1, 3, 256, 256],
                || model.flows(input.clone()).expect("Cellpose flows"),
                |actual| {
                    assert_close("flows", actual, &reference[..size], size, 1e-3, 1e-3);
                },
            );
        }
        name if name.starts_with("stardist_") => {
            let spec = [
                stardist::HE,
                stardist::FLUORESCENCE,
                stardist::FLUORESCENCE_PAPER,
            ]
            .into_iter()
            .find(|s| s.name == name)
            .expect("known StarDist");
            let fixture = first_fixture(&format!("stardist/{name}"));
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
            let height = meta["crop_yxhw"][2].as_u64().unwrap() as usize;
            let width = meta["crop_yxhw"][3].as_u64().unwrap() as usize;
            let input = read_f32(&fixture.join("input.bin"));
            let prob = read_f32(&fixture.join("probability.bin"));
            let dist = read_f32(&fixture.join("distances.bin"));
            let model = stardist::StarDist::load(&store, &device, &spec).expect("load StarDist");
            let pixels = (height / spec.grid) * (width / spec.grid);
            bench.run(
                &fixture.display().to_string(),
                &[1, spec.channels, height, width],
                || {
                    model
                        .predict(input.clone(), height, width)
                        .expect("StarDist output")
                },
                |actual| {
                    assert_close(
                        "probability",
                        &actual.probability,
                        &prob,
                        pixels,
                        1e-4,
                        1e-4,
                    );
                    assert_close(
                        "distances",
                        &actual.distances,
                        &dist,
                        pixels * 32,
                        1e-3,
                        1e-4,
                    );
                },
            );
        }
        "titan" => {
            let fixture = root().join("fixtures/titan/patches");
            let input = read_f32(&fixture.join("features.bin"));
            let coords: Vec<i64> = read_le(&fixture.join("coords.bin"), i64::from_le_bytes);
            assert_eq!(coords.len() % 2, 0);
            let coords: Vec<[i64; 2]> = coords.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
            let patch_size = meta["patch_size_lv0"].as_u64().unwrap() as u32;
            let reference = read_f32(&fixture.join("embedding.bin"));
            let model = Titan::load(&store, &device).expect("load TITAN");
            bench.run(
                &fixture.display().to_string(),
                &[1, coords.len(), input.len() / coords.len()],
                || {
                    model
                        .encode_slide_from_patch_features(input.clone(), &coords, patch_size)
                        .expect("TITAN embedding")
                },
                |actual| {
                    assert_close("embedding", actual, &reference, 768, 1e-5, 1e-4);
                },
            );
        }
        other => panic!("unknown model: {other}; tile encoders use benchmark_conch"),
    }
}
