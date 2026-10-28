//! CONCH v1.5 float32 host-to-host embedding latency, matching benchmark_conch.py.
//!
//! BURN_DEVICE=metal cargo run --locked --release --example benchmark_conch -- [batch=1] [warmups=5] [repeats=7]
//! Set BURN_BENCH_ENCODER=virchow2 to benchmark another registered tile encoder.
//! Loading, fixture I/O and normalization are outside the timer. Each measured
//! call copies normalized CPU input, uploads it and reads back one embedding.

#[path = "../tests/support/mod.rs"]
mod support;

use path_2_burn::encoders::{ALL, TileEncoder};
use path_2_burn::select_device;
use std::hint::black_box;
use std::time::Instant;
use support::{assert_close_scaled, model_store, read_f32, read_tiles, root};

fn main() {
    assert!(
        matches!(std::env::var("BURN_DEVICE").as_deref(), Ok("metal")),
        "this benchmark requires BURN_DEVICE=metal (Apple GPU)"
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() <= 3, "expected batch, warmups, repeats");
    let number = |index: usize, default| {
        args.get(index)
            .map_or(default, |s| s.parse::<usize>().expect("positive integer"))
    };
    let (batch, warmups, repeats) = (number(0, 1), number(1, 5), number(2, 7));
    assert!(batch > 0 && warmups > 0 && repeats > 0);
    let name = std::env::var("BURN_BENCH_ENCODER").unwrap_or_else(|_| "conch_v1_5".into());
    let encoder = ALL
        .iter()
        .copied()
        .find(|encoder| encoder.name == name)
        .unwrap_or_else(|| panic!("unknown tile encoder: {name}"));
    let fixtures = root().join("fixtures/encoders");
    let tile_bytes = 3 * encoder.tile * encoder.tile;
    let tile_file = if encoder.tile == 224 {
        "tiles.bin".into()
    } else {
        format!("tiles_{}.bin", encoder.tile)
    };
    let (rgb, available) = read_tiles(&fixtures.join(tile_file), tile_bytes);
    assert!(batch <= available, "batch exceeds fixture count");
    let (device, choice) = select_device();
    eprintln!("device: {device:?} ({choice})");
    let model = TileEncoder::load(&model_store(), &device, encoder).expect("load tile encoder");
    let input = model
        .normalize_rgb(&rgb[..batch * tile_bytes])
        .expect("normalize");
    let reference = read_f32(&fixtures.join(encoder.name).join("embedding.bin"));
    let expected = batch * encoder.embedding_dim();
    // Check token values outside the timer, using the existing parity-test tolerance.
    let grid = encoder.tile / encoder.vit.patch_size;
    let token_count = batch * (encoder.vit.prefix_tokens() + grid * grid) * encoder.vit.embed_dim;
    let token_reference = read_f32(&fixtures.join(encoder.name).join("tokens.bin"));
    assert_close_scaled(
        "tokens",
        &model.tokens(input.clone()).expect("tokens"),
        &token_reference[..token_count],
        token_count,
        2e-4,
        1e-3,
    );
    for _ in 0..warmups {
        black_box(model.embed(input.clone()).expect("warmup"));
    }
    let mut seconds = Vec::with_capacity(repeats);
    let mut last = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        last = model.embed(input.clone()).expect("embedding");
        seconds.push(start.elapsed().as_secs_f64());
        black_box(&last);
    }
    assert_close_scaled(
        "embedding",
        &last,
        &reference[..expected],
        expected,
        2e-4,
        1e-3,
    );
    let mut sorted = seconds.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[(repeats - 1) / 2] + sorted[repeats / 2]) / 2.0;
    println!(
        "{}",
        serde_json::json!({
            "framework": "burn", "device": format!("{device:?}"), "model": encoder.name,
            "dtype": "float32", "scope": "normalized_cpu_input_to_cpu_embedding",
            "batch": batch, "warmups": warmups, "repeats": repeats, "seconds": seconds,
            "median_seconds": median, "images_per_second": batch as f64 / median,
        })
    );
}
