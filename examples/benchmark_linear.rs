//! CONCH dense-layer shapes, with resident float32 tensors and synchronized calls.
//! BURN_DEVICE=metal cargo run --locked --release --example benchmark_linear -- [batch=1]
//! Matches export/benchmark_linear.py. Initialization and readback are not timed.

use burn::prelude::*;
use burn::tensor::module::linear;
use path_2_burn::select_device;
use std::hint::black_box;
use std::time::Instant;

const TOKENS: usize = 785;
const WARMUPS: usize = 5;
const REPEATS: usize = 21;

fn input_value(i: usize) -> f32 {
    (((i * 13 + 7) % 127) as f32 - 63.0) / 256.0
}

fn weight_value(i: usize) -> f32 {
    (((i * 17 + 3) % 127) as f32 - 63.0) / 256.0
}

fn bias_value(i: usize) -> f32 {
    ((i % 17) as f32 - 8.0) / 256.0
}

fn main() {
    assert!(matches!(
        std::env::var("BURN_DEVICE").as_deref(),
        Ok("metal")
    ));
    let batch: usize = std::env::args().nth(1).map_or(1, |s| s.parse().unwrap());
    assert!([1, 4].contains(&batch));
    let (device, _) = select_device();
    for (name, k, n) in [
        ("qkv", 1024, 3072),
        ("projection", 1024, 1024),
        ("mlp_expand", 1024, 4096),
        ("mlp_contract", 4096, 1024),
    ] {
        let x = Tensor::<3>::from_data(
            TensorData::new(
                (0..batch * TOKENS * k).map(input_value).collect(),
                [batch, TOKENS, k],
            ),
            &device,
        );
        let bias = Tensor::<1>::from_data(
            TensorData::new((0..n).map(bias_value).collect(), [n]),
            &device,
        );
        // Both physical layouts represent the same logical [k, n] matrix.
        let row = Tensor::<2>::from_data(
            TensorData::new(
                (0..k * n)
                    .map(|i| weight_value((i % n) * k + i / n))
                    .collect(),
                [k, n],
            ),
            &device,
        );
        let column = Tensor::<2>::from_data(
            TensorData::new((0..n * k).map(weight_value).collect(), [n, k]),
            &device,
        )
        .transpose();
        for flatten in [false, true] {
            if flatten && batch == 1 {
                continue;
            }
            for (layout, weight) in [("row", &row), ("column", &column)] {
                let run = || {
                    if flatten {
                        linear(
                            x.clone().reshape([batch * TOKENS, k]),
                            weight.clone(),
                            Some(bias.clone()),
                        )
                        .reshape([batch, TOKENS, n])
                    } else {
                        linear(x.clone(), weight.clone(), Some(bias.clone()))
                    }
                };
                for _ in 0..WARMUPS {
                    let out = run();
                    device.sync().unwrap();
                    black_box(&out);
                }
                let mut seconds = Vec::with_capacity(REPEATS);
                let mut last = None;
                for _ in 0..REPEATS {
                    let start = Instant::now();
                    let out = run();
                    device.sync().unwrap();
                    seconds.push(start.elapsed().as_secs_f64());
                    last = Some(out);
                }
                let data = last.unwrap().into_data();
                let values = data.as_slice::<f32>().unwrap();
                assert!(values.iter().all(|x| x.is_finite()));
                let mut max_error = 0.0_f64;
                for b in 0..batch {
                    for t in [0, TOKENS / 2, TOKENS - 1] {
                        for j in [0, n / 2, n - 1] {
                            let expected = bias_value(j) as f64
                                + (0..k)
                                    .map(|i| {
                                        input_value((b * TOKENS + t) * k + i) as f64
                                            * weight_value(j * k + i) as f64
                                    })
                                    .sum::<f64>();
                            let error = (values[(b * TOKENS + t) * n + j] as f64 - expected).abs();
                            max_error = max_error.max(error);
                        }
                    }
                }
                assert!(max_error <= 1e-5, "{name}: {max_error}");
                let mut sorted = seconds.clone();
                sorted.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({
                        "framework": "burn", "operation": name, "batch": batch,
                        "tokens": TOKENS, "k": k, "n": n, "weight_layout": layout,
                        "flatten": flatten, "warmups": WARMUPS, "repeats": REPEATS,
                        "scope": "resident_linear_dispatch_and_sync", "dtype": "float32",
                        "seconds": seconds, "median_seconds": sorted[REPEATS / 2],
                        "sample_max_error": max_error,
                    })
                );
            }
        }
    }
}
