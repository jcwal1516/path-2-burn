//! Native port of the postprocessing in InstanSeg's scripted forward for the
//! single-head brightfield layout `[field_x, field_y, sigma_0, sigma_1, seed]`.

mod crops;
mod merge;
mod seeds;

use super::CHANNELS;
use super::params::PostprocessParams;
use crops::{Crop, Fields};

/// Checks that `raw` holds a `[5, height, width]` network output.
pub(super) fn check_len(raw: &[f32], width: usize, height: usize) -> Result<(), String> {
    if raw.len() == CHANNELS * width * height {
        return Ok(());
    }
    Err(format!(
        "expected {CHANNELS}x{height}x{width} values, got {}",
        raw.len()
    ))
}

/// Network output `[5, H, W]` to a label map (`0` = background). Labels are the
/// smallest seed index of each merged instance, so they are not consecutive.
pub(super) fn decode(
    raw: &[f32],
    width: usize,
    height: usize,
    params: &PostprocessParams,
) -> Result<Vec<u32>, String> {
    check_len(raw, width, height)?;
    // Smaller images would make the scripted model index windows from the far edge.
    if width < 2 * params.window_size || height < 2 * params.window_size {
        return Err(format!(
            "image must be at least {0}x{0} pixels",
            2 * params.window_size
        ));
    }
    let plane = width * height;
    let seeds = find_seeds(raw, width, height, params);
    if seeds.is_empty() {
        return Ok(vec![0; plane]);
    }
    let fields = Fields::new(raw, width, height);
    let crops = build_crops(&fields, &seeds, params);
    let side = 2 * crops::half_window(params, width, height);
    Ok(merge::merge(
        &crops,
        side,
        width,
        height,
        params.mask_threshold,
        params.overlap_threshold,
    ))
}

/// Local maxima of the seed map `raw[4] / 15 + 0.5`, (row, col) in row-major order.
pub(super) fn find_seeds(
    raw: &[f32],
    width: usize,
    height: usize,
    params: &PostprocessParams,
) -> Vec<(usize, usize)> {
    let plane = width * height;
    let seed_map: Vec<f32> = raw[4 * plane..5 * plane]
        .iter()
        .map(|&v| v / 15.0 + 0.5)
        .collect();
    seeds::peak_local_max(
        &seed_map,
        width,
        height,
        params.peak_distance,
        params.seed_threshold,
    )
}

/// Windows are independent; build them in order-preserving chunks across threads.
fn build_crops(
    fields: &Fields<'_>,
    seeds: &[(usize, usize)],
    params: &PostprocessParams,
) -> Vec<Crop> {
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(seeds.len());
    let chunk = seeds.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = seeds
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|&s| crops::build(fields, s, params))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("crop worker panicked"))
            .collect()
    })
}
