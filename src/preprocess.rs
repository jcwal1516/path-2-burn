//! Input preparation shared by the models: RGB tiles to planar network input,
//! and per-channel percentile statistics.

use crate::store::ModelError;

/// HWC RGB u8 tiles of `side x side` pixels, concatenated, to NCHW float32,
/// mapping each byte with `normalize(channel, value)`. `tiles` is the exact
/// number expected, or `None` to accept any positive number.
pub(crate) fn rgb_to_nchw(
    rgb: &[u8],
    side: usize,
    tiles: Option<usize>,
    model: &'static str,
    normalize: impl Fn(usize, u8) -> f32,
) -> Result<Vec<f32>, ModelError> {
    let pixels = side * side;
    let count = rgb.len() / (pixels * 3);
    let whole = !rgb.is_empty() && rgb.len().is_multiple_of(pixels * 3);
    if !whole || tiles.is_some_and(|expected| expected != count) {
        let wanted = tiles.map_or("one or more".to_string(), |n| n.to_string());
        return Err(ModelError::InvalidInput {
            model,
            message: format!(
                "expected {wanted} {side}x{side} RGB tiles ({} bytes each), got {} bytes",
                pixels * 3,
                rgb.len()
            ),
        });
    }
    let mut out = vec![0f32; rgb.len()];
    for (tile_rgb, tile_out) in rgb
        .chunks_exact(pixels * 3)
        .zip(out.chunks_exact_mut(pixels * 3))
    {
        for (i, px) in tile_rgb.chunks_exact(3).enumerate() {
            for (c, &value) in px.iter().enumerate() {
                tile_out[c * pixels + i] = normalize(c, value);
            }
        }
    }
    Ok(out)
}

/// `np.percentile` (linear method) of each channel of a whole HWC RGB image,
/// at the two percentiles in `q` (0–100), in 0–255 units.
pub(crate) fn channel_percentiles(
    rgb: &[u8],
    width: usize,
    height: usize,
    model: &'static str,
    q: [f64; 2],
) -> Result<[[f64; 2]; 3], ModelError> {
    let pixels = width * height;
    if pixels == 0 || rgb.len() != pixels * 3 {
        return Err(ModelError::InvalidInput {
            model,
            message: format!(
                "expected a non-empty {width}x{height} RGB image ({} bytes), got {}",
                pixels * 3,
                rgb.len()
            ),
        });
    }
    let mut histograms = [[0u64; 256]; 3];
    for px in rgb.chunks_exact(3) {
        for (c, &v) in px.iter().enumerate() {
            histograms[c][v as usize] += 1;
        }
    }
    Ok(histograms.map(|histogram| q.map(|q| percentile(&histogram, pixels, q))))
}

/// `np.percentile(values, q)` (linear method) from a histogram of `n` u8 values.
fn percentile(histogram: &[u64; 256], n: usize, q: f64) -> f64 {
    let position = q / 100.0 * (n - 1) as f64;
    let below = position.floor();
    let fraction = position - below;
    let below = below as u64;
    let lower = value_at(histogram, below);
    let upper = value_at(histogram, (below + 1).min(n as u64 - 1));
    lower + (upper - lower) * fraction
}

/// The `rank`-th smallest value (0-based).
fn value_at(histogram: &[u64; 256], rank: u64) -> f64 {
    let mut seen = 0u64;
    for (value, &count) in histogram.iter().enumerate() {
        seen += count;
        if seen > rank {
            return value as f64;
        }
    }
    255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_tiles_become_planar_channels() {
        // Four pixels in HWC order; channel c of byte v maps to 10 * c + v.
        let rgb = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let normalize = |c: usize, v: u8| (10 * c) as f32 + v as f32;
        // One 2x2 tile: planes R, G, B.
        let tile = rgb_to_nchw(&rgb, 2, Some(1), "test", normalize).unwrap();
        assert_eq!(
            tile,
            [
                1.0, 4.0, 7.0, 10.0, 12.0, 15.0, 18.0, 21.0, 23.0, 26.0, 29.0, 32.0
            ]
        );
        // Four 1x1 tiles, each planar on its own.
        let batch = rgb_to_nchw(&rgb, 1, None, "test", normalize).unwrap();
        assert_eq!(
            batch,
            [
                1.0, 12.0, 23.0, 4.0, 15.0, 26.0, 7.0, 18.0, 29.0, 10.0, 21.0, 32.0
            ]
        );
        assert!(rgb_to_nchw(&rgb, 1, Some(3), "test", normalize).is_err());
        assert!(rgb_to_nchw(&rgb[..5], 1, None, "test", normalize).is_err());
        assert!(rgb_to_nchw(&[], 1, None, "test", normalize).is_err());
    }
}
