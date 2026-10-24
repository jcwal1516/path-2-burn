//! Network input normalization.
//!
//! instanseg's `percentile_normalize` (its `eval_*` default, and the model's
//! bioimage.io `scale_range`): per channel, `(v - p0.1) / max(1e-3, p99.9 - p0.1)`
//! on raw 0–255 values, percentiles with `np.percentile`'s linear interpolation,
//! no clipping. `eval_medium_image` takes the statistics over the whole image
//! before tiling, so they are a value computed once and applied to every tile.

use crate::preprocess::channel_percentiles;
use crate::store::ModelError;

const PERCENTILE: f64 = 0.1;
const EPSILON: f64 = 1e-3;

/// Per-channel affine map from RGB u8 to the network's input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputNorm {
    low: [f32; 3],
    range: [f32; 3],
}

impl InputNorm {
    /// `v / 255`, the contract the reference fixtures were generated with.
    pub const UNIT: Self = Self {
        low: [0.0; 3],
        range: [255.0; 3],
    };

    /// instanseg's percentile statistics over a whole HWC RGB image.
    pub fn percentile(rgb: &[u8], width: usize, height: usize) -> Result<Self, ModelError> {
        let bounds = channel_percentiles(
            rgb,
            width,
            height,
            super::NAME,
            [PERCENTILE, 100.0 - PERCENTILE],
        )?;
        Ok(Self {
            low: bounds.map(|[low, _]| low as f32),
            range: bounds.map(|[low, high]| (high - low).max(EPSILON) as f32),
        })
    }

    /// Lower percentile per channel, in 0–255 units.
    pub fn low(&self) -> [f32; 3] {
        self.low
    }

    /// Divisor per channel (`max(epsilon, high - low)`), in 0–255 units.
    pub fn range(&self) -> [f32; 3] {
        self.range
    }

    pub(super) fn apply(&self, channel: usize, value: u8) -> f32 {
        (value as f32 - self.low[channel]) / self.range[channel]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(values: &[u8]) -> Vec<u8> {
        values.iter().flat_map(|&v| [v, v, v]).collect()
    }

    #[test]
    fn percentiles_interpolate_like_numpy() {
        // np.percentile(np.arange(256), [0.1, 99.9]) == [0.255, 254.745]
        let values: Vec<u8> = (0..=255).collect();
        let norm = InputNorm::percentile(&gray(&values), 16, 16).unwrap();
        assert!((norm.low()[0] - 0.255).abs() < 1e-5, "{norm:?}");
        assert!(
            (norm.range()[0] - (254.745 - 0.255)).abs() < 1e-4,
            "{norm:?}"
        );
    }

    #[test]
    fn percentiles_ignore_rare_outliers() {
        // One dark and one bright pixel in 10,000: np.percentile gives [40, 200].
        let mut values = vec![40u8; 5000];
        values.extend(vec![200u8; 4998]);
        values.push(0);
        values.push(255);
        let norm = InputNorm::percentile(&gray(&values), 100, 100).unwrap();
        assert_eq!(norm.low(), [40.0; 3]);
        assert_eq!(norm.range(), [160.0; 3]);
        assert_eq!(norm.apply(0, 40), 0.0);
        assert_eq!(norm.apply(0, 200), 1.0);
        assert!(norm.apply(0, 255) > 1.0, "values are not clipped");
    }

    #[test]
    fn channels_are_independent() {
        let rgb: Vec<u8> = (0..64u8).flat_map(|i| [i, 100, 255 - i]).collect();
        let norm = InputNorm::percentile(&rgb, 8, 8).unwrap();
        assert_eq!(norm.low()[1], 100.0);
        assert!(norm.low()[0] < 1.0 && norm.low()[2] > 191.0, "{norm:?}");
    }

    #[test]
    fn flat_channel_uses_epsilon() {
        let norm = InputNorm::percentile(&gray(&[128; 64]), 8, 8).unwrap();
        assert_eq!(norm.range(), [EPSILON as f32; 3]);
        assert_eq!(norm.apply(0, 128), 0.0);
    }

    #[test]
    fn unit_is_divide_by_255() {
        assert_eq!(InputNorm::UNIT.apply(2, 255), 1.0);
        assert_eq!(InputNorm::UNIT.apply(0, 0), 0.0);
    }

    #[test]
    fn rejects_mismatched_sizes() {
        assert!(InputNorm::percentile(&[0; 10], 2, 2).is_err());
        assert!(InputNorm::percentile(&[], 0, 0).is_err());
    }
}
