//! InstanSeg brightfield nuclei: the UNet in Burn and the scripted model's
//! postprocessing ported to Rust.
//!
//! Input contract: RGB u8 at 0.5 µm/px, mapped per channel by an [`InputNorm`].
//! instanseg normalizes with percentile statistics of the whole image
//! ([`InputNorm::percentile`]); the reference fixtures use [`InputNorm::UNIT`].
//! Tiles up to 1024x1024 are reflect-padded to the network's fixed 1024x1024
//! input and the output is cropped back, so tiles smaller than 1024 can differ
//! from the scripted model (which pads only to a multiple of 16) within the
//! network's receptive field of the padded edge.

mod network;
mod normalize;
mod params;
mod postprocess;

pub use normalize::InputNorm;
pub use params::PostprocessParams;

use crate::runtime::{Checkpoint, load_pytorch_weights, read_f32, tensor_from_f32};
use crate::store::{ModelError, ModelStore};
use burn::prelude::*;
use network::Network;

const NAME: &str = "instanseg_net";
const CHANNELS: usize = 5;

/// Instance labels for one tile, row-major. `0` is background; labels are not
/// consecutive (each instance keeps the smallest seed index it merged).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelMap {
    pub width: usize,
    pub height: usize,
    pub labels: Vec<u32>,
}

impl LabelMap {
    pub fn instance_count(&self) -> usize {
        let mut ids: Vec<u32> = self.labels.iter().copied().filter(|&l| l != 0).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }
}

pub struct InstanSeg {
    net: Network,
    device: Device,
    params: PostprocessParams,
}

impl InstanSeg {
    /// Fixed network input side.
    pub const TILE: usize = 1024;
    pub const NAME: &'static str = NAME;

    pub fn load(store: &ModelStore, device: &Device) -> Result<Self, ModelError> {
        let params = PostprocessParams::load(store)?;
        // `export/export_instanseg.py` writes the scripted model's UNet weights.
        let checkpoint = Checkpoint {
            state_dict_key: None,
            renames: &[],
            unused: &[],
        };
        let net = load_pytorch_weights(Network::new(device), store, NAME, &checkpoint)?;
        Ok(Self {
            net,
            device: device.clone(),
            params,
        })
    }

    pub fn params(&self) -> &PostprocessParams {
        &self.params
    }

    /// Microns per pixel the model expects its input at (0.5).
    pub fn pixel_size_um(&self) -> f64 {
        self.params.pixel_size_um
    }

    /// Segment an HWC RGB tile (`64..=1024` px per side at 0.5 µm/px).
    pub fn segment(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        norm: &InputNorm,
    ) -> Result<LabelMap, ModelError> {
        let raw = self.predict(rgb, width, height, norm)?;
        decode(&raw, width, height, &self.params)
    }

    /// Network output `[5, height, width]` for an HWC RGB tile.
    pub fn predict(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        norm: &InputNorm,
    ) -> Result<Vec<f32>, ModelError> {
        let input = network_input(rgb, width, height, norm)?;
        let full = self.forward_raw(input)?;
        Ok(crop_planes(&full, Self::TILE, width, height))
    }

    /// Normalized `[1, 3, 1024, 1024]` input to raw `[1, 5, 1024, 1024]` output.
    pub fn forward_raw(&self, input: Vec<f32>) -> Result<Vec<f32>, ModelError> {
        let n = Self::TILE;
        let input = tensor_from_f32(input, [1, 3, n, n], &self.device, NAME)?;
        read_f32(self.net.forward(input), NAME)
    }
}

/// Decode a `[5, height, width]` network output with the model's postprocessing.
pub fn decode(
    raw: &[f32],
    width: usize,
    height: usize,
    params: &PostprocessParams,
) -> Result<LabelMap, ModelError> {
    let labels = postprocess::decode(raw, width, height, params).map_err(invalid_input)?;
    Ok(LabelMap {
        width,
        height,
        labels,
    })
}

/// Seed points (row, col) the decoder starts from, for diagnostics.
pub fn find_seeds(
    raw: &[f32],
    width: usize,
    height: usize,
    params: &PostprocessParams,
) -> Result<Vec<(usize, usize)>, ModelError> {
    postprocess::check_len(raw, width, height).map_err(invalid_input)?;
    Ok(postprocess::find_seeds(raw, width, height, params))
}

fn invalid_input(message: String) -> ModelError {
    ModelError::InvalidInput {
        model: NAME,
        message,
    }
}

/// HWC RGB u8 to normalized `[1, 3, 1024, 1024]`, reflect-padding bottom/right.
fn network_input(
    rgb: &[u8],
    width: usize,
    height: usize,
    norm: &InputNorm,
) -> Result<Vec<f32>, ModelError> {
    let n = InstanSeg::TILE;
    if !(64..=n).contains(&width) || !(64..=n).contains(&height) {
        return Err(invalid_input(format!(
            "tile must be 64..={n} px per side, got {width}x{height}"
        )));
    }
    if rgb.len() != width * height * 3 {
        return Err(invalid_input(format!(
            "expected {width}x{height} RGB ({} bytes), got {}",
            width * height * 3,
            rgb.len()
        )));
    }
    let plane = n * n;
    let mut out = vec![0f32; 3 * plane];
    for y in 0..n {
        let sy = reflect(y, height);
        for x in 0..n {
            let src = (sy * width + reflect(x, width)) * 3;
            for c in 0..3 {
                out[c * plane + y * n + x] = norm.apply(c, rgb[src + c]);
            }
        }
    }
    Ok(out)
}

/// `torch` "reflect" padding index (edge not repeated), repeated as needed.
fn reflect(i: usize, len: usize) -> usize {
    if i < len || len == 1 {
        return i.min(len - 1);
    }
    let period = 2 * (len - 1);
    let m = i % period;
    if m < len { m } else { period - m }
}

fn crop_planes(full: &[f32], side: usize, width: usize, height: usize) -> Vec<f32> {
    if width == side && height == side {
        return full[..CHANNELS * side * side].to_vec();
    }
    let mut out = Vec::with_capacity(CHANNELS * width * height);
    for c in 0..CHANNELS {
        for y in 0..height {
            let start = c * side * side + y * side;
            out.extend_from_slice(&full[start..start + width]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_matches_torch_reflect_padding() {
        // torch reflect of [0,1,2,3] padded right by 4: 0 1 2 3 2 1 0 1
        let got: Vec<usize> = (0..8).map(|i| reflect(i, 4)).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 2, 1, 0, 1]);
    }

    #[test]
    fn network_input_rejects_bad_sizes() {
        let unit = InputNorm::UNIT;
        assert!(network_input(&[0; 3 * 32 * 32], 32, 32, &unit).is_err());
        assert!(network_input(&[0; 10], 64, 64, &unit).is_err());
    }

    #[test]
    fn network_input_applies_the_norm_to_every_padded_pixel() {
        let rgb: Vec<u8> = (0..64 * 64)
            .flat_map(|i| [(i % 256) as u8, 7, 200])
            .collect();
        let norm = InputNorm::percentile(&rgb, 64, 64).unwrap();
        let input = network_input(&rgb, 64, 64, &norm).unwrap();
        let plane = InstanSeg::TILE * InstanSeg::TILE;
        assert_eq!(input[0], norm.apply(0, 0));
        assert_eq!(input[plane + 5], norm.apply(1, 7));
        // Reflected pixel beyond the tile edge carries the same mapping.
        assert_eq!(input[64], norm.apply(0, 62));
    }

    #[test]
    fn crop_planes_takes_top_left() {
        let side = 4;
        let full: Vec<f32> = (0..CHANNELS * side * side).map(|v| v as f32).collect();
        let out = crop_planes(&full, side, 2, 1);
        assert_eq!(
            out,
            vec![0.0, 1.0, 16.0, 17.0, 32.0, 33.0, 48.0, 49.0, 64.0, 65.0]
        );
    }
}
