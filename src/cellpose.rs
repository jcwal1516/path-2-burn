//! Cellpose-SAM (Pachitariu, Rariden & Stringer, 2025): SAM's ViT-L image
//! encoder with 8 px patches and global attention in every block, read out to
//! per-pixel flows and cell probability on 256 px tiles.
//!
//! This is the network and Cellpose's input normalization. Turning flows into
//! masks (Cellpose's flow dynamics) is not ported yet.

use crate::preprocess::channel_percentiles;
use crate::runtime::{Checkpoint, batch_from_f32, load_pytorch_weights, read_f32};
use crate::store::{ModelError, ModelStore};
use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::module::conv_transpose2d;
use burn::tensor::ops::ConvTransposeOptions;
use sam_burn::{ImageEncoder, ImageEncoderConfig};

const NAME: &str = "cellpose_sam";
/// Pixels per token side.
const PATCH: usize = 8;
/// Flows (dY, dX) and the cell-probability logit.
const OUTPUTS: usize = 3;

const CHECKPOINT: Checkpoint<'static> = Checkpoint {
    state_dict_key: None,
    renames: &[
        // Multi-GPU training saves keys under `module.`; Cellpose strips it.
        (r"^module\.", ""),
        (r"^W2$", "w2"),
    ],
    // The training images' mean cell diameters; Cellpose-SAM does not rescale.
    unused: &["diam_labels", "diam_mean"],
};

/// Cellpose-SAM's network on one device.
pub struct CellposeSam {
    network: Network,
    device: Device,
}

impl CellposeSam {
    pub const NAME: &'static str = NAME;
    /// Tile side the network was trained on.
    pub const TILE: usize = 256;
    /// Where the `cpsam` checkpoint is published.
    pub const UPSTREAM: &'static str = "https://huggingface.co/mouseland/cellpose-sam";

    pub fn load(store: &ModelStore, device: &Device) -> Result<Self, ModelError> {
        let network = load_pytorch_weights(Network::new(device), store, NAME, &CHECKPOINT)?;
        Ok(Self {
            network,
            device: device.clone(),
        })
    }

    /// Cellpose's default input normalization (`normalize_img`) of a whole HWC
    /// RGB image, as planar CHW. Per channel, the 1st percentile maps to 0 and
    /// the 99th to 1, unclipped; a channel spanning less than 1e-3 between them
    /// becomes 0, and a constant channel keeps its raw values. Above 224³
    /// pixels the percentiles come from every `side / 224`-th row and column.
    pub fn normalize_image(
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let bounds = percentile_bounds(rgb, width, height)?;
        let pixels = width * height;
        let mut out = vec![0f32; 3 * pixels];
        for (c, [low, high]) in bounds.into_iter().enumerate() {
            let plane = &mut out[c * pixels..(c + 1) * pixels];
            let values = rgb.chunks_exact(3).map(|px| px[c] as f32);
            let constant = rgb.chunks_exact(3).all(|px| px[c] == rgb[c]);
            let (low, range) = (low as f32, high as f32 - low as f32);
            if constant {
                plane.iter_mut().zip(values).for_each(|(out, v)| *out = v);
            } else if range > 1e-3 {
                plane
                    .iter_mut()
                    .zip(values)
                    .for_each(|(out, v)| *out = (v - low) / range);
            }
        }
        Ok(out)
    }

    /// For normalized `[batch, 3, 256, 256]` tiles, per pixel the flows (dY,
    /// dX) then the cell-probability logit: `[batch, 3, 256, 256]`.
    pub fn flows(&self, input: Vec<f32>) -> Result<Vec<f32>, ModelError> {
        let side = Self::TILE;
        let input = batch_from_f32(input, [3, side, side], &self.device, NAME)?;
        read_f32(self.network.forward(input), NAME)
    }
}

/// Per-channel 1st and 99th percentiles, over `normalize99(downsample=True)`'s
/// subsample for images above 224³ pixels.
fn percentile_bounds(rgb: &[u8], width: usize, height: usize) -> Result<[[f64; 2]; 3], ModelError> {
    const SUBSAMPLE_ABOVE: usize = 224 * 224 * 224;
    const Q: [f64; 2] = [1.0, 99.0];
    if width * height <= SUBSAMPLE_ABOVE || rgb.len() != width * height * 3 {
        return channel_percentiles(rgb, width, height, NAME, Q);
    }
    let (row_step, col_step) = ((height / 224).max(1), (width / 224).max(1));
    let rows = height.div_ceil(row_step);
    let cols = width.div_ceil(col_step);
    let mut sample = Vec::with_capacity(rows * cols * 3);
    for y in (0..height).step_by(row_step) {
        for x in (0..width).step_by(col_step) {
            let i = (y * width + x) * 3;
            sample.extend_from_slice(&rgb[i..i + 3]);
        }
    }
    channel_percentiles(&sample, cols, rows, NAME, Q)
}

#[derive(Module, Debug)]
struct Network {
    encoder: ImageEncoder,
    /// Each token's `OUTPUTS x PATCH x PATCH` pixel values.
    out: Conv2d,
    /// Fixed identity weights that move those values into pixel space.
    w2: Param<Tensor<4>>,
}

impl Network {
    fn new(device: &Device) -> Self {
        let config = ImageEncoderConfig::cellpose_sam();
        let per_token = OUTPUTS * PATCH * PATCH;
        Self {
            encoder: config.init(device),
            out: Conv2dConfig::new([config.out_chans, per_token], [1, 1]).init(device),
            w2: Param::from_tensor(Tensor::zeros([per_token, OUTPUTS, PATCH, PATCH], device)),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let tokens = self.out.forward(self.encoder.forward(x));
        let options = ConvTransposeOptions::new([PATCH; 2], [0; 2], [0; 2], [1; 2], 1);
        conv_transpose2d(tokens, self.w2.val(), None, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_maps_percentiles_to_zero_and_one() {
        // 100 pixels: red 0..=99, green constant, blue reversed.
        let rgb: Vec<u8> = (0..100u8).flat_map(|v| [v, 7, 99 - v]).collect();
        let x = CellposeSam::normalize_image(&rgb, 10, 10).unwrap();
        // np.percentile(np.arange(100), [1, 99]) == [0.99, 98.01]
        let red = |v: f32| (v - 0.99) / (98.01 - 0.99);
        assert!((x[0] - red(0.0)).abs() < 1e-6);
        assert!((x[99] - red(99.0)).abs() < 1e-6);
        assert!(
            x[100..200].iter().all(|&v| v == 7.0),
            "constant channel kept raw"
        );
        assert!((x[200] - red(99.0)).abs() < 1e-6);
        assert!(CellposeSam::normalize_image(&rgb[1..], 10, 10).is_err());
    }

    #[test]
    fn narrow_percentile_range_becomes_zero() {
        // One brighter pixel in 10,000: the 1st and 99th percentiles are equal.
        let mut rgb = vec![50u8; 100 * 100 * 3];
        rgb[0] = 51;
        let x = CellposeSam::normalize_image(&rgb, 100, 100).unwrap();
        assert!(x[..10_000].iter().all(|&v| v == 0.0));
    }
}
