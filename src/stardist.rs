//! StarDist2D's published QuPath U-Nets and star-convex polygon suppression.
//!
//! `export/reference_stardist.py export` extracts convolution tensors from the
//! original frozen graphs. Predictions are on the native grid, before QuPath's
//! fixed output upsampling. Distances remain in input-image pixels.

mod network;
mod nms;

use crate::{
    Checkpoint, Device, ModelError, ModelStore, load_pytorch_weights, read_f32, tensor_from_f32,
};
use network::StarDistNet;
pub use nms::{Instance, NmsOptions};

pub const RAYS: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct StarDistModel {
    pub name: &'static str,
    pub channels: usize,
    pub grid: usize,
}

pub const HE: StarDistModel = StarDistModel {
    name: "stardist_he",
    channels: 3,
    grid: 2,
};
pub const FLUORESCENCE: StarDistModel = StarDistModel {
    name: "stardist_fluorescence",
    channels: 1,
    grid: 2,
};
pub const FLUORESCENCE_PAPER: StarDistModel = StarDistModel {
    name: "stardist_fluorescence_paper",
    channels: 1,
    grid: 1,
};

impl StarDistModel {
    /// Per-channel CSBDeep normalization of one CHW image: 1st / 99.8th
    /// percentiles, linear interpolation, epsilon 1e-20, no clipping.
    pub fn normalize(
        &self,
        image: &mut [f32],
        height: usize,
        width: usize,
    ) -> Result<(), ModelError> {
        self.validate(image, height, width)?;
        for channel in image.chunks_exact_mut(height * width) {
            let mut sorted = channel.to_vec();
            sorted.sort_unstable_by(f32::total_cmp);
            let percentile = |q: f64| {
                let pos = q * (sorted.len() - 1) as f64;
                let i = pos.floor() as usize;
                let a = f64::from(sorted[i]);
                (a + (pos - i as f64) * (f64::from(sorted[(i + 1).min(sorted.len() - 1)]) - a))
                    as f32
            };
            let low = percentile(0.01);
            let range = percentile(0.998) - low + 1e-20;
            for v in channel {
                *v = (*v - low) / range;
            }
        }
        Ok(())
    }

    fn validate(&self, image: &[f32], height: usize, width: usize) -> Result<(), ModelError> {
        if height == 0
            || width == 0
            || ![1, 3].contains(&self.channels)
            || ![1, 2].contains(&self.grid)
            || height
                .checked_mul(width)
                .and_then(|n| n.checked_mul(self.channels))
                != Some(image.len())
            || image.iter().any(|v| !v.is_finite())
        {
            return Err(ModelError::InvalidInput { model: self.name,
                message: "expected a nonempty finite CHW image matching the model's channels and dimensions".into() });
        }
        Ok(())
    }
}

pub struct StarDist {
    spec: StarDistModel,
    model: StarDistNet,
    device: Device,
}

impl StarDist {
    pub fn load(
        store: &ModelStore,
        device: &Device,
        spec: &StarDistModel,
    ) -> Result<Self, ModelError> {
        if ![1, 3].contains(&spec.channels) || ![1, 2].contains(&spec.grid) {
            return Err(ModelError::InvalidInput {
                model: spec.name,
                message: "unsupported channels or grid".into(),
            });
        }
        let checkpoint = Checkpoint {
            state_dict_key: None,
            renames: &[
                (r"^conv2d_1\.", "stem.first."),
                (r"^conv2d_2\.", "stem.second."),
                (r"^down_level_(\d+)_no_0\.", "down.$1.first."),
                (r"^down_level_(\d+)_no_1\.", "down.$1.second."),
                (r"^middle_0\.", "middle.first."),
                (r"^middle_2\.", "middle.second."),
                (r"^up_level_(\d+)_no_0\.", "up.$1.first."),
                (r"^up_level_(\d+)_no_2\.", "up.$1.second."),
            ],
            unused: &[],
        };
        Ok(Self {
            spec: *spec,
            model: load_pytorch_weights(
                StarDistNet::new(spec.channels, spec.grid, device),
                store,
                spec.name,
                &checkpoint,
            )?,
            device: device.clone(),
        })
    }

    /// One normalized CHW image. Each side must be divisible by `8 * grid`;
    /// callers pad or tile their images before inference.
    pub fn predict(
        &self,
        input: Vec<f32>,
        height: usize,
        width: usize,
    ) -> Result<StarDistRaw, ModelError> {
        self.spec.validate(&input, height, width)?;
        let divisor = 8 * self.spec.grid;
        if height % divisor != 0 || width % divisor != 0 {
            return Err(ModelError::InvalidInput {
                model: self.spec.name,
                message: format!("input sides must be divisible by {divisor}"),
            });
        }
        let input = tensor_from_f32(
            input,
            [1, self.spec.channels, height, width],
            &self.device,
            self.spec.name,
        )?;
        let (probability, distances) = self.model.forward(input);
        Ok(StarDistRaw {
            height: height / self.spec.grid,
            width: width / self.spec.grid,
            grid: self.spec.grid,
            probability: read_f32(probability, self.spec.name)?,
            distances: read_f32(distances, self.spec.name)?,
        })
    }
}

/// Native-grid predictions: probability `[H, W]` and ray distances `[32, H, W]`.
#[derive(Debug, Clone)]
pub struct StarDistRaw {
    pub height: usize,
    pub width: usize,
    pub grid: usize,
    pub probability: Vec<f32>,
    pub distances: Vec<f32>,
}

impl StarDistRaw {
    /// Select star-convex polygons, with centers and vertices in input pixels.
    pub fn instances(&self, options: &NmsOptions) -> Result<Vec<Instance>, ModelError> {
        nms::suppress(self, options)
    }
}
