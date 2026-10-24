//! CellViT on one tile: nucleus segmentation, horizontal/vertical distance maps,
//! PanNuke nucleus types and tissue type.

use super::native::{Backbone, Network};
use crate::hover::device_maps;
use crate::preprocess::rgb_to_nchw;
use crate::runtime::{read_f32, tensor_from_f32};
use crate::store::{ModelError, ModelStore};
use burn::prelude::*;
use burn::tensor::activation::softmax;

/// PanNuke nucleus classes, index = type id. 0 is background.
pub const NUCLEI_CLASSES: [&str; 6] = [
    "background",
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
];
/// PanNuke tissue classes for the tissue head.
pub const TISSUE_CLASSES: [&str; 19] = [
    "adrenal_gland",
    "bile_duct",
    "bladder",
    "breast",
    "cervix",
    "colon",
    "esophagus",
    "head_neck",
    "kidney",
    "liver",
    "lung",
    "ovarian",
    "pancreatic",
    "prostate",
    "skin",
    "stomach",
    "testis",
    "thyroid",
    "uterus",
];

/// Raw network outputs, NCHW float32, exactly as the PyTorch model returns them.
pub struct CellVitRaw {
    pub nuclei_binary_map: Vec<f32>,
    pub hv_map: Vec<f32>,
    pub nuclei_type_map: Vec<f32>,
    pub tissue_types: Vec<f32>,
}

/// Decoder-ready maps, reduced on the device to cut readback.
pub struct CellVitPrediction {
    /// P(nucleus) per pixel, [H, W].
    pub nuclei_prob: Vec<f32>,
    /// Horizontal then vertical distance maps, [2, H, W].
    pub hv: Vec<f32>,
    /// Argmax PanNuke type per pixel, [H, W].
    pub type_pred: Vec<u8>,
    /// Softmax over [`TISSUE_CLASSES`].
    pub tissue_probs: Vec<f32>,
}

/// One CellViT checkpoint and the input it was trained on.
#[derive(Debug, Clone, Copy)]
pub struct CellVitWeights {
    /// The checkpoint's name in the weights manifest.
    pub name: &'static str,
    pub backbone: Backbone,
    /// Tile side in pixels, a multiple of 16.
    pub input_size: usize,
    /// Microns per pixel the tiles should be read at.
    pub mpp: f64,
    /// Network input for one byte of RGB channel `c`.
    pub normalize: fn(usize, u8) -> f32,
}

/// CellViT's training normalization, `(x / 255 - 0.5) / 0.5`.
fn half_range(_channel: usize, x: u8) -> f32 {
    x as f32 / 127.5 - 1.0
}

/// The official PanNuke CellViT-SAM-H (TIO-IKIM, `CellViT-SAM-H-x40-AMP.pth`):
/// 1024 px tiles at 0.25 µm/px.
pub const SAM_H_X40: CellVitWeights = CellVitWeights {
    name: "cellvit_sam_h",
    backbone: Backbone::SamH,
    input_size: 1024,
    mpp: 0.25,
    normalize: half_range,
};

/// The official PanNuke CellViT-256 (TIO-IKIM, `CellViT-256-x40.pth`), on
/// HIPT's ViT-S/16: 256 px tiles at 0.25 µm/px.
pub const VIT256_X40: CellVitWeights = CellVitWeights {
    name: "cellvit_256",
    backbone: Backbone::Vit256,
    input_size: 256,
    ..SAM_H_X40
};

/// A CellViT checkpoint on one device, fed one square tile at a time.
pub struct CellVit {
    weights: CellVitWeights,
    network: Network,
    device: Device,
}

impl CellVit {
    pub fn load(
        store: &ModelStore,
        device: &Device,
        weights: &CellVitWeights,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            weights: *weights,
            network: Network::load(store, device, weights.name, weights.backbone)?,
            device: device.clone(),
        })
    }

    pub fn weights(&self) -> &CellVitWeights {
        &self.weights
    }

    /// One HWC RGB u8 tile of `input_size` pixels per side to the normalized
    /// NCHW input.
    pub fn normalize_rgb(&self, rgb: &[u8]) -> Result<Vec<f32>, ModelError> {
        let CellVitWeights {
            name,
            input_size,
            normalize,
            ..
        } = self.weights;
        rgb_to_nchw(rgb, input_size, Some(1), name, normalize)
    }

    /// Network outputs without any reduction, for parity checks.
    pub fn forward_raw(&self, input: Vec<f32>) -> Result<CellVitRaw, ModelError> {
        let name = self.weights.name;
        let (binary, hv, types, tissue) = self.run(input)?;
        Ok(CellVitRaw {
            nuclei_binary_map: read_f32(binary, name)?,
            hv_map: read_f32(hv, name)?,
            nuclei_type_map: read_f32(types, name)?,
            tissue_types: read_f32(tissue, name)?,
        })
    }

    /// Normalized input to decoder-ready maps, reduced on the device.
    pub fn predict(&self, input: Vec<f32>) -> Result<CellVitPrediction, ModelError> {
        let name = self.weights.name;
        let (binary, hv, types, tissue) = self.run(input)?;
        let maps = device_maps(binary, hv, Some(types), name)?;
        Ok(CellVitPrediction {
            nuclei_prob: maps.nuclei_prob,
            hv: maps.hv,
            type_pred: maps.type_pred,
            tissue_probs: read_f32(softmax(tissue, 1), name)?,
        })
    }

    fn run(
        &self,
        input: Vec<f32>,
    ) -> Result<(Tensor<4>, Tensor<4>, Tensor<4>, Tensor<2>), ModelError> {
        let CellVitWeights {
            name,
            input_size: n,
            ..
        } = self.weights;
        let input = tensor_from_f32(input, [1, 3, n, n], &self.device, name)?;
        Ok(self.network.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_matches_training() {
        let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(close(half_range(0, 0), -1.0) && close(half_range(2, 255), 1.0));
    }

    #[test]
    fn entries_fit_their_backbones() {
        for weights in [SAM_H_X40, VIT256_X40] {
            assert_eq!(weights.input_size % 16, 0, "{}", weights.name);
        }
        // SAM-H's position embedding covers 1024 px; larger tiles do not fit.
        assert!(SAM_H_X40.input_size <= 1024);
    }
}
