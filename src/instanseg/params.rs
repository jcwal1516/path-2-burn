//! Postprocessing parameters and the pixel classifier, read from the verified
//! `instanseg_postprocess.json` written by `export/export_instanseg.py`.

use crate::store::{ModelError, ModelStore};
use serde::Deserialize;

pub(super) const NAME: &str = "instanseg_postprocess";

#[derive(Debug, Clone, Deserialize)]
struct LinearJson {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClassifierJson {
    fc1: LinearJson,
    fc2: LinearJson,
    fc3: LinearJson,
}

#[derive(Debug, Clone, Deserialize)]
struct ParamsJson {
    pixel_size_um: f64,
    dim_coords: usize,
    n_sigma: usize,
    dim_seeds: usize,
    mask_threshold: f32,
    peak_distance: usize,
    seed_threshold: f32,
    overlap_threshold: f32,
    window_size: usize,
    cleanup_fragments: bool,
    pixel_classifier: ClassifierJson,
}

/// Dense layer `y = W x + b` with `W` stored row-major as `[out, in]`.
#[derive(Debug, Clone)]
pub(super) struct Dense<const IN: usize, const OUT: usize> {
    weight: [[f32; IN]; OUT],
    bias: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> Dense<IN, OUT> {
    fn from_json(layer: &LinearJson, name: &str) -> Result<Self, String> {
        if layer.weight.len() != OUT
            || layer.bias.len() != OUT
            || layer.weight.iter().any(|row| row.len() != IN)
        {
            return Err(format!("{name} must be {OUT}x{IN} with {OUT} biases"));
        }
        let mut weight = [[0f32; IN]; OUT];
        for (dst, src) in weight.iter_mut().zip(&layer.weight) {
            dst.copy_from_slice(src);
        }
        let mut bias = [0f32; OUT];
        bias.copy_from_slice(&layer.bias);
        Ok(Self { weight, bias })
    }

    #[inline]
    fn apply(&self, x: &[f32; IN]) -> [f32; OUT] {
        let mut out = self.bias;
        for (o, row) in out.iter_mut().zip(&self.weight) {
            for (w, v) in row.iter().zip(x) {
                *o += w * v;
            }
        }
        out
    }
}

/// InstanSeg's `ProbabilityNet`: 4 -> 5 -> 5 -> 1 with ReLU, returning the logit.
#[derive(Debug, Clone)]
pub(super) struct PixelClassifier {
    fc1: Dense<4, 5>,
    fc2: Dense<5, 5>,
    fc3: Dense<5, 1>,
}

impl PixelClassifier {
    #[inline]
    pub(super) fn logit(&self, features: &[f32; 4]) -> f32 {
        let relu = |v: [f32; 5]| v.map(|x| x.max(0.0));
        let h1 = relu(self.fc1.apply(features));
        let h2 = relu(self.fc2.apply(&h1));
        self.fc3.apply(&h2)[0]
    }
}

/// Parameters of the scripted `brightfield_nuclei` forward, at their exported defaults.
#[derive(Debug, Clone)]
pub struct PostprocessParams {
    /// Microns per pixel the model expects.
    pub pixel_size_um: f64,
    pub mask_threshold: f32,
    pub peak_distance: usize,
    pub seed_threshold: f32,
    pub overlap_threshold: f32,
    pub window_size: usize,
    pub cleanup_fragments: bool,
    pub(super) classifier: PixelClassifier,
}

impl PostprocessParams {
    pub fn load(store: &ModelStore) -> Result<Self, ModelError> {
        let path = store.weights(NAME)?;
        let raw = std::fs::read_to_string(&path).map_err(ModelError::io(&path))?;
        Self::from_json(&raw).map_err(|message| ModelError::Load {
            model: NAME.to_string(),
            message,
        })
    }

    pub(super) fn from_json(raw: &str) -> Result<Self, String> {
        let p: ParamsJson = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        // The Rust port implements the single-head brightfield layout only:
        // channels = [field_x, field_y, sigma_0, sigma_1, seed].
        if p.dim_coords != 2 || p.n_sigma != 2 || p.dim_seeds != 1 {
            return Err(format!(
                "unsupported head layout dim_coords={} n_sigma={} dim_seeds={} (expected 2/2/1)",
                p.dim_coords, p.n_sigma, p.dim_seeds
            ));
        }
        if p.window_size == 0 || p.peak_distance == 0 {
            return Err("window_size and peak_distance must be positive".into());
        }
        let c = &p.pixel_classifier;
        let classifier = PixelClassifier {
            fc1: Dense::from_json(&c.fc1, "fc1")?,
            fc2: Dense::from_json(&c.fc2, "fc2")?,
            fc3: Dense::from_json(&c.fc3, "fc3")?,
        };
        Ok(Self {
            pixel_size_um: p.pixel_size_um,
            mask_threshold: p.mask_threshold,
            peak_distance: p.peak_distance,
            seed_threshold: p.seed_threshold,
            overlap_threshold: p.overlap_threshold,
            window_size: p.window_size,
            cleanup_fragments: p.cleanup_fragments,
            classifier,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(fc1_rows: usize) -> String {
        let row = |n: usize| format!("[{}]", vec!["0.5"; n].join(","));
        let layer = |out: usize, inp: usize| {
            format!(
                r#"{{"weight":[{}],"bias":[{}]}}"#,
                vec![row(inp); out].join(","),
                vec!["0.1"; out].join(",")
            )
        };
        format!(
            r#"{{"pixel_size_um":0.5,"dim_coords":2,"n_sigma":2,"dim_seeds":1,"mask_threshold":0.53,"peak_distance":5,
                "seed_threshold":0.7,"overlap_threshold":0.3,"window_size":32,"cleanup_fragments":true,
                "pixel_classifier":{{"fc1":{},"fc2":{},"fc3":{}}}}}"#,
            layer(fc1_rows, 4),
            layer(5, 5),
            layer(1, 5)
        )
    }

    #[test]
    fn parses_and_evaluates_classifier() {
        let p = PostprocessParams::from_json(&json(5)).unwrap();
        // fc1: 0.5*4 + 0.1 = 2.1; fc2: 0.5*2.1*5 + 0.1 = 5.35; fc3: 0.5*5.35*5 + 0.1 = 13.475
        assert!((p.classifier.logit(&[1.0; 4]) - 13.475).abs() < 1e-4);
    }

    #[test]
    fn rejects_wrong_layer_shapes() {
        assert!(
            PostprocessParams::from_json(&json(4))
                .unwrap_err()
                .contains("fc1")
        );
    }
}
