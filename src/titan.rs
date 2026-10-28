//! TITAN's slide encoder over CONCH v1.5 patch features and level-0 coordinates.
//!
//! Loads the unchanged published TITAN checkpoint. The text tower, captioning
//! pooler and vision-language projection are not part of the slide embedding.

mod input;
mod network;

use crate::runtime::{Checkpoint, load_pytorch_weights, read_f32, tensor_from_f32};
use crate::store::{ModelError, ModelStore};
use burn::prelude::*;
use network::TitanNetwork;

pub const EMBEDDING_DIM: usize = 768;
const NAME: &str = "titan";

/// One slide at a time, matching upstream `encode_slide_from_patch_features`.
pub struct Titan {
    model: TitanNetwork,
    device: Device,
}

impl Titan {
    pub fn load(store: &ModelStore, device: &Device) -> Result<Self, ModelError> {
        let unused = unused_keys();
        let unused: Vec<&str> = unused.iter().map(String::as_str).collect();
        let checkpoint = Checkpoint {
            state_dict_key: None,
            renames: &[
                (
                    r"^vision_encoder\.patch_embed\.0\.",
                    "vision_encoder.patch_embed.",
                ),
                (
                    r"^vision_encoder\.blocks\.modules_list\.",
                    "vision_encoder.blocks.",
                ),
            ],
            unused: &unused,
        };
        Ok(Self {
            model: load_pytorch_weights(TitanNetwork::new(device), store, NAME, &checkpoint)?,
            device: device.clone(),
        })
    }

    /// Flattened `[N, 768]` CONCH v1.5 features, corresponding `[x, y]` level-0
    /// coordinates and the level-0 patch spacing (1024 at 40×, 512 at 20×).
    ///
    /// Coordinates are offset by their per-axis minimum and floor-divided by
    /// `patch_size_lv0`. Features in the same cell are summed; all-zero cells
    /// are removed. Returns 768 values without projection or L2 normalization.
    /// Attention memory grows quadratically with the number of occupied cells.
    pub fn encode_slide_from_patch_features(
        &self,
        features: Vec<f32>,
        coords: &[[i64; 2]],
        patch_size_lv0: u32,
    ) -> Result<Vec<f32>, ModelError> {
        let (features, positions) = input::prepare(features, coords, patch_size_lv0)?;
        let count = positions.len();
        let features = tensor_from_f32(features, [1, count, EMBEDDING_DIM], &self.device, NAME)?;
        let bias = input::alibi(&positions, &self.device)?;
        read_f32(self.model.forward(features, bias), NAME)
    }
}

/// Exact checkpoint keys outside the slide-embedding path. Do not ignore whole
/// prefixes: an unexpected tensor in a new checkpoint must still fail loading.
fn unused_keys() -> Vec<String> {
    let mut keys = crate::open_clip::text_keys("text_encoder");
    keys.push("vision_encoder.proj".into());
    keys.extend(
        crate::open_clip::FUSED_POOL_KEYS.map(|key| format!("vision_encoder.attn_pool.{key}")),
    );
    keys
}
