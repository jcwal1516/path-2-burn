//! CONCH v1's unprojected image features, as used for downstream pathology tasks.

use crate::open_clip::{AttentionalPooler, FUSED_POOL_KEYS, text_keys};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;
use vit_burn::{VisionTransformer, VitConfig};

#[derive(Module, Debug)]
pub(super) struct ConchV1 {
    visual: Visual,
}

#[derive(Module, Debug)]
struct Visual {
    trunk: VisionTransformer,
    attn_pool_contrast: AttentionalPooler,
    ln_contrast: LayerNorm,
}

impl ConchV1 {
    pub(super) fn new(vit: &VitConfig, device: &Device) -> Self {
        Self {
            visual: Visual {
                trunk: vit.init(device),
                attn_pool_contrast: AttentionalPooler::new(512, 768, device),
                ln_contrast: LayerNormConfig::new(512).with_epsilon(1e-5).init(device),
            },
        }
    }
    pub(super) fn tokens(&self, input: Tensor<4>) -> Tensor<3> {
        self.visual.trunk.forward(input)
    }
    pub(super) fn pool(&self, tokens: Tensor<3>) -> Tensor<2> {
        let batch = tokens.dims()[0];
        self.visual.ln_contrast.forward(
            self.visual
                .attn_pool_contrast
                .forward(tokens)
                .reshape([batch, 512]),
        )
    }
}

pub(super) fn unused_keys() -> Vec<String> {
    let mut keys = text_keys("text");
    keys.extend(
        [
            "logit_scale",
            "visual.proj_contrast",
            "visual.ln_caption.weight",
            "visual.ln_caption.bias",
        ]
        .map(String::from),
    );
    keys.extend(FUSED_POOL_KEYS.map(|key| format!("visual.attn_pool_caption.{key}")));
    keys
}
