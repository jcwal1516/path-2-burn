//! TITAN's vision tower; checkpoint paths are preserved except Sequential indices.

use crate::open_clip::AttentionalPooler;
use burn::module::Param;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use vit_burn::{Block, VitConfig};

#[derive(Module, Debug)]
pub(super) struct TitanNetwork {
    vision_encoder: VisionTower,
}

impl TitanNetwork {
    pub(super) fn new(device: &Device) -> Self {
        Self {
            vision_encoder: VisionTower::new(device),
        }
    }

    pub(super) fn forward(&self, features: Tensor<3>, bias: Tensor<4>) -> Tensor<2> {
        self.vision_encoder.forward(features, bias)
    }
}

#[derive(Module, Debug)]
struct VisionTower {
    patch_embed: Linear,
    cls_token: Param<Tensor<3>>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    attn_pool_contrastive: AttentionalPooler,
}

impl VisionTower {
    fn new(device: &Device) -> Self {
        let config = VitConfig {
            depth: 6,
            ..VitConfig::BASE_PATCH16_224
        };
        Self {
            patch_embed: LinearConfig::new(768, 768).init(device),
            cls_token: Param::from_tensor(Tensor::zeros([1, 1, 768], device)),
            blocks: (0..6).map(|_| Block::new(&config, device)).collect(),
            norm: LayerNormConfig::new(768).with_epsilon(1e-6).init(device),
            attn_pool_contrastive: AttentionalPooler::new(768, 768, device),
        }
    }

    fn forward(&self, features: Tensor<3>, bias: Tensor<4>) -> Tensor<2> {
        let mut tokens = Tensor::cat(
            vec![
                self.cls_token.val(),
                gelu(self.patch_embed.forward(features)),
            ],
            1,
        );
        for block in &self.blocks {
            tokens = block.forward(tokens, Some(bias.clone()));
        }
        self.attn_pool_contrastive
            .forward(self.norm.forward(tokens))
            .reshape([1, 768])
    }
}
