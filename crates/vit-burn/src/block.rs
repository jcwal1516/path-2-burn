//! timm's pre-norm transformer block, with optional LayerScale.

use crate::VitConfig;
use crate::attention::Attention;
use crate::mlp::Mlp;
use burn::module::Param;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub(crate) struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: Option<LayerScale>,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: Option<LayerScale>,
}

impl Block {
    pub(crate) fn new(config: &VitConfig, device: &Device) -> Self {
        let embed = config.embed_dim;
        let norm = || {
            LayerNormConfig::new(embed)
                .with_epsilon(config.norm_eps)
                .init(device)
        };
        let scale = || config.layer_scale.then(|| LayerScale::new(embed, device));
        Self {
            norm1: norm(),
            attn: Attention::new(embed, config.num_heads, config.fused_qkv, device),
            ls1: scale(),
            norm2: norm(),
            mlp: Mlp::new(embed, config.mlp_hidden, config.mlp, device),
            ls2: scale(),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let attended = self.attn.forward(self.norm1.forward(x.clone()));
        let x = x + LayerScale::apply(&self.ls1, attended);
        let transformed = self.mlp.forward(self.norm2.forward(x.clone()));
        x + LayerScale::apply(&self.ls2, transformed)
    }
}

/// A learned per-channel scale on a residual branch.
#[derive(Module, Debug)]
pub(crate) struct LayerScale {
    gamma: Param<Tensor<1>>,
}

impl LayerScale {
    fn new(channels: usize, device: &Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([channels], device)),
        }
    }

    fn apply(scale: &Option<Self>, x: Tensor<3>) -> Tensor<3> {
        match scale {
            Some(scale) => {
                let channels = scale.gamma.dims()[0];
                x * scale.gamma.val().reshape([1, 1, channels])
            }
            None => x,
        }
    }
}
