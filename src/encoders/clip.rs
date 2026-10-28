//! PLIP's transformers CLIP image tower, with the original parameter shapes.

use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use vit_burn::{Block, VitConfig};

pub(super) const RENAMES: &[(&str, &str)] = &[
    (
        r"^(vision_model\.encoder\.layers\.\d+)\.self_attn\.q_proj\.",
        "$1.attn.query.",
    ),
    (
        r"^(vision_model\.encoder\.layers\.\d+)\.self_attn\.k_proj\.",
        "$1.attn.key.",
    ),
    (
        r"^(vision_model\.encoder\.layers\.\d+)\.self_attn\.v_proj\.",
        "$1.attn.value.",
    ),
    (
        r"^(vision_model\.encoder\.layers\.\d+)\.self_attn\.out_proj\.",
        "$1.attn.proj.",
    ),
    (
        r"^(vision_model\.encoder\.layers\.\d+)\.layer_norm1\.",
        "$1.norm1.",
    ),
    (
        r"^(vision_model\.encoder\.layers\.\d+)\.layer_norm2\.",
        "$1.norm2.",
    ),
];

#[derive(Module, Debug)]
pub(super) struct Clip {
    vision_model: VisionModel,
    visual_projection: Linear,
}

#[derive(Module, Debug)]
struct VisionModel {
    embeddings: Embeddings,
    pre_layrnorm: LayerNorm,
    encoder: Encoder,
    post_layernorm: LayerNorm,
}

#[derive(Module, Debug)]
struct Embeddings {
    class_embedding: Param<Tensor<1>>,
    patch_embedding: Conv2d,
    position_embedding: Embedding,
}

#[derive(Module, Debug)]
struct Encoder {
    layers: Vec<Block>,
}

impl Clip {
    pub(super) fn new(config: &VitConfig, device: &Device) -> Self {
        let width = config.embed_dim;
        let norm = || {
            LayerNormConfig::new(width)
                .with_epsilon(config.norm_eps)
                .init(device)
        };
        Self {
            vision_model: VisionModel {
                embeddings: Embeddings {
                    class_embedding: Param::from_tensor(Tensor::zeros([width], device)),
                    patch_embedding: Conv2dConfig::new([3, width], [config.patch_size; 2])
                        .with_stride([config.patch_size; 2])
                        .with_bias(false)
                        .init(device),
                    position_embedding: EmbeddingConfig::new(
                        1 + config.grid() * config.grid(),
                        width,
                    )
                    .init(device),
                },
                pre_layrnorm: norm(),
                post_layernorm: norm(),
                encoder: Encoder {
                    layers: (0..config.depth)
                        .map(|_| Block::new(config, device))
                        .collect(),
                },
            },
            visual_projection: LinearConfig::new(width, 512).with_bias(false).init(device),
        }
    }

    /// CLIP's last hidden state; post_layernorm is applied only when pooling.
    pub(super) fn tokens(&self, input: Tensor<4>) -> Tensor<3> {
        let embedding = &self.vision_model.embeddings;
        let patches = embedding.patch_embedding.forward(input);
        let [batch, width, h, w] = patches.dims();
        let patches = patches.reshape([batch, width, h * w]).swap_dims(1, 2);
        let class = embedding
            .class_embedding
            .val()
            .reshape([1, 1, width])
            .expand([batch, 1, width]);
        let position = embedding.position_embedding.weight.val().unsqueeze::<3>();
        let mut tokens = self
            .vision_model
            .pre_layrnorm
            .forward(Tensor::cat(vec![class, patches], 1) + position);
        for block in &self.vision_model.encoder.layers {
            tokens = block.forward(tokens, None);
        }
        tokens
    }

    pub(super) fn pool(&self, tokens: Tensor<3>) -> Tensor<2> {
        let [batch, _, width] = tokens.dims();
        let class = tokens
            .slice([0..batch, 0..1, 0..width])
            .reshape([batch, width]);
        self.visual_projection
            .forward(self.vision_model.post_layernorm.forward(class))
    }
}

pub(super) fn unused_keys() -> Vec<String> {
    // position_ids are fixed arange buffers persisted by transformers 4.x.
    let mut keys: Vec<String> = [
        "logit_scale",
        "vision_model.embeddings.position_ids",
        "text_model.embeddings.position_ids",
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.final_layer_norm.bias",
        "text_projection.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for layer in 0..12 {
        for suffix in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "layer_norm1",
            "layer_norm2",
            "mlp.fc1",
            "mlp.fc2",
        ] {
            for param in ["weight", "bias"] {
                keys.push(format!(
                    "text_model.encoder.layers.{layer}.{suffix}.{param}"
                ));
            }
        }
    }
    keys
}
