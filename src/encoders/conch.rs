//! CONCH v1.5's image tower. Field names match the published checkpoint.

use burn::module::Param;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use vit_burn::{VisionTransformer, VitConfig};

#[derive(Module, Debug)]
pub(super) struct Conch {
    pub(super) trunk: VisionTransformer,
    attn_pool_contrast: AttentionalPooler,
    ln_contrast: LayerNorm,
}

impl Conch {
    pub(super) fn new(vit: &VitConfig, device: &Device) -> Self {
        Self {
            trunk: vit.init(device),
            attn_pool_contrast: AttentionalPooler::new(device),
            // The head uses torch LayerNorm's default, unlike the trunk's 1e-6.
            ln_contrast: LayerNormConfig::new(768).with_epsilon(1e-5).init(device),
        }
    }

    pub(super) fn pool(&self, tokens: Tensor<3>) -> Tensor<2> {
        let pooled = self.attn_pool_contrast.forward(tokens);
        let [batch, _, width] = pooled.dims();
        self.ln_contrast.forward(pooled.reshape([batch, width]))
    }
}

#[derive(Module, Debug)]
struct AttentionalPooler {
    query: Param<Tensor<2>>,
    ln_k: LayerNorm,
    ln_q: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
}

impl AttentionalPooler {
    fn new(device: &Device) -> Self {
        let norm = |width| LayerNormConfig::new(width).with_epsilon(1e-5).init(device);
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            query: Param::from_tensor(Tensor::zeros([1, 768], device)),
            ln_k: norm(1024),
            ln_q: norm(768),
            to_q: linear(768, 768),
            to_kv: linear(1024, 1536),
            to_out: linear(768, 768),
        }
    }

    /// One learned query attends to every trunk token, including the class token.
    fn forward(&self, tokens: Tensor<3>) -> Tensor<3> {
        let [batch, count, _] = tokens.dims();
        let query = self.query.val().unsqueeze::<3>().expand([batch, 1, 768]);
        let q = self
            .to_q
            .forward(self.ln_q.forward(query))
            .reshape([batch, 1, 8, 96])
            .swap_dims(1, 2);
        let kv = self
            .to_kv
            .forward(self.ln_k.forward(tokens))
            .reshape([batch, count, 2, 8, 96])
            .permute([2, 0, 3, 1, 4]);
        let [k, v] = [0, 1].map(|i| {
            kv.clone()
                .slice([i..i + 1, 0..batch, 0..8, 0..count, 0..96])
                .reshape([batch, 8, count, 96])
        });
        let scores = (q * 96_f64.powf(-0.5)).matmul(k.swap_dims(2, 3));
        let pooled = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, 1, 768]);
        self.to_out.forward(pooled)
    }
}
