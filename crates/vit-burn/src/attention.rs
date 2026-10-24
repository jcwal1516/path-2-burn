//! Multi-head self-attention over all tokens.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

/// Holds either timm's fused `qkv` projection or transformers' separate
/// `query`, `key` and `value` projections; the other fields are `None`.
#[derive(Module, Debug)]
pub(crate) struct Attention {
    qkv: Option<Linear>,
    query: Option<Linear>,
    key: Option<Linear>,
    value: Option<Linear>,
    proj: Linear,
    heads: usize,
}

impl Attention {
    pub(crate) fn new(embed: usize, heads: usize, fused_qkv: bool, device: &Device) -> Self {
        let linear = |outputs: usize| LinearConfig::new(embed, outputs).init(device);
        let separate = || (!fused_qkv).then(|| linear(embed));
        Self {
            qkv: fused_qkv.then(|| linear(3 * embed)),
            query: separate(),
            key: separate(),
            value: separate(),
            proj: linear(embed),
            heads,
        }
    }

    /// `[batch, tokens, embed]` to `[batch, tokens, embed]`.
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, tokens, embed] = x.dims();
        let (heads, head_dim) = (self.heads, embed / self.heads);
        let [q, k, v] = match (&self.qkv, &self.query, &self.key, &self.value) {
            (Some(qkv), None, None, None) => {
                let fused = qkv
                    .forward(x)
                    .reshape([batch, tokens, 3, heads, head_dim])
                    .permute([2, 0, 3, 1, 4]);
                [0, 1, 2].map(|i| {
                    fused
                        .clone()
                        .slice([i..i + 1, 0..batch, 0..heads, 0..tokens, 0..head_dim])
                        .reshape([batch, heads, tokens, head_dim])
                })
            }
            (None, Some(query), Some(key), Some(value)) => [query, key, value].map(|linear| {
                linear
                    .forward(x.clone())
                    .reshape([batch, tokens, heads, head_dim])
                    .swap_dims(1, 2)
            }),
            _ => unreachable!("attention has a fused or a separate q/k/v projection"),
        };
        let scores = (q * (head_dim as f64).powf(-0.5)).matmul(k.swap_dims(2, 3));
        let x = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, tokens, embed]);
        self.proj.forward(x)
    }
}
