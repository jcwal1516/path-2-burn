//! OpenCLIP's one-query attention pooler, shared by CONCH v1 and TITAN.

use burn::module::Param;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[derive(Module, Debug)]
pub(crate) struct AttentionalPooler {
    query: Param<Tensor<2>>,
    attn: CrossAttention,
    ln_q: LayerNorm,
    ln_k: LayerNorm,
}

impl AttentionalPooler {
    pub(crate) fn new(embed: usize, context: usize, device: &Device) -> Self {
        let norm = |width| LayerNormConfig::new(width).with_epsilon(1e-5).init(device);
        Self {
            query: Param::from_tensor(Tensor::zeros([1, embed], device)),
            attn: CrossAttention::new(embed, context, device),
            ln_q: norm(embed),
            ln_k: norm(context),
        }
    }

    pub(crate) fn forward(&self, tokens: Tensor<3>) -> Tensor<3> {
        self.attn.forward(
            self.ln_q.forward(self.query.val()),
            self.ln_k.forward(tokens),
        )
    }
}

#[derive(Module, Debug)]
struct CrossAttention {
    // torch packs Q/K/V only when query and context widths are equal.
    in_proj_weight: Option<Param<Tensor<2>>>,
    q_proj_weight: Option<Param<Tensor<2>>>,
    k_proj_weight: Option<Param<Tensor<2>>>,
    v_proj_weight: Option<Param<Tensor<2>>>,
    in_proj_bias: Param<Tensor<1>>,
    out_proj: Linear,
}

impl CrossAttention {
    fn new(embed: usize, context: usize, device: &Device) -> Self {
        let weight = |rows, cols| Param::from_tensor(Tensor::zeros([rows, cols], device));
        Self {
            in_proj_weight: (embed == context).then(|| weight(3 * embed, embed)),
            q_proj_weight: (embed != context).then(|| weight(embed, embed)),
            k_proj_weight: (embed != context).then(|| weight(embed, context)),
            v_proj_weight: (embed != context).then(|| weight(embed, context)),
            in_proj_bias: Param::from_tensor(Tensor::zeros([3 * embed], device)),
            out_proj: LinearConfig::new(embed, embed).init(device),
        }
    }

    fn forward(&self, query: Tensor<2>, tokens: Tensor<3>) -> Tensor<3> {
        let [batch, count, context] = tokens.dims();
        let [_, embed] = query.dims();
        let head = embed / 8;
        let [wq, wk, wv] = if let Some(weight) = &self.in_proj_weight {
            [0, 1, 2].map(|i| weight.val().slice([i * embed..(i + 1) * embed, 0..embed]))
        } else {
            [
                &self.q_proj_weight,
                &self.k_proj_weight,
                &self.v_proj_weight,
            ]
            .map(|weight| weight.as_ref().expect("separate Q/K/V weights").val())
        };
        let project = |x: Tensor<2>, weight: Tensor<2>, i: usize| {
            x.matmul(weight.transpose())
                + self
                    .in_proj_bias
                    .val()
                    .slice([i * embed..(i + 1) * embed])
                    .reshape([1, embed])
        };
        let q = project(query, wq, 0)
            .reshape([1, 8, 1, head])
            .expand([batch, 8, 1, head]);
        let tokens = tokens.reshape([batch * count, context]);
        let k = project(tokens.clone(), wk, 1)
            .reshape([batch, count, 8, head])
            .swap_dims(1, 2);
        let v = project(tokens, wv, 2)
            .reshape([batch, count, 8, head])
            .swap_dims(1, 2);
        let scores = (q * (head as f64).powf(-0.5)).matmul(k.swap_dims(2, 3));
        let pooled = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, 1, embed]);
        self.out_proj.forward(pooled)
    }
}

/// Parameter suffixes of an equal-width pooler, used to name an omitted
/// captioning pooler exactly rather than ignoring a whole checkpoint prefix.
pub(crate) const FUSED_POOL_KEYS: [&str; 9] = [
    "query",
    "ln_k.weight",
    "ln_k.bias",
    "ln_q.weight",
    "ln_q.bias",
    "attn.in_proj_weight",
    "attn.in_proj_bias",
    "attn.out_proj.weight",
    "attn.out_proj.bias",
];

/// Exact keys of the twelve-block OpenCLIP text towers in CONCH v1 and TITAN.
pub(crate) fn text_keys(prefix: &str) -> Vec<String> {
    let mut keys: Vec<String> = [
        "cls_emb",
        "ln_final.bias",
        "ln_final.weight",
        "positional_embedding",
        "text_projection",
        "token_embedding.weight",
    ]
    .into_iter()
    .map(|key| format!("{prefix}.{key}"))
    .collect();
    for block in 0..12 {
        for suffix in [
            "attn.in_proj_bias",
            "attn.in_proj_weight",
            "attn.out_proj.bias",
            "attn.out_proj.weight",
            "ln_1.bias",
            "ln_1.weight",
            "ln_2.bias",
            "ln_2.weight",
            "mlp.c_fc.bias",
            "mlp.c_fc.weight",
            "mlp.c_proj.bias",
            "mlp.c_proj.weight",
        ] {
            keys.push(format!("{prefix}.transformer.resblocks.{block}.{suffix}"));
        }
    }
    keys
}
