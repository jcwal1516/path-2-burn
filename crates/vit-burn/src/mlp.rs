//! The transformer MLPs: GELU, or SwiGLU with the gate and value packed in one
//! layer (timm `SwiGLUPacked`, transformers `Dinov2SwiGLUFFN`, DINOv2
//! `SwiGLUFFNFused`).

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, silu};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// `fc2(gelu(fc1(x)))` with the exact (erf) GELU.
    Gelu,
    /// `fc2(silu(a) * b)` where `fc1(x)` is `a` then `b` along the channels.
    SwiGlu,
}

#[derive(Module, Debug)]
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
    swiglu: bool,
}

impl Mlp {
    pub(crate) fn new(embed: usize, hidden: usize, kind: MlpKind, device: &Device) -> Self {
        let swiglu = kind == MlpKind::SwiGlu;
        assert!(
            !swiglu || hidden % 2 == 0,
            "a SwiGLU MLP needs an even hidden width, got {hidden}"
        );
        let inner = if swiglu { hidden / 2 } else { hidden };
        Self {
            fc1: LinearConfig::new(embed, hidden).init(device),
            fc2: LinearConfig::new(inner, embed).init(device),
            swiglu,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = self.fc1.forward(x);
        let hidden = if self.swiglu {
            let [batch, tokens, width] = hidden.dims();
            let half = width / 2;
            let gate = hidden.clone().slice([0..batch, 0..tokens, 0..half]);
            silu(gate) * hidden.slice([0..batch, 0..tokens, half..width])
        } else {
            gelu(hidden)
        };
        self.fc2.forward(hidden)
    }
}
