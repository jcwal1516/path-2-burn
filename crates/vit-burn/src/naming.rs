//! Renames from other libraries' checkpoint keys to this crate's (timm's)
//! parameter names, as `(regex, replacement)` pairs applied in order, plus the
//! checkpoint tensors that have no counterpart here.
//!
//! timm checkpoints need no renames. Both transformers layouts keep query, key
//! and value separate, so they load into a [`VitConfig`](crate::VitConfig)
//! with `fused_qkv: false`.

/// The renames both transformers layouts share (embeddings, attention, final
/// norm) around each layout's own block renames, which come between them.
macro_rules! transformers_renames {
    ($($block:expr),* $(,)?) => {
        &[
            (r"^embeddings\.patch_embeddings\.projection\.", "patch_embed.proj."),
            (r"^embeddings\.cls_token$", "cls_token"),
            (r"^embeddings\.position_embeddings$", "pos_embed"),
            (
                r"^encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.",
                "blocks.${1}.attn.${2}.",
            ),
            (
                r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.",
                "blocks.${1}.attn.proj.",
            ),
            $($block,)*
            (r"^layernorm\.", "norm."),
        ]
    };
}

/// transformers `Dinov2Model` (GELU or SwiGLU MLP, always LayerScale).
pub const HF_DINOV2: &[(&str, &str)] = transformers_renames![
    (
        r"^encoder\.layer\.(\d+)\.layer_scale(1|2)\.lambda1$",
        "blocks.${1}.ls${2}.gamma",
    ),
    (
        r"^encoder\.layer\.(\d+)\.mlp\.weights_in\.",
        "blocks.${1}.mlp.fc1."
    ),
    (
        r"^encoder\.layer\.(\d+)\.mlp\.weights_out\.",
        "blocks.${1}.mlp.fc2."
    ),
    // What is left (norm1, norm2, mlp.fc1, mlp.fc2) keeps its name.
    (r"^encoder\.layer\.(\d+)\.", "blocks.${1}."),
];

/// The mask token is used only in masked-image pretraining.
pub const HF_DINOV2_UNUSED: &[&str] = &["embeddings.mask_token"];

/// transformers `ViTModel` (GELU MLP, no LayerScale).
pub const HF_VIT: &[(&str, &str)] = transformers_renames![
    (
        r"^encoder\.layer\.(\d+)\.layernorm_before\.",
        "blocks.${1}.norm1."
    ),
    (
        r"^encoder\.layer\.(\d+)\.layernorm_after\.",
        "blocks.${1}.norm2."
    ),
    (
        r"^encoder\.layer\.(\d+)\.intermediate\.dense\.",
        "blocks.${1}.mlp.fc1."
    ),
    (
        r"^encoder\.layer\.(\d+)\.output\.dense\.",
        "blocks.${1}.mlp.fc2."
    ),
];

/// `ViTModel`'s pooler (a tanh layer on the class token) is not part of the
/// token output.
pub const HF_VIT_UNUSED: &[&str] = &["pooler.dense.weight", "pooler.dense.bias"];
