//! The Vision Transformer in Burn, as timm's `VisionTransformer` builds it,
//! covering the DINOv2-style variants most pathology tile encoders use: class
//! and register tokens, LayerScale, GELU or SwiGLU MLPs, and position
//! embeddings with or without rows for the prefix tokens.
//!
//! Parameter names follow timm (`patch_embed.proj`, `blocks.N.attn.qkv`,
//! `blocks.N.ls1.gamma`, `norm`), so timm checkpoints load by name. Hugging Face
//! `ViTModel` and `Dinov2Model` checkpoints load with the renames in [`naming`];
//! they keep query, key and value as separate layers, which
//! [`VitConfig::fused_qkv`] selects.
//!
//! [`VisionTransformer::forward`] returns every token after the final norm:
//! the class token, the register tokens, then the patch tokens in row-major
//! order. An input whose patch grid differs from the one the position embedding
//! was trained for resamples it by the config's [`PosResample`].

mod attention;
mod block;
mod mlp;
pub mod naming;
mod resample;

use block::Block;
use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;

pub use mlp::MlpKind;

/// How the position embedding is resized for another patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosResample {
    /// timm's `resample_abs_pos_embed`: bicubic to the target size, antialiased.
    Antialiased,
    /// transformers' DINOv2 and ViT `interpolate_pos_encoding`: bicubic to the
    /// target size, not antialiased.
    Bicubic,
    /// DINO's and HIPT's `interpolate_pos_encoding`: bicubic by the scale factor
    /// `(target + 0.1) / trained`, not antialiased.
    Dino,
}

/// One Vision Transformer's architecture. Names follow timm's
/// `VisionTransformer` arguments where one exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VitConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    /// Input side the position embedding was trained for.
    pub img_size: usize,
    /// Output width of the MLP's first layer. A SwiGLU MLP splits it into gate
    /// and value halves, so its second layer takes half of it.
    pub mlp_hidden: usize,
    pub mlp: MlpKind,
    pub class_token: bool,
    pub reg_tokens: usize,
    /// timm's `no_embed_class`: the position embedding covers only the patch
    /// grid. Otherwise it also has rows for the class and register tokens.
    pub no_embed_class: bool,
    /// LayerScale after attention and after the MLP (timm `init_values`).
    pub layer_scale: bool,
    /// One `qkv` projection (timm), or separate `query`, `key` and `value`
    /// projections (transformers).
    pub fused_qkv: bool,
    pub norm_eps: f64,
    pub pos_resample: PosResample,
}

impl VitConfig {
    /// timm `vit_base_patch16_224`.
    pub const BASE_PATCH16_224: Self = Self {
        embed_dim: 768,
        depth: 12,
        num_heads: 12,
        patch_size: 16,
        img_size: 224,
        mlp_hidden: 3072,
        mlp: MlpKind::Gelu,
        class_token: true,
        reg_tokens: 0,
        no_embed_class: false,
        layer_scale: false,
        fused_qkv: true,
        norm_eps: 1e-6,
        pos_resample: PosResample::Antialiased,
    };

    /// timm `vit_large_patch16_224`.
    pub const LARGE_PATCH16_224: Self = Self {
        embed_dim: 1024,
        depth: 24,
        num_heads: 16,
        mlp_hidden: 4096,
        ..Self::BASE_PATCH16_224
    };

    /// timm `vit_huge_patch14_224`.
    pub const HUGE_PATCH14_224: Self = Self {
        embed_dim: 1280,
        depth: 32,
        num_heads: 16,
        patch_size: 14,
        mlp_hidden: 5120,
        ..Self::BASE_PATCH16_224
    };

    /// timm `vit_giant_patch14_dinov2`: DINOv2's ViT-g/14 at 518 px, with
    /// LayerScale and a packed SwiGLU MLP (4096 after DINOv2's 2/3 reduction).
    pub const GIANT_PATCH14_DINOV2: Self = Self {
        embed_dim: 1536,
        depth: 40,
        num_heads: 24,
        patch_size: 14,
        img_size: 518,
        mlp_hidden: 8192,
        mlp: MlpKind::SwiGlu,
        layer_scale: true,
        ..Self::BASE_PATCH16_224
    };

    /// timm `vit_giant_patch14_reg4_dinov2`: four register tokens, and a
    /// position embedding for the patch grid only.
    pub const GIANT_PATCH14_REG4_DINOV2: Self = Self {
        reg_tokens: 4,
        no_embed_class: true,
        ..Self::GIANT_PATCH14_DINOV2
    };

    /// Patches per side at `img_size`.
    pub fn grid(&self) -> usize {
        self.img_size / self.patch_size
    }

    /// Class and register tokens ahead of the patch tokens.
    pub fn prefix_tokens(&self) -> usize {
        usize::from(self.class_token) + self.reg_tokens
    }

    /// A transformer with placeholder parameters, to be filled from a checkpoint.
    pub fn init(&self, device: &Device) -> VisionTransformer {
        let (embed, patch, grid) = (self.embed_dim, self.patch_size, self.grid());
        let pos_prefix = if self.no_embed_class {
            0
        } else {
            self.prefix_tokens()
        };
        let token = |count: usize| Param::from_tensor(Tensor::zeros([1, count, embed], device));
        VisionTransformer {
            patch_embed: PatchEmbed {
                proj: Conv2dConfig::new([3, embed], [patch, patch])
                    .with_stride([patch, patch])
                    .init(device),
            },
            cls_token: self.class_token.then(|| token(1)),
            reg_token: (self.reg_tokens > 0).then(|| token(self.reg_tokens)),
            pos_embed: token(pos_prefix + grid * grid),
            blocks: (0..self.depth).map(|_| Block::new(self, device)).collect(),
            norm: LayerNormConfig::new(embed)
                .with_epsilon(self.norm_eps)
                .init(device),
            pos_prefix,
            grid,
            patch_size: patch,
            pos_resample: self.pos_resample,
        }
    }
}

#[derive(Module, Debug)]
pub struct VisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Option<Param<Tensor<3>>>,
    reg_token: Option<Param<Tensor<3>>>,
    pos_embed: Param<Tensor<3>>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    /// Rows of `pos_embed` ahead of the patch grid: the prefix tokens, or 0.
    pos_prefix: usize,
    /// Patches per side the position embedding covers.
    grid: usize,
    patch_size: usize,
    #[module(skip)]
    pos_resample: PosResample,
}

impl VisionTransformer {
    /// Every token after the final norm, `[batch, prefix + patches, embed]`,
    /// for a normalized `[batch, 3, H, W]` input. Like timm, a side that is not
    /// a multiple of the patch size drops its remainder.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<3> {
        self.forward_with_intermediates(x, &[]).0
    }

    /// The final tokens plus the tokens after each block in `taps` (0-based,
    /// before the final norm), in block order.
    ///
    /// Panics if a tap is not a block index.
    pub fn forward_with_intermediates(
        &self,
        x: Tensor<4>,
        taps: &[usize],
    ) -> (Tensor<3>, Vec<Tensor<3>>) {
        assert!(
            taps.iter().all(|&tap| tap < self.blocks.len()),
            "taps {taps:?} must be below the depth {}",
            self.blocks.len()
        );
        let mut x = self.embed(x);
        let mut intermediates = Vec::with_capacity(taps.len());
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x);
            if taps.contains(&i) {
                intermediates.push(x.clone());
            }
        }
        (self.norm.forward(x), intermediates)
    }

    /// Class and register tokens ahead of the patch tokens in the output.
    pub fn prefix_tokens(&self) -> usize {
        let count = |token: &Option<Param<Tensor<3>>>| token.as_ref().map_or(0, |t| t.dims()[1]);
        count(&self.cls_token) + count(&self.reg_token)
    }

    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn has_class_token(&self) -> bool {
        self.cls_token.is_some()
    }

    /// The sequence entering the first block: class and register tokens, then
    /// the patch tokens, with the position embedding added.
    fn embed(&self, x: Tensor<4>) -> Tensor<3> {
        let x = self.patch_embed.proj.forward(x);
        let [batch, embed, grid_h, grid_w] = x.dims();
        let patches = x.reshape([batch, embed, grid_h * grid_w]).swap_dims(1, 2);
        let position = self.position_embedding(grid_h, grid_w);
        let mut tokens: Vec<Tensor<3>> = [&self.cls_token, &self.reg_token]
            .into_iter()
            .flatten()
            .map(|token| {
                let count = token.dims()[1];
                token.val().expand([batch, count, embed])
            })
            .collect();
        if self.pos_prefix == 0 {
            // timm `no_embed_class`: the embedding covers only the patches, and
            // the prefix tokens are prepended after it is added.
            tokens.push(patches + position);
            Tensor::cat(tokens, 1)
        } else {
            tokens.push(patches);
            Tensor::cat(tokens, 1) + position
        }
    }

    /// `pos_embed` for a `grid_h x grid_w` patch grid, `[1, rows, embed]`.
    fn position_embedding(&self, grid_h: usize, grid_w: usize) -> Tensor<3> {
        let table = self.pos_embed.val();
        if grid_h == self.grid && grid_w == self.grid {
            return table;
        }
        let [_, rows, embed] = table.dims();
        let grid = table
            .clone()
            .slice([0..1, self.pos_prefix..rows, 0..embed])
            .reshape([self.grid, self.grid, embed]);
        let resampled = resample::resize(grid, grid_h, grid_w, self.pos_resample).reshape([
            1,
            grid_h * grid_w,
            embed,
        ]);
        if self.pos_prefix == 0 {
            return resampled;
        }
        let prefix = table.slice([0..1, 0..self.pos_prefix, 0..embed]);
        Tensor::cat(vec![prefix, resampled], 1)
    }
}

#[derive(Module, Debug)]
struct PatchEmbed {
    proj: Conv2d,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    /// A two-block transformer small enough for the CPU backend.
    fn tiny() -> VitConfig {
        VitConfig {
            embed_dim: 32,
            depth: 2,
            num_heads: 4,
            patch_size: 8,
            img_size: 32,
            mlp_hidden: 64,
            ..VitConfig::BASE_PATCH16_224
        }
    }

    #[test]
    fn presets_match_timm() {
        let giant = VitConfig::GIANT_PATCH14_REG4_DINOV2;
        assert_eq!(
            (
                giant.embed_dim,
                giant.depth,
                giant.num_heads,
                giant.mlp_hidden
            ),
            (1536, 40, 24, 8192)
        );
        assert_eq!((giant.grid(), giant.prefix_tokens()), (37, 5));
        let large = VitConfig::LARGE_PATCH16_224;
        assert_eq!((large.grid(), large.prefix_tokens()), (14, 1));
    }

    #[test]
    fn token_layout_for_each_position_embedding_style() {
        let device = Device::flex();
        let x = Tensor::<4>::random([2, 3, 32, 32], Distribution::Default, &device);
        for (reg_tokens, no_embed_class, rows) in [(0, false, 17), (2, false, 19), (2, true, 16)] {
            let config = VitConfig {
                reg_tokens,
                no_embed_class,
                ..tiny()
            };
            let model = config.init(&device);
            assert_eq!(model.pos_embed.dims(), [1, rows, 32]);
            assert_eq!(model.prefix_tokens(), 1 + reg_tokens);
            let tokens = model.forward(x.clone());
            assert_eq!(tokens.dims(), [2, 1 + reg_tokens + 16, 32]);
        }
    }

    #[test]
    fn variants_build_and_run() {
        let device = Device::flex();
        let x = Tensor::<4>::random([1, 3, 32, 32], Distribution::Default, &device);
        let config = VitConfig {
            mlp: MlpKind::SwiGlu,
            layer_scale: true,
            fused_qkv: false,
            class_token: false,
            ..tiny()
        };
        let model = config.init(&device);
        assert!(!model.has_class_token());
        let (tokens, taps) = model.forward_with_intermediates(x, &[0, 1]);
        assert_eq!(tokens.dims(), [1, 16, 32]);
        assert!(taps.iter().all(|t| t.dims() == [1, 16, 32]));
    }

    #[test]
    fn other_grids_resample_the_position_embedding() {
        let device = Device::flex();
        let model = tiny().init(&device);
        let x = Tensor::<4>::random([1, 3, 48, 40], Distribution::Default, &device);
        assert_eq!(model.forward(x).dims(), [1, 1 + 6 * 5, 32]);
    }
}
