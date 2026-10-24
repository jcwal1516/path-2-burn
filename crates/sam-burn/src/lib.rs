//! The Segment Anything (SAM) ViT image encoder in Burn.
//!
//! Follows `ImageEncoderViT` from Meta's `segment_anything`: a 16x16 patch
//! embedding with an absolute position embedding, transformer blocks that
//! attend within 14x14 windows except for a few global ones, decomposed
//! relative positions in every block, and a convolutional neck. Parameter names
//! match the PyTorch module, so checkpoints load by name: a model that keeps the
//! encoder in a field called `image_encoder` (SAM) or `encoder` (CellViT) reads
//! the original state dict without renaming.
//!
//! Tiles smaller than `img_size` use the top-left corner of the position
//! embedding and resize the global-attention relative-position tables, as
//! CellViT does for 256 px inputs.

mod attention;
mod block;

use block::Block;
use burn::module::Param;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;

/// SAM's `ImageEncoderViT` hyperparameters, named as in `segment_anything`.
#[derive(Config, Debug)]
pub struct ImageEncoderConfig {
    /// Token width: 768 (ViT-B), 1024 (ViT-L) or 1280 (ViT-H).
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    /// Blocks (0-based) with global attention; the others attend within windows.
    pub global_attn_indexes: Vec<usize>,
    /// Input side the position embedding covers.
    #[config(default = 1024)]
    pub img_size: usize,
    #[config(default = 16)]
    pub patch_size: usize,
    #[config(default = 14)]
    pub window_size: usize,
    /// Channels of the neck output, the image embedding.
    #[config(default = 256)]
    pub out_chans: usize,
    /// Whether blocks outside `global_attn_indexes` attend within windows.
    /// Without windows every block attends globally but keeps its
    /// window-sized relative-position tables, as Cellpose-SAM does.
    #[config(default = true)]
    pub windowed: bool,
    /// Token grid side the global blocks' relative-position tables were trained
    /// for, when it is not `img_size / patch_size` (Cellpose-SAM keeps SAM's 64
    /// while its position embedding covers 32).
    #[config(default = "None")]
    pub rel_pos_grid: Option<usize>,
}

impl ImageEncoderConfig {
    pub fn vit_b() -> Self {
        Self::new(768, 12, 12, vec![2, 5, 8, 11])
    }

    pub fn vit_l() -> Self {
        Self::new(1024, 24, 16, vec![5, 11, 17, 23])
    }

    pub fn vit_h() -> Self {
        Self::new(1280, 32, 16, vec![7, 15, 23, 31])
    }

    /// Cellpose-SAM's encoder: ViT-L with 8 px patches on 256 px tiles,
    /// attending globally in every block.
    pub fn cellpose_sam() -> Self {
        Self::vit_l()
            .with_img_size(256)
            .with_patch_size(8)
            .with_windowed(false)
            .with_rel_pos_grid(Some(1024 / 16))
    }

    /// An encoder with placeholder parameters, to be filled from a checkpoint.
    pub fn init(&self, device: &Device) -> ImageEncoder {
        let (embed, out, grid) = (
            self.embed_dim,
            self.out_chans,
            self.img_size / self.patch_size,
        );
        let patch = [self.patch_size, self.patch_size];
        ImageEncoder {
            patch_embed: PatchEmbed {
                proj: Conv2dConfig::new([3, embed], patch)
                    .with_stride(patch)
                    .init(device),
            },
            pos_embed: Param::from_tensor(Tensor::zeros([1, grid, grid, embed], device)),
            blocks: (0..self.depth)
                .map(|i| {
                    let global = self.global_attn_indexes.contains(&i);
                    let tables = if global {
                        self.rel_pos_grid.unwrap_or(grid)
                    } else {
                        self.window_size
                    };
                    let window = if global || !self.windowed {
                        0
                    } else {
                        self.window_size
                    };
                    Block::new(embed, self.num_heads, window, tables, device)
                })
                .collect(),
            neck: (
                Conv2dConfig::new([embed, out], [1, 1])
                    .with_bias(false)
                    .init(device),
                LayerNorm2d::new(out, device),
                Conv2dConfig::new([out, out], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .with_bias(false)
                    .init(device),
                LayerNorm2d::new(out, device),
            ),
        }
    }
}

#[derive(Module, Debug)]
pub struct ImageEncoder {
    patch_embed: PatchEmbed,
    pos_embed: Param<Tensor<4>>,
    blocks: Vec<Block>,
    neck: (Conv2d, LayerNorm2d, Conv2d, LayerNorm2d),
}

impl ImageEncoder {
    /// The image embedding, `[batch, out_chans, H / patch, W / patch]`, for a
    /// normalized `[batch, 3, H, W]` input.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.forward_with_intermediates(x, &[]).0
    }

    /// The image embedding plus the token grid `[batch, H / patch, W / patch,
    /// embed_dim]` after each block in `taps` (0-based), in block order.
    ///
    /// Panics if a tap is not a block index, or if the token grid is larger
    /// than the position embedding (`img_size / patch_size` per side).
    pub fn forward_with_intermediates(
        &self,
        x: Tensor<4>,
        taps: &[usize],
    ) -> (Tensor<4>, Vec<Tensor<4>>) {
        assert!(
            taps.iter().all(|&tap| tap < self.blocks.len()),
            "taps {taps:?} must be below the depth {}",
            self.blocks.len()
        );
        let x = self.patch_embed.proj.forward(x).permute([0, 2, 3, 1]);
        let [_, height, width, embed] = x.dims();
        let [_, grid_h, grid_w, _] = self.pos_embed.dims();
        assert!(
            height <= grid_h && width <= grid_w,
            "token grid {height}x{width} exceeds the position embedding {grid_h}x{grid_w}"
        );
        let position = self
            .pos_embed
            .val()
            .slice([0..1, 0..height, 0..width, 0..embed]);
        let mut x = x + position;
        let mut intermediates = Vec::with_capacity(taps.len());
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x);
            if taps.contains(&i) {
                intermediates.push(x.clone());
            }
        }
        let (conv1, norm1, conv2, norm2) = &self.neck;
        let x = norm1.forward(conv1.forward(x.permute([0, 3, 1, 2])));
        (norm2.forward(conv2.forward(x)), intermediates)
    }

    pub fn depth(&self) -> usize {
        self.blocks.len()
    }
}

#[derive(Module, Debug)]
struct PatchEmbed {
    proj: Conv2d,
}

/// LayerNorm over the channels of an NCHW tensor, SAM's `LayerNorm2d`.
#[derive(Module, Debug)]
struct LayerNorm2d {
    weight: Param<Tensor<1>>,
    bias: Param<Tensor<1>>,
}

impl LayerNorm2d {
    fn new(channels: usize, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([channels], device)),
            bias: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let channels = x.dims()[1];
        let mean = x.clone().mean_dim(1);
        let centered = x - mean;
        let variance = (centered.clone() * centered.clone()).mean_dim(1);
        centered / (variance + 1e-6).sqrt() * self.weight.val().reshape([1, channels, 1, 1])
            + self.bias.val().reshape([1, channels, 1, 1])
    }
}

/// A tensor's values, for the unit tests.
#[cfg(test)]
fn test_values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().expect("f32 tensor")
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn presets_match_segment_anything() {
        let b = ImageEncoderConfig::vit_b();
        assert_eq!((b.embed_dim, b.depth, b.num_heads), (768, 12, 12));
        let l = ImageEncoderConfig::vit_l();
        assert_eq!(l.global_attn_indexes, [5, 11, 17, 23]);
        let h = ImageEncoderConfig::vit_h();
        assert_eq!((h.embed_dim, h.depth, h.num_heads), (1280, 32, 16));
        assert_eq!(
            (h.img_size, h.patch_size, h.window_size, h.out_chans),
            (1024, 16, 14, 256)
        );
    }

    #[test]
    fn small_encoder_shapes() {
        // Two blocks, one windowed and one global, on a 64 px input (4x4 tokens).
        let device = Device::flex();
        let config = ImageEncoderConfig::new(32, 2, 4, vec![1])
            .with_img_size(128)
            .with_window_size(3)
            .with_out_chans(8);
        let encoder = config.init(&device);
        let x = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
        let (embedding, taps) = encoder.forward_with_intermediates(x, &[0, 1]);
        assert_eq!(embedding.dims(), [2, 8, 4, 4]);
        assert_eq!(taps.len(), 2);
        assert!(taps.iter().all(|t| t.dims() == [2, 4, 4, 32]));
    }

    #[test]
    fn cellpose_sam_attends_globally_with_sam_sized_tables() {
        let config = ImageEncoderConfig::cellpose_sam();
        assert_eq!(
            (config.img_size / config.patch_size, config.embed_dim),
            (32, 1024)
        );
        // A two-block analogue: block 0 keeps 3x3-window tables, block 1 is
        // global with tables for an 8x8 grid, on a 4x4 token grid.
        let device = Device::flex();
        let encoder = ImageEncoderConfig::new(32, 2, 4, vec![1])
            .with_img_size(16)
            .with_patch_size(4)
            .with_window_size(3)
            .with_out_chans(8)
            .with_windowed(false)
            .with_rel_pos_grid(Some(8))
            .init(&device);
        assert_eq!(encoder.pos_embed.dims(), [1, 4, 4, 32]);
        let tables: Vec<usize> = encoder.blocks.iter().map(Block::table_rows).collect();
        assert_eq!(tables, [5, 15]);
        assert!(encoder.blocks.iter().all(|block| block.window() == 0));
        let x = Tensor::<4>::random([1, 3, 16, 16], Distribution::Default, &device);
        assert_eq!(encoder.forward(x).dims(), [1, 8, 4, 4]);
    }

    #[test]
    #[should_panic(expected = "exceeds the position embedding")]
    fn rejects_inputs_larger_than_the_embedding() {
        let device = Device::flex();
        let encoder = ImageEncoderConfig::new(32, 1, 4, vec![0])
            .with_img_size(32)
            .init(&device);
        encoder.forward(Tensor::<4>::zeros([1, 3, 64, 64], &device));
    }
}
