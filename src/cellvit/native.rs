//! CellViT's networks in Burn, with the original checkpoint parameter names.
//!
//! A ViT encoder's outputs after each quarter of its depth feed U-Net style
//! decoders for the nucleus map, the horizontal/vertical distance maps and the
//! nucleus types; a linear head gives the tissue type. The encoder is SAM's
//! image encoder ([`sam_burn`], CellViT-SAM) or HIPT's ViT-S/16 ([`vit_burn`],
//! CellViT-256). Architecture follows the CellViT sources retained in
//! `export/cellvit_ref`. All inference is float32.

use super::model::{NUCLEI_CLASSES, TISSUE_CLASSES};
use crate::runtime::{Checkpoint, load_pytorch_weights};
use crate::store::{ModelError, ModelStore};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::relu;
use sam_burn::{ImageEncoder, ImageEncoderConfig};
use vit_burn::{PosResample, VisionTransformer, VitConfig};

/// CellViT checkpoints, official and fine-tuned, nest their weights here.
const STATE_DICT_KEY: &str = "model_state_dict";

/// The checkpoints keep the decoders at the top level; here they sit in `heads`.
const HEADS: (&str, &str) = (
    r"^(decoder[0-3]|nuclei_binary_map_decoder|hv_map_decoder|nuclei_type_maps_decoder)\.",
    "heads.${1}.",
);
const SAM_RENAMES: &[(&str, &str)] = &[HEADS];
/// HIPT's classifier head on the class token gives the tissue type.
const VIT_RENAMES: &[(&str, &str)] = &[HEADS, (r"^encoder\.head\.", "tissue_head.")];

/// HIPT's ViT-S/16 as CellViT-256 builds it: DINO-trained at 224 px, so other
/// tile sizes resize its position embedding the DINO way.
const HIPT_VIT_S16: VitConfig = VitConfig {
    embed_dim: 384,
    depth: 12,
    num_heads: 6,
    mlp_hidden: 1536,
    pos_resample: PosResample::Dino,
    ..VitConfig::BASE_PATCH16_224
};

/// The encoder a CellViT checkpoint was built on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backbone {
    /// SAM's ViT-H image encoder (CellViT-SAM-H).
    SamH,
    /// HIPT's ViT-S/16 (CellViT-256).
    Vit256,
}

/// Nucleus map logits, HV maps and nucleus type logits (NCHW), then tissue
/// logits `[batch, classes]`.
pub(super) type Outputs = (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<2>);

pub(super) enum Network {
    Sam(SamCellVit),
    Vit(VitCellVit),
}

impl Network {
    pub(super) fn load(
        store: &ModelStore,
        device: &Device,
        name: &str,
        backbone: Backbone,
    ) -> Result<Self, ModelError> {
        let checkpoint = |renames: &'static [(&'static str, &'static str)]| Checkpoint {
            state_dict_key: Some(STATE_DICT_KEY),
            renames,
            unused: &[],
        };
        Ok(match backbone {
            Backbone::SamH => {
                let model = SamCellVit::new(&ImageEncoderConfig::vit_h(), device);
                Self::Sam(load_pytorch_weights(
                    model,
                    store,
                    name,
                    &checkpoint(SAM_RENAMES),
                )?)
            }
            Backbone::Vit256 => {
                let model = VitCellVit::new(&HIPT_VIT_S16, device);
                Self::Vit(load_pytorch_weights(
                    model,
                    store,
                    name,
                    &checkpoint(VIT_RENAMES),
                )?)
            }
        })
    }

    /// A normalized `[batch, 3, H, W]` input; sides are multiples of 16.
    pub(super) fn forward(&self, x: Tensor<4>) -> Outputs {
        match self {
            Self::Sam(model) => model.forward(x),
            Self::Vit(model) => model.forward(x),
        }
    }
}

/// CellViT's skip connections: the blocks ending each quarter of the depth.
fn quarter_taps(depth: usize) -> [usize; 4] {
    [1, 2, 3, 4].map(|quarter| quarter * depth / 4 - 1)
}

/// The outputs for [`quarter_taps`], one per tap.
fn four<T>(taps: Vec<T>) -> [T; 4] {
    let count = taps.len();
    taps.try_into()
        .unwrap_or_else(|_| unreachable!("four taps gave {count} outputs"))
}

/// CellViT-SAM: the tissue type comes from the pooled neck output.
#[derive(Module, Debug)]
pub(super) struct SamCellVit {
    encoder: ImageEncoder,
    classifier_head: Linear,
    heads: Heads,
}

impl SamCellVit {
    fn new(config: &ImageEncoderConfig, device: &Device) -> Self {
        Self {
            encoder: config.init(device),
            classifier_head: LinearConfig::new(config.out_chans, TISSUE_CLASSES.len()).init(device),
            heads: Heads::new(config.embed_dim, device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Outputs {
        let taps = quarter_taps(self.encoder.depth());
        let (embedding, skips) = self.encoder.forward_with_intermediates(x.clone(), &taps);
        let [batch, channels, height, width] = embedding.dims();
        let pooled = embedding
            .reshape([batch, channels, height * width])
            .mean_dim(2)
            .reshape([batch, channels]);
        let skips = four(skips).map(|z| z.permute([0, 3, 1, 2]));
        let (binary, hv, types) = self.heads.forward(x, skips);
        (binary, hv, types, self.classifier_head.forward(pooled))
    }
}

/// CellViT-256: the tissue type comes from the class token.
#[derive(Module, Debug)]
pub(super) struct VitCellVit {
    encoder: VisionTransformer,
    tissue_head: Linear,
    heads: Heads,
}

impl VitCellVit {
    fn new(config: &VitConfig, device: &Device) -> Self {
        Self {
            encoder: config.init(device),
            tissue_head: LinearConfig::new(config.embed_dim, TISSUE_CLASSES.len()).init(device),
            heads: Heads::new(config.embed_dim, device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Outputs {
        let [_, _, height, width] = x.dims();
        let patch = self.encoder.patch_size();
        let (grid_h, grid_w) = (height / patch, width / patch);
        let prefix = self.encoder.prefix_tokens();
        let taps = quarter_taps(self.encoder.depth());
        let (tokens, skips) = self.encoder.forward_with_intermediates(x.clone(), &taps);
        let [batch, count, embed] = tokens.dims();
        let class = tokens
            .slice([0..batch, 0..1, 0..embed])
            .reshape([batch, embed]);
        let skips = four(skips).map(|z| {
            z.slice([0..batch, prefix..count, 0..embed])
                .reshape([batch, grid_h, grid_w, embed])
                .permute([0, 3, 1, 2])
        });
        let (binary, hv, types) = self.heads.forward(x, skips);
        (binary, hv, types, self.tissue_head.forward(class))
    }
}

/// The decoders every CellViT shares: skip branches from the input and the
/// encoder taps, and one upsampling branch per output map.
#[derive(Module, Debug)]
struct Heads {
    decoder0: Vec<ConvBlock>,
    decoder1: Vec<DeconvBlock>,
    decoder2: Vec<DeconvBlock>,
    decoder3: Vec<DeconvBlock>,
    nuclei_binary_map_decoder: Decoder,
    hv_map_decoder: Decoder,
    nuclei_type_maps_decoder: Decoder,
}

impl Heads {
    fn new(embed: usize, device: &Device) -> Self {
        // CellViT narrows the skips and bottleneck for encoders below 512 wide.
        let (skip11, skip12, bottleneck) = if embed < 512 {
            (256, 128, 312)
        } else {
            (512, 256, 512)
        };
        let deconv = |input, output| DeconvBlock::new(input, output, device);
        Self {
            decoder0: vec![
                ConvBlock::new(3, 32, device),
                ConvBlock::new(32, 64, device),
            ],
            decoder1: vec![
                deconv(embed, skip11),
                deconv(skip11, skip12),
                deconv(skip12, 128),
            ],
            decoder2: vec![deconv(embed, skip11), deconv(skip11, 256)],
            decoder3: vec![deconv(embed, bottleneck)],
            nuclei_binary_map_decoder: Decoder::new(embed, bottleneck, 2, device),
            hv_map_decoder: Decoder::new(embed, bottleneck, 2, device),
            nuclei_type_maps_decoder: Decoder::new(embed, bottleneck, NUCLEI_CLASSES.len(), device),
        }
    }

    /// The input image and the four encoder taps (NCHW) to the three maps.
    fn forward(&self, x: Tensor<4>, taps: [Tensor<4>; 4]) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [z1, z2, z3, z4] = taps;
        // The skips are shared by all three branches; compute each only once.
        let s0 = self.decoder0.iter().fold(x, |x, block| block.forward(x));
        let deconv = |blocks: &[DeconvBlock], z: Tensor<4>| {
            blocks.iter().fold(z, |z, block| block.forward(z))
        };
        let s1 = deconv(&self.decoder1, z1);
        let s2 = deconv(&self.decoder2, z2);
        let s3 = deconv(&self.decoder3, z3);
        let skips = [&s0, &s1, &s2, &s3];
        (
            self.nuclei_binary_map_decoder.forward(z4.clone(), skips),
            self.hv_map_decoder.forward(z4.clone(), skips),
            self.nuclei_type_maps_decoder.forward(z4, skips),
        )
    }
}

#[derive(Module, Debug)]
struct ConvBlock {
    block: (Conv2d, BatchNorm),
}

/// A same-padded 3x3 convolution.
fn conv3x3(input: usize, output: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([input, output], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

impl ConvBlock {
    fn new(input: usize, output: usize, device: &Device) -> Self {
        Self {
            block: (
                conv3x3(input, output, device),
                BatchNormConfig::new(output).init(device),
            ),
        }
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(self.block.1.forward(self.block.0.forward(x)))
    }
}

#[derive(Module, Debug)]
struct DeconvBlock {
    block: (ConvTranspose2d, Conv2d, BatchNorm),
}

impl DeconvBlock {
    fn new(input: usize, output: usize, device: &Device) -> Self {
        Self {
            block: (
                upsample(input, output, device),
                conv3x3(output, output, device),
                BatchNormConfig::new(output).init(device),
            ),
        }
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(
            self.block
                .2
                .forward(self.block.1.forward(self.block.0.forward(x))),
        )
    }
}

fn upsample(input: usize, output: usize, device: &Device) -> ConvTranspose2d {
    ConvTranspose2dConfig::new([input, output], [2, 2])
        .with_stride([2, 2])
        .init(device)
}

/// One output branch: upsample the last encoder tap through the shared skips.
#[derive(Module, Debug)]
struct Decoder {
    bottleneck_upsampler: ConvTranspose2d,
    decoder3_upsampler: (ConvBlock, ConvBlock, ConvBlock, ConvTranspose2d),
    decoder2_upsampler: (ConvBlock, ConvBlock, ConvTranspose2d),
    decoder1_upsampler: (ConvBlock, ConvBlock, ConvTranspose2d),
    decoder0_header: (ConvBlock, ConvBlock, Conv2d),
}

impl Decoder {
    fn new(embed: usize, bottleneck: usize, classes: usize, device: &Device) -> Self {
        let conv = |input, output| ConvBlock::new(input, output, device);
        Self {
            bottleneck_upsampler: upsample(embed, bottleneck, device),
            decoder3_upsampler: (
                conv(2 * bottleneck, bottleneck),
                conv(bottleneck, bottleneck),
                conv(bottleneck, bottleneck),
                upsample(bottleneck, 256, device),
            ),
            decoder2_upsampler: (conv(512, 256), conv(256, 256), upsample(256, 128, device)),
            decoder1_upsampler: (conv(256, 128), conv(128, 128), upsample(128, 64, device)),
            decoder0_header: (
                conv(128, 64),
                conv(64, 64),
                Conv2dConfig::new([64, classes], [1, 1]).init(device),
            ),
        }
    }

    fn forward(&self, x: Tensor<4>, skips: [&Tensor<4>; 4]) -> Tensor<4> {
        let x = self.bottleneck_upsampler.forward(x);
        let x = Tensor::cat(vec![skips[3].clone(), x], 1);
        let (a, b, c, up) = &self.decoder3_upsampler;
        let x = up.forward(c.forward(b.forward(a.forward(x))));
        let x = Tensor::cat(vec![skips[2].clone(), x], 1);
        let (a, b, up) = &self.decoder2_upsampler;
        let x = up.forward(b.forward(a.forward(x)));
        let x = Tensor::cat(vec![skips[1].clone(), x], 1);
        let (a, b, up) = &self.decoder1_upsampler;
        let x = up.forward(b.forward(a.forward(x)));
        let x = Tensor::cat(vec![skips[0].clone(), x], 1);
        let (a, b, out) = &self.decoder0_header;
        out.forward(b.forward(a.forward(x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_are_the_last_block_of_each_quarter() {
        // CellViT's extract_layers, 1-based: [8, 16, 24, 32] for SAM-H and
        // [3, 6, 9, 12] for ViT-S/16.
        assert_eq!(quarter_taps(32), [7, 15, 23, 31]);
        assert_eq!(quarter_taps(12), [2, 5, 8, 11]);
    }
}
