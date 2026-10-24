//! InstanSeg's U-Net (`InstanSeg_UNet`, instanseg-torch 0.1.1, Apache-2.0) in
//! Burn, with its parameter names: four residual encoder stages, three decoder
//! stages and 1x1 heads for the two field coordinates, two sigmas and the seed
//! map. Every convolution is followed by batch norm, and all but the heads by
//! ReLU.

use crate::ops::upsample_nearest2x;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::relu;

/// Encoder widths, shallow to deep.
const WIDTHS: [usize; 4] = [32, 64, 128, 256];
/// Head outputs: field coordinates, sigmas, seed.
const HEADS: [usize; 3] = [2, 2, 1];

/// `conv_norm_act`: an `nn.Sequential` of a same-padded convolution and batch
/// norm, so its parameters are `0.*` and `1.*`.
type ConvNorm = (Conv2d, BatchNorm);

fn conv_norm(input: usize, output: usize, kernel: usize, device: &Device) -> ConvNorm {
    let pad = kernel / 2;
    (
        Conv2dConfig::new([input, output], [kernel, kernel])
            .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
            .init(device),
        BatchNormConfig::new(output).init(device),
    )
}

fn conv_norm_relu(layer: &ConvNorm, x: Tensor<4>) -> Tensor<4> {
    relu(layer.1.forward(layer.0.forward(x)))
}

/// A block's `conv0`..`conv4`: a 1x1 projection and a 3x3 convolution from
/// `input` channels, then three 3x3 convolutions at `output`.
fn block_convs(input: usize, output: usize, device: &Device) -> [ConvNorm; 5] {
    [
        conv_norm(input, output, 1, device),
        conv_norm(input, output, 3, device),
        conv_norm(output, output, 3, device),
        conv_norm(output, output, 3, device),
        conv_norm(output, output, 3, device),
    ]
}

/// `x + conv4(conv3(x))`, how every encoder and decoder block ends.
fn residual(x: Tensor<4>, conv3: &ConvNorm, conv4: &ConvNorm) -> Tensor<4> {
    x.clone() + conv_norm_relu(conv4, conv_norm_relu(conv3, x))
}

#[derive(Module, Debug)]
pub(super) struct Network {
    encoder: Vec<EncoderBlock>,
    decoders: Vec<Decoder>,
}

impl Network {
    pub(super) fn new(device: &Device) -> Self {
        let encoder = (0..WIDTHS.len())
            .map(|i| {
                let input = if i == 0 { 3 } else { WIDTHS[i - 1] };
                EncoderBlock::new(input, WIDTHS[i], i > 0, device)
            })
            .collect();
        Self {
            encoder,
            decoders: vec![Decoder::new(device)],
        }
    }

    /// Normalized `[batch, 3, H, W]` (sides multiples of 8) to `[batch, 5, H, W]`.
    pub(super) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mut skips = Vec::with_capacity(WIDTHS.len() - 1);
        let mut x = x;
        for (i, block) in self.encoder.iter().enumerate() {
            x = block.forward(x);
            if i + 1 < self.encoder.len() {
                skips.push(x.clone());
            }
        }
        let outputs = self
            .decoders
            .iter()
            .map(|decoder| decoder.forward(x.clone(), &skips))
            .collect();
        Tensor::cat(outputs, 1)
    }
}

#[derive(Module, Debug)]
struct EncoderBlock {
    conv0: ConvNorm,
    conv1: ConvNorm,
    conv2: ConvNorm,
    conv3: ConvNorm,
    conv4: ConvNorm,
    pool: bool,
}

impl EncoderBlock {
    fn new(input: usize, output: usize, pool: bool, device: &Device) -> Self {
        let [conv0, conv1, conv2, conv3, conv4] = block_convs(input, output, device);
        Self {
            conv0,
            conv1,
            conv2,
            conv3,
            conv4,
            pool,
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = if self.pool {
            MaxPool2dConfig::new([2, 2]).init().forward(x)
        } else {
            x
        };
        let projection = conv_norm_relu(&self.conv0, x.clone());
        let x = projection + conv_norm_relu(&self.conv2, conv_norm_relu(&self.conv1, x));
        residual(x, &self.conv3, &self.conv4)
    }
}

#[derive(Module, Debug)]
struct Decoder {
    decoder: Vec<DecoderBlock>,
    final_block: Vec<ConvNorm>,
}

impl Decoder {
    fn new(device: &Device) -> Self {
        // Deep to shallow: 256 -> 128 -> 64 -> 32.
        let decoder = (1..WIDTHS.len())
            .rev()
            .map(|i| DecoderBlock::new(WIDTHS[i], WIDTHS[i - 1], device))
            .collect();
        let final_block = HEADS
            .iter()
            .map(|&outputs| conv_norm(WIDTHS[0], outputs, 1, device))
            .collect();
        Self {
            decoder,
            final_block,
        }
    }

    /// `skips` shallow to deep, as the encoder produced them.
    fn forward(&self, x: Tensor<4>, skips: &[Tensor<4>]) -> Tensor<4> {
        let x = self
            .decoder
            .iter()
            .zip(skips.iter().rev())
            .fold(x, |x, (block, skip)| block.forward(x, skip.clone()));
        // The heads have batch norm but no activation.
        let heads = self
            .final_block
            .iter()
            .map(|(conv, norm)| norm.forward(conv.forward(x.clone())))
            .collect();
        Tensor::cat(heads, 1)
    }
}

#[derive(Module, Debug)]
struct DecoderBlock {
    conv0: ConvNorm,
    conv_skip: ConvNorm,
    conv1: ConvNorm,
    conv2: ConvNorm,
    conv3: ConvNorm,
    conv4: ConvNorm,
}

impl DecoderBlock {
    fn new(input: usize, output: usize, device: &Device) -> Self {
        let [conv0, conv1, conv2, conv3, conv4] = block_convs(input, output, device);
        Self {
            conv0,
            conv_skip: conv_norm(output, output, 1, device),
            conv1,
            conv2,
            conv3,
            conv4,
        }
    }

    fn forward(&self, x: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
        let x = upsample_nearest2x(x);
        let projection = conv_norm_relu(&self.conv0, x.clone());
        let x = conv_norm_relu(&self.conv1, x) + conv_norm_relu(&self.conv_skip, skip);
        let x = projection + conv_norm_relu(&self.conv2, x);
        residual(x, &self.conv3, &self.conv4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_rgb_to_five_channels_at_full_resolution() {
        let device = Device::flex();
        let network = Network::new(&device);
        let x = Tensor::<4>::zeros([1, 3, 64, 64], &device);
        assert_eq!(network.forward(x).dims(), [1, 5, 64, 64]);
    }
}
