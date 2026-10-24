//! HoVer-Net (Graham et al., 2019): a pre-activation ResNet-50 encoder and
//! dense decoders for nucleus probability, horizontal/vertical distances and,
//! when trained for them, nucleus types. [`crate::hover::decode`] turns the maps
//! into nuclei. Ported from vqdang/hover_net (MIT); the published checkpoints
//! load by their original parameter names.

use crate::hover::{DeviceMaps, device_maps};
use crate::ops::upsample_nearest2x;
use crate::preprocess::rgb_to_nchw;
use crate::runtime::{Checkpoint, load_pytorch_weights, read_f32, tensor_from_f32};
use crate::store::{ModelError, ModelStore};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::ops::PadMode;

const CHECKPOINT: Checkpoint<'static> = Checkpoint {
    // hover_net saves the network next to its optimizer state.
    state_dict_key: Some("desc"),
    renames: &[
        (r"^module\.", ""),
        (r"^conv0\./\.", "conv0.conv."),
        // Names such as `preact/bn` and `conv1/bn` become `preact_bn`, `conv1_bn`.
        ("/", "_"),
    ],
    // The constant 2x2 ones that `UpSample2x` multiplies by for nearest-neighbour
    // upsampling, which Burn does directly.
    unused: &["upsample2x.unpool_mat"],
};

/// HoVer-Net's two architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The paper's network: 270 px in, the central 80 px out, 5x5 decoders.
    Original,
    /// The PanNuke/MoNuSAC network: 256 px in, the central 164 px out, 3x3
    /// decoders, TensorFlow `same` padding on the first convolution.
    Fast,
}

impl Mode {
    /// Input tile side.
    pub fn input_size(self) -> usize {
        match self {
            Self::Original => 270,
            Self::Fast => 256,
        }
    }

    /// Side of the predicted maps, centred in the input tile.
    pub fn output_size(self) -> usize {
        match self {
            Self::Original => 80,
            Self::Fast => 164,
        }
    }

    fn decoder_kernel(self) -> usize {
        match self {
            Self::Original => 5,
            Self::Fast => 3,
        }
    }

    /// Centre crops of the first two encoder stages for the skip connections.
    fn skip_crops(self) -> [usize; 2] {
        match self {
            Self::Original => [184, 72],
            Self::Fast => [92, 36],
        }
    }
}

/// One published HoVer-Net checkpoint (vqdang/hover_net, Google Drive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoverNetWeights {
    /// The checkpoint's name in the weights manifest.
    pub name: &'static str,
    pub mode: Mode,
    /// Nucleus type channels including background, or `None` when the
    /// checkpoint segments without classifying.
    pub types: Option<usize>,
    /// Microns per pixel the tiles should be read at (40x).
    pub mpp: f64,
}

/// PanNuke, fast mode, 6 types (`type_info.json`: none, neoplastic,
/// inflammatory, connective, necrosis, non-neoplastic epithelial). The weights
/// carry PanNuke's CC BY-NC-SA 4.0 terms.
pub const PANNUKE: HoverNetWeights = HoverNetWeights {
    name: "hovernet_pannuke",
    mode: Mode::Fast,
    types: Some(6),
    mpp: 0.25,
};

/// MoNuSAC, fast mode, 5 types (none, epithelial, lymphocyte, macrophage,
/// neutrophil).
pub const MONUSAC: HoverNetWeights = HoverNetWeights {
    name: "hovernet_monusac",
    types: Some(5),
    ..PANNUKE
};

/// CoNSeP, original mode, 5 types (none, other, inflammatory, epithelial,
/// spindle-shaped).
pub const CONSEP: HoverNetWeights = HoverNetWeights {
    name: "hovernet_consep",
    mode: Mode::Original,
    types: Some(5),
    mpp: 0.25,
};

/// CoNSeP, original mode, segmentation only.
pub const CONSEP_NOTYPE: HoverNetWeights = HoverNetWeights {
    name: "hovernet_consep_notype",
    types: None,
    ..CONSEP
};

/// Kumar, original mode, segmentation only.
pub const KUMAR: HoverNetWeights = HoverNetWeights {
    name: "hovernet_kumar",
    ..CONSEP_NOTYPE
};

/// CPM17, original mode, segmentation only.
pub const CPM17: HoverNetWeights = HoverNetWeights {
    name: "hovernet_cpm17",
    ..CONSEP_NOTYPE
};

/// Raw outputs, NCHW float32, as HoVer-Net returns them.
pub struct HoverNetRaw {
    /// Background and nucleus logits, `[2, S, S]`.
    pub np: Vec<f32>,
    /// Horizontal then vertical distances, `[2, S, S]`.
    pub hv: Vec<f32>,
    /// Type logits, `[types, S, S]`; empty without a type branch.
    pub tp: Vec<f32>,
}

/// A HoVer-Net checkpoint on one device, fed one tile at a time.
pub struct HoverNet {
    weights: HoverNetWeights,
    network: Network,
    device: Device,
}

impl HoverNet {
    pub fn load(
        store: &ModelStore,
        device: &Device,
        weights: &HoverNetWeights,
    ) -> Result<Self, ModelError> {
        let network = Network::new(weights.mode, weights.types, device);
        Ok(Self {
            weights: *weights,
            network: load_pytorch_weights(network, store, weights.name, &CHECKPOINT)?,
            device: device.clone(),
        })
    }

    pub fn weights(&self) -> &HoverNetWeights {
        &self.weights
    }

    /// One HWC RGB u8 tile of the mode's input size to the network's NCHW
    /// input: RGB 0-255 as float, which the network scales itself.
    pub fn normalize_rgb(&self, rgb: &[u8]) -> Result<Vec<f32>, ModelError> {
        let side = self.weights.mode.input_size();
        rgb_to_nchw(rgb, side, Some(1), self.weights.name, |_, x| x as f32)
    }

    /// Network outputs for one `[1, 3, N, N]` RGB tile with values 0-255 (the
    /// network scales them itself), N being the mode's input size.
    pub fn forward_raw(&self, input: Vec<f32>) -> Result<HoverNetRaw, ModelError> {
        let name = self.weights.name;
        let (np, hv, tp) = self.run(input)?;
        Ok(HoverNetRaw {
            np: read_f32(np, name)?,
            hv: read_f32(hv, name)?,
            tp: tp
                .map(|tp| read_f32(tp, name))
                .transpose()?
                .unwrap_or_default(),
        })
    }

    /// Decoder-ready maps for [`crate::hover::decode`], reduced on the device.
    /// Without a type branch every pixel's type is 0.
    pub fn predict(&self, input: Vec<f32>) -> Result<DeviceMaps, ModelError> {
        let (np, hv, tp) = self.run(input)?;
        device_maps(np, hv, tp, self.weights.name)
    }

    fn run(
        &self,
        input: Vec<f32>,
    ) -> Result<(Tensor<4>, Tensor<4>, Option<Tensor<4>>), ModelError> {
        let n = self.weights.mode.input_size();
        let input = tensor_from_f32(input, [1, 3, n, n], &self.device, self.weights.name)?;
        Ok(self.network.forward(input))
    }
}

#[derive(Module, Debug)]
struct Network {
    conv0: Stem,
    d0: ResidualBlock,
    d1: ResidualBlock,
    d2: ResidualBlock,
    d3: ResidualBlock,
    conv_bot: Conv2d,
    decoder: Decoders,
    fast: bool,
}

impl Network {
    fn new(mode: Mode, types: Option<usize>, device: &Device) -> Self {
        let kernel = mode.decoder_kernel();
        let branch = |outputs| Branch::new(kernel, outputs, device);
        Self {
            conv0: Stem {
                conv: conv(3, 64, 7, device),
                bn: BatchNormConfig::new(64).init(device),
            },
            d0: ResidualBlock::new(64, [64, 64, 256], 3, 1, device),
            d1: ResidualBlock::new(256, [128, 128, 512], 4, 2, device),
            d2: ResidualBlock::new(512, [256, 256, 1024], 6, 2, device),
            d3: ResidualBlock::new(1024, [512, 512, 2048], 3, 2, device),
            conv_bot: conv(2048, 1024, 1, device),
            decoder: Decoders {
                tp: types.map(branch),
                np: branch(2),
                hv: branch(2),
            },
            fast: mode == Mode::Fast,
        }
    }

    fn mode(&self) -> Mode {
        if self.fast {
            Mode::Fast
        } else {
            Mode::Original
        }
    }

    /// Nucleus logits, HV maps and type logits for RGB 0-255 input.
    fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>, Option<Tensor<4>>) {
        let mode = self.mode();
        let x = x / 255.0;
        let x = if self.fast { same_pad(x, 7, 1) } else { x };
        let d0 = relu(self.conv0.bn.forward(self.conv0.conv.forward(x)));
        let d0 = self.d0.forward(d0);
        let d1 = self.d1.forward(d0.clone());
        let d2 = self.d2.forward(d1.clone());
        let d3 = self.conv_bot.forward(self.d3.forward(d2.clone()));
        let [crop0, crop1] = mode.skip_crops();
        let skips = [
            center_crop(d0, crop0, crop0),
            center_crop(d1, crop1, crop1),
            d2,
            d3,
        ];
        let decoder = &self.decoder;
        let run = |branch: &Branch| branch.forward(&skips, mode.decoder_kernel());
        (
            run(&decoder.np),
            run(&decoder.hv),
            decoder.tp.as_ref().map(run),
        )
    }
}

#[derive(Module, Debug)]
struct Stem {
    conv: Conv2d,
    bn: BatchNorm,
}

/// Bias-free valid convolution, as HoVer-Net uses throughout.
fn conv(input: usize, output: usize, kernel: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([input, output], [kernel, kernel])
        .with_bias(false)
        .init(device)
}

/// TensorFlow's `same` padding (`TFSamepaddingLayer`): any odd pixel goes to
/// the bottom and right. Square inputs.
fn same_pad(x: Tensor<4>, kernel: usize, stride: usize) -> Tensor<4> {
    let side = x.dims()[2];
    let remainder = side % stride;
    let pad = kernel.saturating_sub(if remainder == 0 { stride } else { remainder });
    let (start, end) = (pad / 2, pad - pad / 2);
    x.pad(
        [(0, 0), (0, 0), (start, end), (start, end)],
        PadMode::Constant(0.0),
    )
}

/// `crop_op`: removes `rows` and `cols` pixels, the extra one from the end.
fn center_crop(x: Tensor<4>, rows: usize, cols: usize) -> Tensor<4> {
    let [batch, channels, height, width] = x.dims();
    let (top, left) = (rows / 2, cols / 2);
    let (bottom, right) = (rows - top, cols - left);
    x.slice([
        0..batch,
        0..channels,
        top..height - bottom,
        left..width - right,
    ])
}

/// A batch norm followed by ReLU (`blk_bna`).
#[derive(Module, Debug)]
struct BnRelu {
    bn: BatchNorm,
}

impl BnRelu {
    fn new(channels: usize, device: &Device) -> Self {
        Self {
            bn: BatchNormConfig::new(channels).init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(self.bn.forward(x))
    }
}

/// Pre-activation bottleneck units; the first has no pre-activation and does
/// the stride.
#[derive(Module, Debug)]
struct ResidualBlock {
    units: Vec<ResidualUnit>,
    shortcut: Option<Conv2d>,
    blk_bna: BnRelu,
}

impl ResidualBlock {
    fn new(input: usize, widths: [usize; 3], count: usize, stride: usize, device: &Device) -> Self {
        let [narrow, middle, output] = widths;
        let units = (0..count)
            .map(|i| {
                let unit_input = if i == 0 { input } else { output };
                ResidualUnit {
                    preact_bn: (i > 0).then(|| BatchNormConfig::new(unit_input).init(device)),
                    conv1: conv(unit_input, narrow, 1, device),
                    conv1_bn: BatchNormConfig::new(narrow).init(device),
                    conv2: Conv2dConfig::new([narrow, middle], [3, 3])
                        .with_stride(if i == 0 { [stride; 2] } else { [1; 2] })
                        .with_bias(false)
                        .init(device),
                    conv2_bn: BatchNormConfig::new(middle).init(device),
                    conv3: conv(middle, output, 1, device),
                    stride: if i == 0 { stride } else { 1 },
                }
            })
            .collect();
        let shortcut = (input != output || stride != 1).then(|| {
            Conv2dConfig::new([input, output], [1, 1])
                .with_stride([stride; 2])
                .with_bias(false)
                .init(device)
        });
        Self {
            units,
            shortcut,
            blk_bna: BnRelu::new(output, device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mut shortcut = match &self.shortcut {
            Some(projection) => projection.forward(x.clone()),
            None => x.clone(),
        };
        let mut x = x;
        for unit in &self.units {
            x = unit.forward(x) + shortcut;
            shortcut = x.clone();
        }
        self.blk_bna.forward(x)
    }
}

#[derive(Module, Debug)]
struct ResidualUnit {
    preact_bn: Option<BatchNorm>,
    conv1: Conv2d,
    conv1_bn: BatchNorm,
    conv2: Conv2d,
    conv2_bn: BatchNorm,
    conv3: Conv2d,
    stride: usize,
}

impl ResidualUnit {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = match &self.preact_bn {
            Some(bn) => relu(bn.forward(x)),
            None => x,
        };
        let x = relu(self.conv1_bn.forward(self.conv1.forward(x)));
        let x = same_pad(x, 3, self.stride);
        let x = relu(self.conv2_bn.forward(self.conv2.forward(x)));
        self.conv3.forward(x)
    }
}

/// DenseNet units with valid convolutions; each concatenates its output to its
/// centre-cropped input.
#[derive(Module, Debug)]
struct DenseBlock {
    units: Vec<DenseUnit>,
    blk_bna: BnRelu,
}

impl DenseBlock {
    fn new(input: usize, count: usize, kernel: usize, device: &Device) -> Self {
        let units = (0..count)
            .map(|i| {
                let unit_input = input + i * 32;
                DenseUnit {
                    preact_bna_bn: BatchNormConfig::new(unit_input).init(device),
                    conv1: conv(unit_input, 128, 1, device),
                    conv1_bn: BatchNormConfig::new(128).init(device),
                    conv2: Conv2dConfig::new([128, 32], [kernel, kernel])
                        .with_groups(4)
                        .with_bias(false)
                        .init(device),
                }
            })
            .collect();
        Self {
            units,
            blk_bna: BnRelu::new(input + count * 32, device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.units.iter().fold(x, |previous, unit| {
            let new = unit.forward(previous.clone());
            let [_, _, height, width] = new.dims();
            let [_, _, previous_height, previous_width] = previous.dims();
            let previous = center_crop(previous, previous_height - height, previous_width - width);
            Tensor::cat(vec![previous, new], 1)
        });
        self.blk_bna.forward(x)
    }
}

#[derive(Module, Debug)]
struct DenseUnit {
    preact_bna_bn: BatchNorm,
    conv1: Conv2d,
    conv1_bn: BatchNorm,
    conv2: Conv2d,
}

impl DenseUnit {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = relu(self.preact_bna_bn.forward(x));
        let x = relu(self.conv1_bn.forward(self.conv1.forward(x)));
        self.conv2.forward(x)
    }
}

#[derive(Module, Debug)]
struct Decoders {
    tp: Option<Branch>,
    np: Branch,
    hv: Branch,
}

/// One decoder: two dense stages, a `same` convolution and the 1x1 head.
#[derive(Module, Debug)]
struct Branch {
    u3: DenseStage,
    u2: DenseStage,
    u1: SameConv,
    u0: Head,
}

impl Branch {
    fn new(kernel: usize, outputs: usize, device: &Device) -> Self {
        Self {
            u3: DenseStage::new(1024, 256, 8, kernel, device),
            u2: DenseStage::new(512, 128, 4, kernel, device),
            u1: SameConv {
                conva: conv(256, 64, kernel, device),
            },
            u0: Head {
                bn: BatchNormConfig::new(64).init(device),
                conv: Conv2dConfig::new([64, outputs], [1, 1]).init(device),
            },
        }
    }

    /// `skips` are the cropped d0 and d1, then d2 and the bottleneck d3.
    fn forward(&self, skips: &[Tensor<4>; 4], kernel: usize) -> Tensor<4> {
        let [d0, d1, d2, d3] = skips;
        let u3 = self.u3.forward(upsample_nearest2x(d3.clone()) + d2.clone());
        let u2 = self.u2.forward(upsample_nearest2x(u3) + d1.clone());
        let u1 = upsample_nearest2x(u2) + d0.clone();
        let u1 = self.u1.conva.forward(same_pad(u1, kernel, 1));
        self.u0.conv.forward(relu(self.u0.bn.forward(u1)))
    }
}

/// A valid convolution narrowing to `width`, a dense block, and a 1x1
/// convolution.
#[derive(Module, Debug)]
struct DenseStage {
    conva: Conv2d,
    dense: DenseBlock,
    convf: Conv2d,
}

impl DenseStage {
    fn new(input: usize, width: usize, units: usize, kernel: usize, device: &Device) -> Self {
        let output = width + units * 32;
        Self {
            conva: conv(input, width, kernel, device),
            dense: DenseBlock::new(width, units, kernel, device),
            convf: conv(output, output, 1, device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.convf
            .forward(self.dense.forward(self.conva.forward(x)))
    }
}

#[derive(Module, Debug)]
struct SameConv {
    conva: Conv2d,
}

#[derive(Module, Debug)]
struct Head {
    bn: BatchNorm,
    conv: Conv2d,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_padding_puts_the_odd_pixel_last() {
        let device = Device::flex();
        let pad = |side: usize, kernel, stride| {
            same_pad(
                Tensor::<4>::zeros([1, 1, side, side], &device),
                kernel,
                stride,
            )
            .dims()[2]
        };
        // TFSamepaddingLayer: k=7,s=1 pads 3+3; k=3,s=2 on an even side pads 0+1.
        assert_eq!(pad(256, 7, 1), 262);
        assert_eq!(pad(128, 3, 2), 129);
        assert_eq!(pad(129, 3, 2), 131);
    }

    #[test]
    fn fast_mode_maps_256_to_164() {
        // The full network on the CPU backend: shapes only, random weights.
        let device = Device::flex();
        let network = Network::new(Mode::Fast, Some(6), &device);
        let x = Tensor::<4>::zeros([1, 3, 256, 256], &device);
        let (np, hv, tp) = network.forward(x);
        assert_eq!(np.dims(), [1, 2, 164, 164]);
        assert_eq!(hv.dims(), [1, 2, 164, 164]);
        assert_eq!(tp.expect("type branch").dims(), [1, 6, 164, 164]);
    }
}
