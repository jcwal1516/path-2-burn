//! The three-depth StarDist U-Net, with an optional grid-2 input stem.

use crate::ops::upsample_nearest2x;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::MaxPool2dConfig;
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};

fn conv(input: usize, output: usize, kernel: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([input, output], [kernel, kernel])
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}

#[derive(Module, Debug)]
struct Pair {
    first: Conv2d,
    second: Conv2d,
}

impl Pair {
    fn new(input: usize, middle: usize, output: usize, device: &Device) -> Self {
        Self {
            first: conv(input, middle, 3, device),
            second: conv(middle, output, 3, device),
        }
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(self.second.forward(relu(self.first.forward(x))))
    }
}

#[derive(Module, Debug)]
pub(super) struct StarDistNet {
    stem: Option<Pair>,
    down: Vec<Pair>,
    middle: Pair,
    up: Vec<Pair>,
    features: Conv2d,
    prob: Conv2d,
    dist: Conv2d,
}

impl StarDistNet {
    pub(super) fn new(channels: usize, grid: usize, device: &Device) -> Self {
        let input = if grid == 2 { 32 } else { channels };
        Self {
            stem: (grid == 2).then(|| Pair::new(channels, 32, 32, device)),
            down: [(input, 32, 32), (32, 64, 64), (64, 128, 128)]
                .into_iter()
                .map(|(a, b, c)| Pair::new(a, b, c, device))
                .collect(),
            middle: Pair::new(128, 256, 128, device),
            up: [(64, 32, 32), (128, 64, 32), (256, 128, 64)]
                .into_iter()
                .map(|(a, b, c)| Pair::new(a, b, c, device))
                .collect(),
            features: conv(32, 128, 3, device),
            prob: conv(128, 1, 1, device),
            dist: conv(128, 32, 1, device),
        }
    }

    pub(super) fn forward(&self, mut x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let pool = MaxPool2dConfig::new([2, 2]).init();
        if let Some(stem) = &self.stem {
            x = pool.forward(stem.forward(x));
        }
        let mut skips = Vec::with_capacity(3);
        for down in &self.down {
            x = down.forward(x);
            skips.push(x.clone());
            x = pool.forward(x);
        }
        x = self.middle.forward(x);
        for (up, skip) in self.up.iter().rev().zip(skips.into_iter().rev()) {
            x = up.forward(Tensor::cat(vec![upsample_nearest2x(x), skip], 1));
        }
        x = relu(self.features.forward(x));
        (sigmoid(self.prob.forward(x.clone())), self.dist.forward(x))
    }
}
