//! SAM's transformer block: pre-norm attention (windowed or global) and MLP.

use crate::attention::Attention;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::ops::PadMode;

#[derive(Module, Debug)]
pub(crate) struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    /// Attention window side; 0 attends globally.
    window: usize,
}

impl Block {
    /// `window` is the attention window side (0 attends globally); `tables` is
    /// the grid side the relative-position tables were trained for.
    pub(crate) fn new(
        embed: usize,
        heads: usize,
        window: usize,
        tables: usize,
        device: &Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(embed).with_epsilon(1e-6).init(device),
            attn: Attention::new(embed, heads, tables, device),
            norm2: LayerNormConfig::new(embed).with_epsilon(1e-6).init(device),
            mlp: Mlp {
                lin1: LinearConfig::new(embed, 4 * embed).init(device),
                lin2: LinearConfig::new(4 * embed, embed).init(device),
            },
            window,
        }
    }

    #[cfg(test)]
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    #[cfg(test)]
    pub(crate) fn table_rows(&self) -> usize {
        self.attn.table_rows()
    }

    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let normalized = self.norm1.forward(x.clone());
        let attended = if self.window > 0 {
            let [batch, height, width, _] = x.dims();
            let windows = self.attn.forward(window_partition(normalized, self.window));
            window_unpartition(windows, self.window, batch, height, width)
        } else {
            self.attn.forward(normalized)
        };
        let x = x + attended;
        let residual = self
            .mlp
            .lin2
            .forward(gelu(self.mlp.lin1.forward(self.norm2.forward(x.clone()))));
        x + residual
    }
}

#[derive(Module, Debug)]
struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

/// SAM zero-pads the token grid to multiples of the window (64x64 becomes 70x70
/// for 14x14 windows), then splits it row-major into windows.
fn window_partition(x: Tensor<4>, window: usize) -> Tensor<4> {
    let [batch, height, width, channels] = x.dims();
    let (rows, cols) = (height.div_ceil(window), width.div_ceil(window));
    x.pad(
        [
            (0, 0),
            (0, rows * window - height),
            (0, cols * window - width),
            (0, 0),
        ],
        PadMode::Constant(0.0),
    )
    .reshape([batch, rows, window, cols, window, channels])
    .permute([0, 1, 3, 2, 4, 5])
    .reshape([batch * rows * cols, window, window, channels])
}

fn window_unpartition(
    windows: Tensor<4>,
    window: usize,
    batch: usize,
    height: usize,
    width: usize,
) -> Tensor<4> {
    let channels = windows.dims()[3];
    let (rows, cols) = (height.div_ceil(window), width.div_ceil(window));
    windows
        .reshape([batch, rows, cols, window, window, channels])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([batch, rows * window, cols * window, channels])
        .slice([0..batch, 0..height, 0..width, 0..channels])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_values as values;
    use burn::tensor::Distribution;

    const WINDOW: usize = 14;
    const CHANNELS: usize = 8;

    #[test]
    fn windows_follow_sam_layout_and_round_trip() {
        // A 1024 px tile: 64x64 tokens, padded to 70x70, 5x5 windows.
        let device = Device::flex();
        let x = Tensor::<4>::random([1, 64, 64, CHANNELS], Distribution::Default, &device);
        let windows = window_partition(x.clone(), WINDOW);
        assert_eq!(windows.dims(), [25, WINDOW, WINDOW, CHANNELS]);
        // Window (row 1, column 2) covers grid rows 14..28 and columns 28..42.
        assert_eq!(
            values(
                windows
                    .clone()
                    .slice([7..8, 0..WINDOW, 0..WINDOW, 0..CHANNELS])
            ),
            values(x.clone().slice([0..1, 14..28, 28..42, 0..CHANNELS]))
        );
        // The last window's rows past the 64x64 grid are zero padding.
        let padding = values(
            windows
                .clone()
                .slice([24..25, 8..WINDOW, 0..WINDOW, 0..CHANNELS]),
        );
        assert!(padding.iter().all(|&v| v == 0.0));
        assert_eq!(
            values(window_unpartition(windows, WINDOW, 1, 64, 64)),
            values(x)
        );
    }

    #[test]
    fn small_tiles_pad_to_whole_windows() {
        // A 256 px tile has a 16x16 grid: padded to 28x28, four windows.
        let device = Device::flex();
        let x = Tensor::<4>::random([2, 16, 16, CHANNELS], Distribution::Default, &device);
        let windows = window_partition(x.clone(), WINDOW);
        assert_eq!(windows.dims(), [2 * 4, WINDOW, WINDOW, CHANNELS]);
        assert_eq!(
            values(windows.clone().slice([5..6, 0..2, 0..2, 0..CHANNELS])),
            values(x.clone().slice([1..2, 0..2, 14..16, 0..CHANNELS]))
        );
        assert_eq!(
            values(window_unpartition(windows, WINDOW, 2, 16, 16)),
            values(x)
        );
    }
}
