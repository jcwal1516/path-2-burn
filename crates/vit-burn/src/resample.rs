//! Position-embedding resizing, as PyTorch's bicubic `F.interpolate` computes it
//! for each [`PosResample`] method.

use crate::PosResample;
use burn::prelude::*;

/// Resizes a `[height, width, channels]` grid to `[out_h, out_w, channels]`.
/// Both methods are separable, so each axis is one matrix product.
pub(crate) fn resize(
    grid: Tensor<3>,
    out_h: usize,
    out_w: usize,
    method: PosResample,
) -> Tensor<3> {
    let [height, width, channels] = grid.dims();
    let device = grid.device();
    let matrix = |input: usize, output: usize| {
        let weights = match method {
            PosResample::Antialiased => antialiased_weights(input, output),
            PosResample::Bicubic => bicubic_weights(input, output, input as f32 / output as f32),
            PosResample::Dino => {
                let factor = (output as f64 + 0.1) / input as f64;
                bicubic_weights(input, output, (1.0 / factor) as f32)
            }
        };
        Tensor::<2>::from_data(TensorData::new(weights, [output, input]), &device)
    };
    let rows = matrix(height, out_h)
        .matmul(grid.reshape([height, width * channels]))
        .reshape([out_h, width, channels])
        .swap_dims(0, 1)
        .reshape([width, out_h * channels]);
    matrix(width, out_w)
        .matmul(rows)
        .reshape([out_w, out_h, channels])
        .swap_dims(0, 1)
}

/// `[output, input]` row-normalized weights of PyTorch's antialiased bicubic
/// resize along one axis (`_compute_weights_aa`, PIL's `a = -0.5` kernel
/// stretched by the scale when shrinking), in float32 like PyTorch.
fn antialiased_weights(input: usize, output: usize) -> Vec<f32> {
    let scale = input as f32 / output as f32;
    let (support, inverse) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    let mut weights = vec![0f32; output * input];
    for (i, row) in weights.chunks_exact_mut(input).enumerate() {
        let center = scale * (i as f32 + 0.5);
        // C++ casts truncate toward zero, as `as` does.
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let mut total = 0f32;
        for (j, weight) in row.iter_mut().enumerate().take(end).skip(start) {
            *weight = keys_kernel((j as f32 - center + 0.5) * inverse, -0.5);
            total += *weight;
        }
        if total != 0.0 {
            row[start..end].iter_mut().for_each(|w| *w /= total);
        }
    }
    weights
}

/// `[output, input]` weights of PyTorch's plain bicubic `F.interpolate`
/// (`a = -0.75`, not antialiased, border indices clamped), sampling the input
/// at `scale` input pixels per output pixel: `input / output` for a target
/// size, `1 / scale_factor` for a scale factor. In float32 like PyTorch.
fn bicubic_weights(input: usize, output: usize, scale: f32) -> Vec<f32> {
    let last = input as i64 - 1;
    let mut weights = vec![0f32; output * input];
    for (i, row) in weights.chunks_exact_mut(input).enumerate() {
        let real = scale * (i as f32 + 0.5) - 0.5;
        let index = (real.floor() as i64).min(last);
        let t = (real - index as f32).clamp(0.0, 1.0);
        let taps = [t + 1.0, t, 1.0 - t, 2.0 - t];
        for (k, distance) in taps.into_iter().enumerate() {
            let column = (index - 1 + k as i64).clamp(0, last) as usize;
            row[column] += keys_kernel(distance, -0.75);
        }
    }
    weights
}

/// Keys' cubic convolution kernel with parameter `a`.
fn keys_kernel(x: f32, a: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `torch.arange(12.).reshape(1, 1, 3, 4) ** 2` resized to `out_h x out_w`
    /// matches torch's output.
    fn assert_resizes_squares(method: PosResample, out_h: usize, out_w: usize, expected: &[f32]) {
        let device = Device::flex();
        let squares: Vec<f32> = (0..12).map(|v| (v * v) as f32).collect();
        let grid = Tensor::<3>::from_data(TensorData::new(squares, [3, 4, 1]), &device);
        let resized = resize(grid, out_h, out_w, method)
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32 tensor");
        assert_eq!(resized.len(), expected.len());
        // Float32 summed in a different order than torch, with negative kernel
        // lobes: about 1e-6 relative. A wrong kernel or scale is off by 0.1+.
        for (got, want) in resized.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-5 + 1e-5 * want.abs(),
                "{resized:?}"
            );
        }
    }

    #[test]
    fn matches_torch_bicubic_antialias() {
        // F.interpolate(squares, size=(2, 5), mode="bicubic", antialias=True)
        #[rustfmt::skip]
        let expected = [
            3.522_044_4, 5.784_291_7, 9.640_042, 15.118_432, 20.431_118,
            47.217_567, 57.114_677, 70.378_41, 85.264_78, 98.212_334,
        ];
        assert_resizes_squares(PosResample::Antialiased, 2, 5, &expected);
    }

    #[test]
    fn matches_torch_sized_bicubic() {
        // F.interpolate(squares, size=(5, 6), mode="bicubic")
        #[rustfmt::skip]
        let expected = [
            -1.556_139_5, -1.629_250_5, -1.037_001_4, 0.244_332_2, 3.253_250_1, 5.527_360_4,
            2.445_766_2, 3.997_767_9, 7.248_945_2, 10.422_432, 16.090_277, 19.989_498,
            15.218_75, 19.468_75, 27.134_253, 33.449_074, 43.531_25, 50.128_468,
            45.271_755, 52.219_75, 64.299_576, 73.755_737, 88.252_251, 97.547_47,
            67.065_651, 75.638_763, 90.377_518, 101.725_838, 118.881_271, 129.801_605,
        ];
        assert_resizes_squares(PosResample::Bicubic, 5, 6, &expected);
    }

    #[test]
    fn matches_torch_dino_scale_factor_bicubic() {
        // F.interpolate(squares, scale_factor=(5.1 / 3, 6.1 / 4), mode="bicubic")
        #[rustfmt::skip]
        let expected = [
            -1.577_565_6, -1.652_765_3, -1.111_363_2, 0.077_687_17, 2.987_670_9, 5.374_241_8,
            2.343_345_4, 3.795_187_2, 6.857_819_6, 9.844_461, 15.301_426, 19.334_87,
            13.873_88, 17.841_114, 25.056_793, 31.004_484, 40.656_914, 47.403_137,
            43.222_046, 49.840_218, 61.432_74, 70.501_05, 84.575_03, 94.180_22,
            66.290_59, 74.568_05, 88.900_13, 99.921_7, 116.763_23, 128.157_91,
        ];
        assert_resizes_squares(PosResample::Dino, 5, 6, &expected);
    }

    #[test]
    fn same_size_is_the_identity_and_rows_sum_to_one() {
        for n in [1, 5, 16, 37] {
            let w = antialiased_weights(n, n);
            for (i, row) in w.chunks_exact(n).enumerate() {
                assert!((row[i] - 1.0).abs() < 1e-6, "n={n} row {i}");
            }
        }
        for (input, output) in [(37, 16), (16, 37), (14, 9)] {
            for weights in [
                antialiased_weights(input, output),
                bicubic_weights(input, output, input as f32 / output as f32),
                bicubic_weights(input, output, 0.9 * input as f32 / output as f32),
            ] {
                for row in weights.chunks_exact(input) {
                    assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                }
            }
        }
    }
}
