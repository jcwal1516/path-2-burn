//! Small tensor operations several networks share.

use burn::prelude::*;

/// Nearest-neighbour 2x upsampling of NCHW: `F.interpolate(scale_factor=2,
/// mode="nearest")`, and TensorPack's `FixedUnpooling`.
pub(crate) fn upsample_nearest2x(x: Tensor<4>) -> Tensor<4> {
    let [batch, channels, height, width] = x.dims();
    x.reshape([batch, channels, height, 1, width, 1])
        .expand([batch, channels, height, 2, width, 2])
        .reshape([batch, channels, 2 * height, 2 * width])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsampling_repeats_each_pixel() {
        let device = Device::flex();
        let x = Tensor::<4>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
            &device,
        );
        let up = upsample_nearest2x(x)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(
            up,
            [
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0
            ]
        );
    }
}
