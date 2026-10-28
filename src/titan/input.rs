//! Sparse equivalent of upstream's grid scatter-add and background masking.

use super::{EMBEDDING_DIM, NAME};
use crate::{Device, ModelError, tensor_from_f32};
use burn::prelude::Tensor;
use std::collections::BTreeMap;

fn invalid(message: &str) -> ModelError {
    ModelError::InvalidInput {
        model: NAME,
        message: message.to_string(),
    }
}

pub(super) fn prepare(
    features: Vec<f32>,
    coords: &[[i64; 2]],
    patch_size: u32,
) -> Result<(Vec<f32>, Vec<[u64; 2]>), ModelError> {
    if patch_size == 0
        || coords.is_empty()
        || features.len().checked_div(EMBEDDING_DIM) != Some(coords.len())
        || features.len() % EMBEDDING_DIM != 0
    {
        return Err(invalid(
            "expected nonempty [N, 768] features, N coordinates and positive patch spacing",
        ));
    }
    if features.iter().any(|v| !v.is_finite()) {
        return Err(invalid("features must be finite"));
    }
    let offset = [0, 1].map(|axis| coords.iter().map(|c| c[axis]).min().unwrap());
    let mut cells: BTreeMap<[u64; 2], Vec<f32>> = BTreeMap::new();
    for (feature, coord) in features.chunks_exact(EMBEDDING_DIM).zip(coords) {
        let cell = [0, 1].map(|axis| {
            ((i128::from(coord[axis]) - i128::from(offset[axis])) / i128::from(patch_size)) as u64
        });
        let summed = cells
            .entry(cell)
            .or_insert_with(|| vec![0.0; EMBEDDING_DIM]);
        for (sum, value) in summed.iter_mut().zip(feature) {
            *sum += value;
        }
    }
    let (mut values, mut positions) = (Vec::new(), Vec::new());
    for (position, feature) in cells {
        if feature.iter().any(|v| !v.is_finite()) {
            return Err(invalid("features overflowed when summing a grid cell"));
        }
        if feature.iter().any(|&v| v != 0.0) {
            positions.push(position);
            values.extend(feature);
        }
    }
    if positions.is_empty() {
        return Err(invalid(
            "no nonzero feature cells remain after summing duplicates",
        ));
    }
    Ok((values, positions))
}

pub(super) fn alibi(positions: &[[u64; 2]], device: &Device) -> Result<Tensor<4>, ModelError> {
    let tokens = positions
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid("too many patches"))?;
    let length = tokens
        .checked_mul(tokens)
        .and_then(|n| n.checked_mul(12))
        .ok_or_else(|| invalid("attention bias size overflow"))?;
    let mut bias = Vec::new();
    bias.try_reserve_exact(length)
        .map_err(|_| invalid("not enough memory for attention bias"))?;
    bias.resize(length, 0.0);
    // ALiBi's non-power-of-two rule: eight slopes, then every other slope of
    // the sixteen-head sequence. Class-token rows/columns have zero bias.
    let slopes: [f32; 12] = std::array::from_fn(|head| {
        if head < 8 {
            2_f32.powi(-((head + 1) as i32))
        } else {
            2_f32.powf(-((2 * (head - 8) + 1) as f32) / 2.0)
        }
    });
    for (i, a) in positions.iter().enumerate() {
        for (j, b) in positions.iter().enumerate() {
            let dx = a[0].abs_diff(b[0]) as f64;
            let dy = a[1].abs_diff(b[1]) as f64;
            let distance = (dx * dx + dy * dy).sqrt() as f32;
            for (head, slope) in slopes.iter().enumerate() {
                bias[(head * tokens + i + 1) * tokens + j + 1] = -distance * slope;
            }
        }
    }
    tensor_from_f32(bias, [1, 12, tokens, tokens], device, NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_cells_sum_duplicates_drop_zeros_and_follow_grid_order() {
        let coords = [[203, 407], [3, 7], [250, 499], [103, 7], [303, 207]];
        let values = [2.0, 3.0, -2.0, 4.0, 0.0]
            .into_iter()
            .flat_map(|v| vec![v; EMBEDDING_DIM])
            .collect();
        let (values, positions) = prepare(values, &coords, 100).unwrap();
        assert_eq!(positions, [[0, 0], [1, 0]]);
        assert_eq!(
            values,
            [vec![3.0; EMBEDDING_DIM], vec![4.0; EMBEDDING_DIM]].concat()
        );
    }

    #[test]
    fn invalid_or_empty_slides_fail_before_inference() {
        assert!(prepare(vec![], &[], 512).is_err());
        assert!(prepare(vec![1.0; 767], &[[0, 0]], 512).is_err());
        assert!(prepare(vec![1.0; 768], &[[0, 0]], 0).is_err());
        assert!(prepare(vec![f32::NAN; 768], &[[0, 0]], 512).is_err());
        assert!(prepare(vec![0.0; 768], &[[0, 0]], 512).is_err());
    }
}
