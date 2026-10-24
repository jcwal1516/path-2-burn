//! Multi-head attention with SAM's decomposed relative positions.

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

/// Keeps each score tensor bounded when attending globally over large grids.
const SCORE_BUDGET: usize = 2 * 1024 * 1024 * 1024;

#[derive(Module, Debug)]
pub(crate) struct Attention {
    qkv: Linear,
    proj: Linear,
    rel_pos_h: Param<Tensor<2>>,
    rel_pos_w: Param<Tensor<2>>,
    heads: usize,
}

impl Attention {
    /// `side` is the grid side the relative-position tables are trained for.
    pub(crate) fn new(embed: usize, heads: usize, side: usize, device: &Device) -> Self {
        let head_dim = embed / heads;
        Self {
            qkv: LinearConfig::new(embed, embed * 3).init(device),
            proj: LinearConfig::new(embed, embed).init(device),
            rel_pos_h: Param::from_tensor(Tensor::zeros([2 * side - 1, head_dim], device)),
            rel_pos_w: Param::from_tensor(Tensor::zeros([2 * side - 1, head_dim], device)),
            heads,
        }
    }

    #[cfg(test)]
    pub(crate) fn table_rows(&self) -> usize {
        self.rel_pos_h.dims()[0]
    }

    /// `[batch, height, width, embed]` tokens attending over the whole grid.
    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, height, width, embed] = x.dims();
        let head_dim = embed / self.heads;
        let pixels = height * width;
        let heads = batch * self.heads;
        let device = x.device();
        let qkv = self
            .qkv
            .forward(x)
            .reshape([batch, pixels, 3, self.heads, head_dim])
            .permute([2, 0, 3, 1, 4])
            .reshape([3, heads, pixels, head_dim]);
        let q = qkv
            .clone()
            .slice([0..1, 0..heads, 0..pixels, 0..head_dim])
            .reshape([heads, pixels, head_dim]);
        let k = qkv
            .clone()
            .slice([1..2, 0..heads, 0..pixels, 0..head_dim])
            .reshape([heads, pixels, head_dim]);
        let v = qkv
            .slice([2..3, 0..heads, 0..pixels, 0..head_dim])
            .reshape([heads, pixels, head_dim]);
        let rel_h = relative_positions(self.rel_pos_h.val(), height, &device);
        let rel_w = relative_positions(self.rel_pos_w.val(), width, &device);
        let group_size = (SCORE_BUDGET / (pixels * pixels * size_of::<f32>())).max(1);
        let mut groups = Vec::new();
        for start in (0..heads).step_by(group_size) {
            let end = (start + group_size).min(heads);
            let count = end - start;
            let query = q.clone().slice([start..end, 0..pixels, 0..head_dim]);
            let keys = k.clone().slice([start..end, 0..pixels, 0..head_dim]);
            let values = v.clone().slice([start..end, 0..pixels, 0..head_dim]);
            let scores =
                (query.clone() * (head_dim as f64).powf(-0.5)).matmul(keys.swap_dims(1, 2));
            // The relative terms use the unscaled query, as SAM does.
            let qr = query.reshape([count, height, width, head_dim]);
            let rh = qr
                .clone()
                .permute([1, 0, 2, 3])
                .reshape([height, count * width, head_dim])
                .matmul(rel_h.clone().swap_dims(1, 2))
                .reshape([height, count, width, height])
                .permute([1, 0, 2, 3])
                .reshape([count, height, width, height, 1]);
            let rw = qr
                .permute([2, 0, 1, 3])
                .reshape([width, count * height, head_dim])
                .matmul(rel_w.clone().swap_dims(1, 2))
                .reshape([width, count, height, width])
                .permute([1, 2, 0, 3])
                .reshape([count, height, width, 1, width]);
            let scores = (scores.reshape([count, height, width, height, width]) + rh + rw)
                .reshape([count, pixels, pixels]);
            groups.push(softmax(scores, 2).matmul(values));
        }
        let x = Tensor::cat(groups, 0)
            .reshape([batch, self.heads, height, width, head_dim])
            .permute([0, 2, 3, 1, 4])
            .reshape([batch, height, width, embed]);
        self.proj.forward(x)
    }
}

/// SAM's decomposed relative positions for equal query and key sizes: entry
/// (q, k) is table row `q - k + side - 1`. A table trained for another grid is
/// first resized to `2 * side - 1` rows, as SAM does.
fn relative_positions(table: Tensor<2>, side: usize, device: &Device) -> Tensor<3> {
    let head_dim = table.dims()[1];
    let table = resize_rows(table, 2 * side - 1, device);
    let indices: Vec<i32> = (0..side)
        .flat_map(|q| (0..side).map(move |k| (q + side - 1 - k) as i32))
        .collect();
    let indices = Tensor::<1, Int>::from_data(TensorData::new(indices, [side * side]), device);
    table.select(0, indices).reshape([side, side, head_dim])
}

/// `F.interpolate(mode="linear", align_corners=False)` along the rows, with
/// PyTorch's float32 source-index arithmetic.
fn resize_rows(table: Tensor<2>, rows: usize, device: &Device) -> Tensor<2> {
    let [len, columns] = table.dims();
    if len == rows {
        return table;
    }
    let scale = len as f32 / rows as f32;
    let (mut lower, mut upper) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    let (mut lower_weight, mut upper_weight) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    for row in 0..rows {
        let source = (scale * (row as f32 + 0.5) - 0.5).max(0.0);
        let index = (source as usize).min(len - 1);
        let lambda = (source - index as f32).clamp(0.0, 1.0);
        lower.push(index as i32);
        upper.push((index + usize::from(index < len - 1)) as i32);
        lower_weight.push(1.0 - lambda);
        upper_weight.push(lambda);
    }
    let rows_of =
        |values: Vec<i32>| Tensor::<1, Int>::from_data(TensorData::new(values, [rows]), device);
    let weights = |values: Vec<f32>| {
        Tensor::<2>::from_data(TensorData::new(values, [rows, 1]), device).expand([rows, columns])
    };
    table.clone().select(0, rows_of(lower)) * weights(lower_weight)
        + table.select(0, rows_of(upper)) * weights(upper_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_values as values;

    #[test]
    fn resize_rows_matches_torch_linear_interpolation() {
        let device = Device::flex();
        let column = |rows: &[f32]| {
            Tensor::<2>::from_data(TensorData::new(rows.to_vec(), [rows.len(), 1]), &device)
        };
        // Expected values from torch.nn.functional.interpolate(mode="linear").
        for (input, size, expected) in [
            (
                &[0.0, 10.0, 20.0, 30.0, 40.0][..],
                3,
                &[3.333_333, 20.0, 36.666_664][..],
            ),
            (
                &[0.0, 10.0, 40.0][..],
                5,
                &[0.0, 4.0, 10.0, 28.000_004, 40.0][..],
            ),
        ] {
            let resized = values(resize_rows(column(input), size, &device));
            for (got, want) in resized.iter().zip(expected) {
                assert!((got - want).abs() < 1e-5, "{input:?} -> {resized:?}");
            }
        }
        let identity = [1.0, 2.0, 3.0];
        assert_eq!(values(resize_rows(column(&identity), 3, &device)), identity);
    }

    #[test]
    fn relative_positions_index_query_minus_key() {
        let device = Device::flex();
        let (side, head_dim) = (3, 4);
        let rows: Vec<f32> = (0..2 * side - 1)
            .flat_map(|r| vec![r as f32; head_dim])
            .collect();
        let table =
            Tensor::<2>::from_data(TensorData::new(rows, [2 * side - 1, head_dim]), &device);
        let positions = values(relative_positions(table, side, &device));
        for q in 0..side {
            for k in 0..side {
                let expected = (q + side - 1 - k) as f32;
                let start = (q * side + k) * head_dim;
                assert!(
                    positions[start..start + head_dim]
                        .iter()
                        .all(|&v| v == expected),
                    "q={q} k={k}"
                );
            }
        }
    }
}
