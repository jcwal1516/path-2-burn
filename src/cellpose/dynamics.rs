//! Cellpose 4.2's 2D flow integration, convergence histograms and mask cleanup.

use crate::ModelError;
use crate::hover::{fill_holes, remove_small_labels};

#[derive(Debug, Clone, Copy)]
pub struct DynamicsOptions {
    pub iterations: usize,
    pub cell_probability_threshold: f32,
    /// Mean squared flow-error cutoff; zero disables this filter.
    pub flow_threshold: f64,
    /// Minimum mask area; zero also disables hole filling, as in `compute_masks`.
    pub min_size: usize,
    pub max_size_fraction: f64,
}

impl Default for DynamicsOptions {
    fn default() -> Self {
        Self {
            iterations: 200,
            cell_probability_threshold: 0.0,
            flow_threshold: 0.4,
            min_size: 15,
            max_size_fraction: 0.4,
        }
    }
}

/// Decode one `[3, H, W]` array (dY, dX, cell logit). Returns row-major labels,
/// background zero, with consecutive IDs in order of first raster occurrence.
/// This matches Cellpose's 2D `compute_masks(..., min_size=15)` on the CPU.
pub fn compute_masks(
    flow: &[f32],
    height: usize,
    width: usize,
    options: &DynamicsOptions,
) -> Result<Vec<u32>, ModelError> {
    let pixels = height.checked_mul(width);
    if height < 2
        || width < 2
        || pixels.and_then(|n| n.checked_mul(3)) != Some(flow.len())
        || pixels.is_none_or(|n| n > u32::MAX as usize)
        || flow.iter().any(|v| !v.is_finite())
        || !options.cell_probability_threshold.is_finite()
        || !options.flow_threshold.is_finite()
        || !(0.0..=1.0).contains(&options.max_size_fraction)
    {
        return Err(ModelError::InvalidInput {
            model: super::NAME,
            message: "expected finite [3, H, W] flows with H,W >= 2 and valid dynamics options"
                .into(),
        });
    }
    let pixels = pixels.unwrap();
    let indices: Vec<usize> = (0..pixels)
        .filter(|&i| flow[2 * pixels + i] > options.cell_probability_threshold)
        .collect();
    if indices.is_empty() {
        return Ok(vec![0; pixels]);
    }
    let positions = follow(flow, &indices, height, width, options.iterations);
    let mut masks = clusters(
        &positions,
        &indices,
        height,
        width,
        options.max_size_fraction,
    );
    if options.flow_threshold > 0.0 && masks.iter().any(|&v| v != 0) {
        filter_flow_errors(&mut masks, flow, height, width, options.flow_threshold);
    }
    if options.min_size > 0 {
        cleanup(&mut masks, height, width, options.min_size);
    }
    Ok(masks)
}

fn follow(
    flow: &[f32],
    indices: &[usize],
    height: usize,
    width: usize,
    iterations: usize,
) -> Vec<[usize; 2]> {
    let pixels = height * width;
    let extents = [(height - 1) as f32, (width - 1) as f32];
    let mut field = vec![[0f32; 2]; pixels];
    for &i in indices {
        field[i] = [
            flow[i] / 5.0 * (2.0 / extents[0]),
            flow[pixels + i] / 5.0 * (2.0 / extents[1]),
        ];
    }
    let mut positions: Vec<[f32; 2]> = indices
        .iter()
        .map(|&i| {
            [
                (i / width) as f32 / extents[0] * 2.0 - 1.0,
                (i % width) as f32 / extents[1] * 2.0 - 1.0,
            ]
        })
        .collect();
    for _ in 0..iterations {
        for point in &mut positions {
            // Upstream normalizes using size-1, but samples with align_corners=False.
            let y = ((point[0] + 1.0) * height as f32 - 1.0) / 2.0;
            let x = ((point[1] + 1.0) * width as f32 - 1.0) / 2.0;
            let (y0, x0) = (y.floor() as isize, x.floor() as isize);
            let (y1, x1) = (y0 + 1, x0 + 1);
            let taps = [
                (y0, x0, (x1 as f32 - x) * (y1 as f32 - y)),
                (y0, x1, (x - x0 as f32) * (y1 as f32 - y)),
                (y1, x0, (x1 as f32 - x) * (y - y0 as f32)),
                (y1, x1, (x - x0 as f32) * (y - y0 as f32)),
            ];
            let mut delta = [0f32; 2];
            for (y, x, weight) in taps {
                if y >= 0 && x >= 0 && y < height as isize && x < width as isize {
                    let value = field[y as usize * width + x as usize];
                    for axis in 0..2 {
                        delta[axis] += value[axis] * weight;
                    }
                }
            }
            for axis in 0..2 {
                point[axis] = (point[axis] + delta[axis]).clamp(-1.0, 1.0);
            }
        }
    }
    positions
        .into_iter()
        .map(|p| [0, 1].map(|axis| (((p[axis] + 1.0) * 0.5) * extents[axis]) as usize))
        .collect()
}

fn clusters(
    positions: &[[usize; 2]],
    indices: &[usize],
    height: usize,
    width: usize,
    max_fraction: f64,
) -> Vec<u32> {
    const PAD: usize = 20;
    let (h, w) = (height + 2 * PAD, width + 2 * PAD);
    let endpoints: Vec<usize> = positions
        .iter()
        .map(|p| (p[0] + PAD) * w + p[1] + PAD)
        .collect();
    let mut histogram = vec![0u32; h * w];
    for &i in &endpoints {
        histogram[i] += 1;
    }
    let mut seeds: Vec<usize> = (2..h - 2)
        .flat_map(|y| (2..w - 2).map(move |x| y * w + x))
        .filter(|&i| {
            histogram[i] > 10
                && (-2isize..=2).all(|dy| {
                    (-2isize..=2).all(|dx| {
                        histogram[(i as isize + dy * w as isize + dx) as usize] <= histogram[i]
                    })
                })
        })
        .collect();
    seeds.sort_by_key(|&i| histogram[i]);
    let mut basin = vec![0u32; h * w];
    for (label, &seed) in seeds.iter().enumerate() {
        let mut region = [false; 121];
        region[60] = true;
        for _ in 0..5 {
            let previous = region;
            for y in 0usize..11 {
                for x in 0usize..11 {
                    let index =
                        (seed as isize + (y as isize - 5) * w as isize + x as isize - 5) as usize;
                    region[y * 11 + x] = histogram[index] > 2
                        && (y.saturating_sub(1)..=(y + 1).min(10)).any(|yy| {
                            (x.saturating_sub(1)..=(x + 1).min(10)).any(|xx| previous[yy * 11 + xx])
                        });
                }
            }
        }
        for (i, &inside) in region.iter().enumerate() {
            if inside {
                let index = (seed as isize + (i as isize / 11 - 5) * w as isize + i as isize % 11
                    - 5) as usize;
                basin[index] = (label + 1) as u32;
            }
        }
    }
    let mut labels = vec![0u32; height * width];
    let mut sizes = vec![0usize; seeds.len() + 1];
    for (&pixel, &endpoint) in indices.iter().zip(&endpoints) {
        let label = basin[endpoint];
        labels[pixel] = label;
        sizes[label as usize] += 1;
    }
    let maximum = (height * width) as f64 * max_fraction;
    for label in &mut labels {
        if sizes[*label as usize] as f64 > maximum {
            *label = 0;
        }
    }
    renumber(&mut labels);
    labels
}

fn renumber(labels: &mut [u32]) {
    let mut mapping = vec![0; labels.iter().copied().max().unwrap_or(0) as usize + 1];
    let mut next = 0;
    for label in labels {
        if *label != 0 {
            let entry = &mut mapping[*label as usize];
            if *entry == 0 {
                next += 1;
                *entry = next;
            }
            *label = *entry;
        }
    }
}

struct Object {
    pixels: Vec<usize>,
    bounds: [usize; 4],
}

fn objects(labels: &[u32], width: usize, height: usize) -> Vec<Object> {
    let count = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut objects: Vec<Object> = (0..count)
        .map(|_| Object {
            pixels: Vec::new(),
            bounds: [height, width, 0, 0],
        })
        .collect();
    for (i, &label) in labels.iter().enumerate() {
        if label == 0 {
            continue;
        }
        let object = &mut objects[label as usize - 1];
        object.pixels.push(i);
        let (y, x) = (i / width, i % width);
        object.bounds[0] = object.bounds[0].min(y);
        object.bounds[1] = object.bounds[1].min(x);
        object.bounds[2] = object.bounds[2].max(y + 1);
        object.bounds[3] = object.bounds[3].max(x + 1);
    }
    objects
}

fn filter_flow_errors(
    labels: &mut [u32],
    flow: &[f32],
    height: usize,
    width: usize,
    threshold: f64,
) {
    let objects = objects(labels, width, height);
    let stride = width + 2;
    let size = (height + 2) * stride;
    let mut padded = vec![0u32; size];
    let padded_index = |i: usize| (i / width + 1) * stride + i % width + 1;
    for (i, &label) in labels.iter().enumerate() {
        padded[padded_index(i)] = label;
    }
    let mut centers = Vec::with_capacity(objects.len());
    let mut iterations = 0;
    for object in &objects {
        let [y0, x0, y1, x1] = object.bounds;
        iterations = iterations.max(2 * (y1 - y0 + x1 - x0 + 2));
        // Cellpose rounds the center within its bounding box, ties to even.
        let cy = (object.pixels.iter().map(|&i| i / width - y0).sum::<usize>() as f64
            / object.pixels.len() as f64)
            .round_ties_even() as usize
            + y0;
        let cx = (object.pixels.iter().map(|&i| i % width - x0).sum::<usize>() as f64
            / object.pixels.len() as f64)
            .round_ties_even() as usize
            + x0;
        let center = *object
            .pixels
            .iter()
            .min_by_key(|&&i| {
                let dy = (i / width).abs_diff(cy) as u64;
                let dx = (i % width).abs_diff(cx) as u64;
                dy * dy + dx * dx
            })
            .unwrap();
        centers.push(padded_index(center));
    }
    let offsets = [
        0,
        -(stride as isize),
        stride as isize,
        -1,
        1,
        -(stride as isize) - 1,
        -(stride as isize) + 1,
        stride as isize - 1,
        stride as isize + 1,
    ];
    let active: Vec<usize> = objects
        .iter()
        .flat_map(|object| object.pixels.iter().map(|&i| padded_index(i)))
        .collect();
    let mut heat = vec![0f64; size];
    let mut next = vec![0f64; size];
    for _ in 0..iterations {
        for &center in &centers {
            heat[center] += 1.0;
        }
        for &i in &active {
            let sum: f64 = offsets
                .iter()
                .map(|&offset| {
                    let j = (i as isize + offset) as usize;
                    if padded[i] == padded[j] { heat[j] } else { 0.0 }
                })
                .sum();
            next[i] = sum / 9.0;
        }
        std::mem::swap(&mut heat, &mut next);
    }
    for object in &objects {
        let mut error = 0.0;
        for &i in &object.pixels {
            let j = padded_index(i);
            let dy = heat[j + stride] - heat[j - stride];
            let dx = heat[j + 1] - heat[j - 1];
            let norm = (dy * dy + dx * dx).sqrt() + 1e-60;
            error += (dy / norm - f64::from(flow[i] / 5.0)).powi(2)
                + (dx / norm - f64::from(flow[height * width + i] / 5.0)).powi(2);
        }
        if error / object.pixels.len() as f64 > threshold {
            for &i in &object.pixels {
                labels[i] = 0;
            }
        }
    }
    renumber(labels);
}

fn cleanup(labels: &mut [u32], height: usize, width: usize, min_size: usize) {
    let count = labels.iter().copied().max().unwrap_or(0);
    remove_small_labels(labels, count, min_size);
    renumber(labels);
    for (label, object) in objects(labels, width, height).iter().enumerate() {
        let [y0, x0, y1, x1] = object.bounds;
        let (h, w) = (y1 - y0, x1 - x0);
        let mask: Vec<bool> = (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| y * width + x))
            .map(|i| labels[i] == (label + 1) as u32)
            .collect();
        for (i, inside) in fill_holes(&mask, w, h).into_iter().enumerate() {
            if inside {
                labels[(y0 + i / w) * width + x0 + i % w] = (label + 1) as u32;
            }
        }
    }
    let count = labels.iter().copied().max().unwrap_or(0);
    remove_small_labels(labels, count, min_size);
    renumber(labels);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cells_and_invalid_shapes() {
        assert_eq!(
            compute_masks(&vec![0.0; 3 * 8 * 8], 8, 8, &DynamicsOptions::default()).unwrap(),
            vec![0; 64]
        );
        assert!(compute_masks(&[0.0; 3], 1, 1, &DynamicsOptions::default()).is_err());
        assert!(
            compute_masks(
                &vec![f32::NAN; 3 * 8 * 8],
                8,
                8,
                &DynamicsOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn normalization_and_grid_sampling_keep_boundary_positions_finite() {
        let mut flow = vec![0.0; 3 * 4 * 6];
        flow[..2 * 4 * 6].fill(-5.0);
        let indices = [0, 5, 18, 23];
        assert!(
            follow(&flow, &indices, 4, 6, 200)
                .iter()
                .all(|p| p[0] < 4 && p[1] < 6)
        );
    }
}
