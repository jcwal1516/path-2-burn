//! Per-seed windows: feature engineering, pixel classifier and fragment cleanup,
//! following the scripted `brightfield_nuclei` forward (`feature_engineering_slow`,
//! `ProbabilityNet`, `flood_fill`, `fill_holes`).

use super::super::params::PostprocessParams;
use std::collections::VecDeque;

/// Probabilities for one seed's `side x side` window at (`top`, `left`).
pub(super) struct Crop {
    pub top: usize,
    pub left: usize,
    pub probs: Vec<f32>,
}

/// Embedding fields with the coordinate map added, two `[H, W]` planes, plus the
/// two sigma planes, borrowed from the network output.
pub(super) struct Fields<'a> {
    pub width: usize,
    pub height: usize,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub sigma0: &'a [f32],
    pub sigma1: &'a [f32],
}

impl<'a> Fields<'a> {
    /// `fields = (sigmoid(raw[0..2]) - 0.5) * 8 + linear coordinate map`.
    pub fn new(raw: &'a [f32], width: usize, height: usize) -> Self {
        let plane = width * height;
        let xx = linspace((width * 64) as f32 / 256.0, width);
        let yy = linspace((height * 64) as f32 / 256.0, height);
        let field = |v: f32| (sigmoid(v) - 0.5) * 8.0;
        let x = (0..plane).map(|i| field(raw[i]) + xx[i % width]).collect();
        let y = (0..plane)
            .map(|i| field(raw[plane + i]) + yy[i / width])
            .collect();
        Self {
            width,
            height,
            x,
            y,
            sigma0: &raw[2 * plane..3 * plane],
            sigma1: &raw[3 * plane..4 * plane],
        }
    }
}

#[inline]
pub(super) fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// `torch.linspace(0, end, steps)` in float32: the first half counts up from the
/// start, the second half counts down from the end.
fn linspace(end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![0.0];
    }
    let step = end / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            }
        })
        .collect()
}

/// Window side length; the scripted model caps the half-width at the image size.
pub(super) fn half_window(params: &PostprocessParams, width: usize, height: usize) -> usize {
    params.window_size.min(width).min(height)
}

pub(super) fn build(fields: &Fields<'_>, seed: (usize, usize), params: &PostprocessParams) -> Crop {
    let (w, h) = (fields.width, fields.height);
    let ws = half_window(params, w, h);
    let side = 2 * ws;
    let (r, c) = seed;
    let top = r.clamp(ws, h - ws) - ws;
    let left = c.clamp(ws, w - ws) - ws;
    let center = r * w + c;
    let (cx, cy) = (fields.x[center], fields.y[center]);

    let mut probs = Vec::with_capacity(side * side);
    for i in 0..side {
        let row = (top + i) * w;
        for j in 0..side {
            let p = row + left + j;
            let features = [
                fields.x[p] - cx,
                fields.y[p] - cy,
                fields.sigma0[p],
                fields.sigma1[p],
            ];
            probs.push(sigmoid(params.classifier.logit(&features)));
        }
    }
    if params.cleanup_fragments {
        cleanup(&mut probs, side, (r - top, c - left), params.mask_threshold);
    }
    Crop { top, left, probs }
}

/// Keep only the 8-connected fragment reachable from the seed, then fill its holes
/// with the mask threshold, as `flood_fill`/`fill_holes` do with max-pooling.
fn cleanup(probs: &mut [f32], side: usize, seed: (usize, usize), threshold: f32) {
    let mask: Vec<bool> = probs.iter().map(|&p| p >= threshold).collect();
    // flood_fill iterates at most max(H, W) times: reach is capped at `side` steps.
    let filled = reconstruct(&mask, side, &[seed.0 * side + seed.1], true, side);
    let ring = dilate(&filled, side, 3);
    let background: Vec<bool> = filled.iter().map(|f| !f).collect();
    let sources: Vec<usize> = (0..side * side).filter(|&i| !ring[i]).collect();
    let outside = reconstruct(&background, side, &sources, false, side);
    for i in 0..side * side {
        if !filled[i] {
            probs[i] = if outside[i] { 0.0 } else { threshold };
        }
    }
}

/// Pixels of `mask` within `limit` 8-connected steps of `sources` (geodesic
/// reconstruction by repeated 3x3 dilation). With `source_needs_mask`, a source
/// outside the mask still seeds its neighbours but is not itself kept.
fn reconstruct(
    mask: &[bool],
    side: usize,
    sources: &[usize],
    source_needs_mask: bool,
    limit: usize,
) -> Vec<bool> {
    let mut reached = vec![false; side * side];
    let mut visited = vec![false; side * side];
    let mut queue = VecDeque::with_capacity(sources.len());
    for &s in sources {
        visited[s] = true;
        reached[s] = !source_needs_mask || mask[s];
        queue.push_back((s, 0usize));
    }
    while let Some((u, d)) = queue.pop_front() {
        if d == limit {
            continue;
        }
        let (ur, uc) = (u / side, u % side);
        for nr in ur.saturating_sub(1)..=(ur + 1).min(side - 1) {
            for nc in uc.saturating_sub(1)..=(uc + 1).min(side - 1) {
                let v = nr * side + nc;
                if !visited[v] && mask[v] {
                    visited[v] = true;
                    reached[v] = true;
                    queue.push_back((v, d + 1));
                }
            }
        }
    }
    reached
}

/// Square (Chebyshev) dilation by `radius`, clipped at the window edge.
fn dilate(mask: &[bool], side: usize, radius: usize) -> Vec<bool> {
    let mut rows = vec![false; side * side];
    for r in 0..side {
        for c in 0..side {
            let (c0, c1) = (c.saturating_sub(radius), (c + radius).min(side - 1));
            rows[r * side + c] = mask[r * side + c0..=r * side + c1].iter().any(|&m| m);
        }
    }
    let mut out = vec![false; side * side];
    for r in 0..side {
        let (r0, r1) = (r.saturating_sub(radius), (r + radius).min(side - 1));
        for c in 0..side {
            out[r * side + c] = (r0..=r1).any(|q| rows[q * side + c]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> Vec<bool> {
        rows.iter()
            .flat_map(|r| r.chars().map(|ch| ch == '#'))
            .collect()
    }

    #[test]
    fn linspace_matches_torch_endpoints() {
        let v = linspace(256.0, 1024);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1023], 256.0);
        assert!((v[512] - 512.0 * 256.0 / 1023.0).abs() < 1e-4);
    }

    #[test]
    fn reconstruct_keeps_seed_component_only() {
        let mask = grid(&["##..", "#...", "...#", "..##"]);
        let got = reconstruct(&mask, 4, &[0], true, 4);
        assert_eq!(got, grid(&["##..", "#...", "....", "...."]));
    }

    #[test]
    fn reconstruct_crosses_diagonals_and_respects_limit() {
        let mask = grid(&["#...", ".#..", "..#.", "...#"]);
        assert_eq!(reconstruct(&mask, 4, &[0], true, 4), mask);
        assert_eq!(
            reconstruct(&mask, 4, &[0], true, 2),
            grid(&["#...", ".#..", "..#.", "...."])
        );
    }

    #[test]
    fn seed_outside_mask_still_grows_into_neighbours() {
        let mask = grid(&[".#", "##"]);
        assert_eq!(reconstruct(&mask, 2, &[0], true, 2), grid(&[".#", "##"]));
    }

    #[test]
    fn cleanup_drops_fragments_and_fills_holes() {
        // 12x12 window: ring object around a hole at the seed's right, plus a stray fragment.
        let side = 12;
        let mut probs = vec![0.1f32; side * side];
        for r in 2..7 {
            for c in 2..7 {
                probs[r * side + c] = 0.9;
            }
        }
        probs[4 * side + 4] = 0.2; // hole inside the object
        probs[10 * side + 10] = 0.95; // fragment
        cleanup(&mut probs, side, (2, 2), 0.53);
        assert_eq!(probs[4 * side + 4], 0.53);
        assert_eq!(probs[10 * side + 10], 0.0);
        assert_eq!(probs[2 * side + 2], 0.9);
        assert_eq!(probs[0], 0.0);
    }
}
