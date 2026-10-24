//! `torch_peak_local_max`: a pixel is a seed when it is the value
//! `max_pool2d_with_indices` reports for its own (2d+1)^2 window and exceeds
//! the threshold.

/// Seeds as (row, col) in row-major order, like `torch.nonzero`.
pub(super) fn peak_local_max(
    map: &[f32],
    width: usize,
    height: usize,
    distance: usize,
    threshold: f32,
) -> Vec<(usize, usize)> {
    let mut seeds = Vec::new();
    for r in 0..height {
        let r0 = r.saturating_sub(distance);
        let r1 = (r + distance).min(height - 1);
        for c in 0..width {
            let v = map[r * width + c];
            if v.is_nan() || v <= threshold {
                continue;
            }
            let c0 = c.saturating_sub(distance);
            let c1 = (c + distance).min(width - 1);
            if is_first_max(map, width, (r0, r1, c0, c1), (r, c), v) {
                seeds.push((r, c));
            }
        }
    }
    seeds
}

/// PyTorch's CPU kernel scans the window row-major and only replaces the running
/// max on a strictly greater value, so the first maximum in scan order wins.
#[inline]
fn is_first_max(
    map: &[f32],
    width: usize,
    (r0, r1, c0, c1): (usize, usize, usize, usize),
    (r, c): (usize, usize),
    v: f32,
) -> bool {
    for qr in r0..=r1 {
        let row = &map[qr * width..qr * width + width];
        for (qc, &q) in row.iter().enumerate().take(c1 + 1).skip(c0) {
            if q > v || q.is_nan() {
                return false;
            }
            if q == v && (qr < r || (qr == r && qc < c)) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_isolated_peaks_above_threshold() {
        let (w, h) = (7, 5);
        let mut map = vec![0.0; w * h];
        map[w + 1] = 0.9;
        map[3 * w + 5] = 0.8;
        map[2 * w + 3] = 0.6; // below threshold
        assert_eq!(peak_local_max(&map, w, h, 1, 0.7), vec![(1, 1), (3, 5)]);
    }

    #[test]
    fn plateau_keeps_only_first_in_scan_order() {
        let (w, h) = (4, 3);
        let mut map = vec![0.0; w * h];
        map[w + 1] = 0.9;
        map[w + 2] = 0.9;
        map[2 * w + 1] = 0.9;
        assert_eq!(peak_local_max(&map, w, h, 1, 0.5), vec![(1, 1)]);
    }

    #[test]
    fn suppressed_by_larger_neighbour_within_distance() {
        let (w, h) = (9, 1);
        let mut map = vec![0.0; w];
        map[2] = 0.8;
        map[6] = 0.9;
        assert_eq!(peak_local_max(&map, w, h, 4, 0.5), vec![(0, 6)]);
        assert_eq!(peak_local_max(&map, w, h, 3, 0.5), vec![(0, 2), (0, 6)]);
    }
}
