//! Window probabilities to one label map: `convert` (highest probability wins
//! per pixel), intersection over minimum area between windows, and the scripted
//! `find_connected_components` label propagation.

use super::crops::Crop;

/// The scripted `find_connected_components` stops after this many sweeps.
const MAX_PROPAGATION_SWEEPS: usize = 100;

pub(super) fn merge(
    crops: &[Crop],
    side: usize,
    width: usize,
    height: usize,
    mask_threshold: f32,
    overlap_threshold: f32,
) -> Vec<u32> {
    let owner = assign_pixels(crops, side, width, height, mask_threshold);
    let components = connected_components(
        crops.len(),
        &overlapping_pairs(crops, side, mask_threshold, overlap_threshold),
    );
    owner
        .into_iter()
        .map(|k| {
            if k == 0 {
                0
            } else {
                components[k as usize - 1]
            }
        })
        .collect()
}

/// Per pixel, the 1-based window with the highest probability >= threshold.
/// A stable descending sort keeps earlier windows first, so ties go to the lower index.
fn assign_pixels(
    crops: &[Crop],
    side: usize,
    width: usize,
    height: usize,
    threshold: f32,
) -> Vec<u32> {
    let mut best = vec![f32::NEG_INFINITY; width * height];
    let mut owner = vec![0u32; width * height];
    for (k, crop) in crops.iter().enumerate() {
        for i in 0..side {
            let row = (crop.top + i) * width + crop.left;
            for (j, &p) in crop.probs[i * side..(i + 1) * side].iter().enumerate() {
                if p >= threshold && p > best[row + j] {
                    best[row + j] = p;
                    owner[row + j] = k as u32 + 1;
                }
            }
        }
    }
    owner
}

/// Window pairs whose foreground intersection over the smaller foreground area
/// exceeds the threshold (float32 division, as the sparse matmul does it).
fn overlapping_pairs(
    crops: &[Crop],
    side: usize,
    threshold: f32,
    overlap_threshold: f32,
) -> Vec<(usize, usize)> {
    let area: Vec<u32> = crops
        .iter()
        .map(|c| c.probs.iter().filter(|&&p| p >= threshold).count() as u32)
        .collect();
    // Bucket windows on a `side` grid: overlapping windows sit in neighbouring cells.
    let mut buckets: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (k, c) in crops.iter().enumerate() {
        buckets
            .entry((c.top / side, c.left / side))
            .or_default()
            .push(k);
    }
    let mut pairs = Vec::new();
    for (a, ca) in crops.iter().enumerate() {
        if area[a] == 0 {
            continue;
        }
        let (br, bc) = (ca.top / side, ca.left / side);
        for nr in br.saturating_sub(1)..=br + 1 {
            for nc in bc.saturating_sub(1)..=bc + 1 {
                let Some(candidates) = buckets.get(&(nr, nc)) else {
                    continue;
                };
                for &b in candidates {
                    if b <= a || area[b] == 0 {
                        continue;
                    }
                    let inter = intersection(ca, &crops[b], side, threshold);
                    if inter > 0 && inter as f32 / area[a].min(area[b]) as f32 > overlap_threshold {
                        pairs.push((a, b));
                    }
                }
            }
        }
    }
    pairs
}

fn intersection(a: &Crop, b: &Crop, side: usize, threshold: f32) -> u32 {
    let r0 = a.top.max(b.top);
    let r1 = (a.top + side).min(b.top + side);
    let c0 = a.left.max(b.left);
    let c1 = (a.left + side).min(b.left + side);
    let mut count = 0;
    for r in r0..r1 {
        for c in c0..c1 {
            let pa = a.probs[(r - a.top) * side + (c - a.left)];
            let pb = b.probs[(r - b.top) * side + (c - b.left)];
            count += u32::from(pa >= threshold && pb >= threshold);
        }
    }
    count
}

/// Synchronous min-label propagation from labels 1..=n, as in the scripted model.
fn connected_components(n: usize, edges: &[(usize, usize)]) -> Vec<u32> {
    let mut labels: Vec<u32> = (1..=n as u32).collect();
    for _ in 0..MAX_PROPAGATION_SWEEPS {
        let mut next = labels.clone();
        for &(a, b) in edges {
            next[a] = next[a].min(labels[b]);
            next[b] = next[b].min(labels[a]);
        }
        if next == labels {
            break;
        }
        labels = next;
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crop(top: usize, left: usize, side: usize, on: &[(usize, usize)], p: f32) -> Crop {
        let mut probs = vec![0.0; side * side];
        for &(r, c) in on {
            probs[r * side + c] = p;
        }
        Crop { top, left, probs }
    }

    #[test]
    fn components_take_the_smallest_label() {
        assert_eq!(
            connected_components(5, &[(3, 4), (1, 3)]),
            vec![1, 2, 3, 2, 2]
        );
        assert_eq!(connected_components(3, &[]), vec![1, 2, 3]);
    }

    #[test]
    fn ties_go_to_the_earlier_window_and_overlaps_merge() {
        let side = 4;
        // Same image pixels (rows 1..3, col 1) in both windows: full overlap -> merged label 1.
        let a = crop(0, 0, side, &[(1, 1), (2, 1)], 0.8);
        let b = crop(0, 0, side, &[(1, 1), (2, 1)], 0.8);
        let c = crop(4, 4, side, &[(0, 0)], 0.9);
        let labels = merge(&[a, b, c], side, 8, 8, 0.53, 0.3);
        assert_eq!(labels[8 + 1], 1);
        assert_eq!(labels[2 * 8 + 1], 1);
        assert_eq!(labels[4 * 8 + 4], 3);
        assert_eq!(labels.iter().filter(|&&l| l != 0).count(), 3);
    }

    #[test]
    fn small_overlap_stays_separate() {
        let side = 4;
        let a = crop(0, 0, side, &[(0, 0), (0, 1), (0, 2), (0, 3)], 0.9);
        let b = crop(0, 3, side, &[(0, 0), (1, 0), (2, 0), (3, 0)], 0.6);
        // One shared pixel out of min area 4: 0.25 <= 0.3.
        let labels = merge(&[a, b], side, 8, 8, 0.53, 0.3);
        assert_eq!(labels[3], 1);
        assert_eq!(labels[8 + 3], 2);
    }
}
