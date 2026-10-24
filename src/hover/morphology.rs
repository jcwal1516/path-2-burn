//! Binary morphology and labelling matching SciPy / OpenCV defaults.

/// 4-neighbour offsets (up, left, right, down) of `idx` inside a `width x height` grid.
pub(crate) fn neighbors4(idx: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (idx % width, idx / width);
    [
        (y > 0).then(|| idx - width),
        (x > 0).then(|| idx - 1),
        (x + 1 < width).then(|| idx + 1),
        (y + 1 < height).then(|| idx + width),
    ]
    .into_iter()
    .flatten()
}

/// `scipy.ndimage.label` with the default cross structure. Labels are numbered
/// by the raster position of each component's first pixel. Returns the label
/// image and the number of labels.
pub(crate) fn label4(fg: &[bool], width: usize, height: usize) -> (Vec<u32>, u32) {
    let mut labels = vec![0u32; fg.len()];
    let mut next = 0u32;
    let mut stack = Vec::new();
    for start in 0..fg.len() {
        if !fg[start] || labels[start] != 0 {
            continue;
        }
        next += 1;
        labels[start] = next;
        stack.push(start);
        while let Some(idx) = stack.pop() {
            for n in neighbors4(idx, width, height) {
                if fg[n] && labels[n] == 0 {
                    labels[n] = next;
                    stack.push(n);
                }
            }
        }
    }
    (labels, next)
}

/// CellViT's `remove_small_objects` on a label image: zero every label whose
/// pixel count is below `min_size` (labels are not renumbered).
pub(crate) fn remove_small_labels(labels: &mut [u32], label_count: u32, min_size: usize) {
    let mut sizes = vec![0usize; label_count as usize + 1];
    for &l in labels.iter() {
        sizes[l as usize] += 1;
    }
    for l in labels.iter_mut() {
        if *l != 0 && sizes[*l as usize] < min_size {
            *l = 0;
        }
    }
}

/// `scipy.ndimage.binary_fill_holes`: background not 4-connected to the image
/// border becomes foreground.
pub(crate) fn fill_holes(fg: &[bool], width: usize, height: usize) -> Vec<bool> {
    let mut outside = vec![false; fg.len()];
    let mut stack: Vec<usize> = (0..fg.len())
        .filter(|&i| {
            let (x, y) = (i % width, i / width);
            !fg[i] && (x == 0 || y == 0 || x + 1 == width || y + 1 == height)
        })
        .collect();
    for &i in &stack {
        outside[i] = true;
    }
    while let Some(idx) = stack.pop() {
        for n in neighbors4(idx, width, height) {
            if !fg[n] && !outside[n] {
                outside[n] = true;
                stack.push(n);
            }
        }
    }
    outside.into_iter().map(|o| !o).collect()
}

/// AND (erode) / OR (dilate) over a `2 * radius + 1` window along rows or
/// columns. Out-of-image taps are skipped: OpenCV's default morphology border
/// never erodes or dilates from outside the image.
fn window_pass(
    src: &[bool],
    width: usize,
    height: usize,
    radius: usize,
    horizontal: bool,
    erode: bool,
) -> Vec<bool> {
    let combine = |a: bool, b: bool| if erode { a && b } else { a || b };
    let mut out = vec![false; src.len()];
    if horizontal {
        for (row, dst) in src.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
            for (x, d) in dst.iter_mut().enumerate() {
                let window = &row[x.saturating_sub(radius)..(x + radius + 1).min(width)];
                *d = window.iter().copied().reduce(combine).unwrap_or(false);
            }
        }
    } else {
        for (y, dst) in out.chunks_exact_mut(width).enumerate() {
            let (lo, hi) = (y.saturating_sub(radius), (y + radius + 1).min(height));
            dst.copy_from_slice(&src[lo * width..(lo + 1) * width]);
            for row in src[(lo + 1) * width..hi * width].chunks_exact(width) {
                dst.iter_mut()
                    .zip(row)
                    .for_each(|(d, &v)| *d = combine(*d, v));
            }
        }
    }
    out
}

/// Erosion/dilation with `cv2.getStructuringElement(MORPH_ELLIPSE, (5, 5))`,
/// decomposed as the union of a 5x3 rectangle and a 1x5 vertical bar.
fn morph_ellipse5(src: &[bool], width: usize, height: usize, erode: bool) -> Vec<bool> {
    let rect = window_pass(
        &window_pass(src, width, height, 2, true, erode),
        width,
        height,
        1,
        false,
        erode,
    );
    let bar = window_pass(src, width, height, 2, false, erode);
    rect.iter()
        .zip(&bar)
        .map(|(&r, &b)| if erode { r && b } else { r || b })
        .collect()
}

/// `cv2.morphologyEx(src, MORPH_OPEN, ellipse 5x5)`.
pub(crate) fn open_ellipse5(src: &[bool], width: usize, height: usize) -> Vec<bool> {
    let eroded = morph_ellipse5(src, width, height, true);
    morph_ellipse5(&eroded, width, height, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> (Vec<bool>, usize, usize) {
        let w = rows[0].len();
        let v = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| b == b'1'))
            .collect();
        (v, w, rows.len())
    }

    fn rows(v: &[bool], w: usize) -> Vec<String> {
        v.chunks(w)
            .map(|r| r.iter().map(|&b| if b { '1' } else { '0' }).collect())
            .collect()
    }

    #[test]
    fn label_matches_scipy_numbering() {
        // scipy.ndimage.label(grid)[0]
        let (fg, w, h) = grid(&["0110010", "0100110", "1001000", "1100011"]);
        let (labels, n) = label4(&fg, w, h);
        assert_eq!(n, 5);
        let expected = [
            0, 1, 1, 0, 0, 2, 0, //
            0, 1, 0, 0, 2, 2, 0, //
            3, 0, 0, 4, 0, 0, 0, //
            3, 3, 0, 0, 0, 5, 5,
        ];
        assert_eq!(labels, expected);
    }

    #[test]
    fn remove_small_labels_keeps_ids() {
        let mut labels = vec![1, 1, 0, 2, 3, 3, 3];
        remove_small_labels(&mut labels, 3, 2);
        assert_eq!(labels, [1, 1, 0, 0, 3, 3, 3]);
    }

    #[test]
    fn fill_holes_matches_scipy() {
        // scipy.ndimage.binary_fill_holes: diagonal gaps do not leak (4-connectivity)
        let (fg, w, h) = grid(&["011100", "100010", "100010", "011100", "000000"]);
        assert_eq!(
            rows(&fill_holes(&fg, w, h), w),
            ["011100", "111110", "111110", "011100", "000000"]
        );
        let (fg, w, h) = grid(&["0111110", "0100010", "0101010", "0100010", "0111110"]);
        assert_eq!(rows(&fill_holes(&fg, w, h), w), ["0111110"; 5]);
    }

    #[test]
    fn ellipse_dilate_and_erode_match_opencv() {
        // cv2.dilate / cv2.erode with getStructuringElement(MORPH_ELLIPSE, (5, 5))
        let (fg, w, h) = grid(&[
            "0000001", "0000000", "0000000", "0001000", "0000000", "0000000", "0000000",
        ]);
        let expected = [
            "0000111", "0001111", "0111111", "0111110", "0111110", "0001000", "0000000",
        ];
        assert_eq!(rows(&morph_ellipse5(&fg, w, h, false), w), expected);
        let (fg, w, h) = grid(&["111111", "111111", "110111", "111111", "111111", "111111"]);
        let expected = ["110111", "000001", "000001", "000001", "110111", "111111"];
        assert_eq!(rows(&morph_ellipse5(&fg, w, h, true), w), expected);
    }

    #[test]
    fn open_matches_opencv() {
        // cv2.morphologyEx(img, cv2.MORPH_OPEN, ellipse5)
        let (fg, w, h) = grid(&[
            "1111100", "1111100", "1111100", "1111000", "0000000", "0000001",
        ]);
        let expected = [
            "1111100", "1111100", "1111100", "1110000", "0000000", "0000000",
        ];
        assert_eq!(rows(&open_ellipse5(&fg, w, h), w), expected);
        let (fg, w, h) = grid(&[
            "0111110", "1111111", "1111111", "1111111", "1111111", "0111110", "0011100",
        ]);
        let expected = [
            "0111110", "1111111", "1111111", "1111111", "1111111", "0111110", "0001000",
        ];
        assert_eq!(rows(&open_ellipse5(&fg, w, h), w), expected);
    }
}
