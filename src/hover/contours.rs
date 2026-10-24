//! First contour of `cv2.findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE)`.
//!
//! OpenCV pads the image with a zero border, scans in raster order and traces
//! borders with Suzuki–Abe following. In `RETR_TREE` the output is a pre-order
//! walk of the hierarchy in which siblings appear in reverse discovery order,
//! so the first contour is the outer border of the *last* top-level component
//! (8-connected) found by the raster scan.

use super::morphology::neighbors4;

/// Freeman directions in OpenCV's chain-code order (y grows downwards).
const DX: [isize; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY: [isize; 8] = [0, -1, -1, -1, 0, 1, 1, 1];

/// Error raised when border following fails to close (never expected for a
/// well-formed binary mask; guards against looping forever).
#[derive(Debug)]
pub(crate) struct UnclosedContour;

/// Returns the first contour (crop coordinates, x/y) of the binary `mask`, or
/// an empty vector when the mask has no foreground.
pub(crate) fn first_contour(
    mask: &[bool],
    width: usize,
    height: usize,
) -> Result<Vec<[i32; 2]>, UnclosedContour> {
    let (pw, ph) = (width + 2, height + 2);
    let mut img = vec![false; pw * ph];
    for (y, row) in mask.chunks_exact(width).enumerate() {
        img[(y + 1) * pw + 1..(y + 1) * pw + 1 + width].copy_from_slice(row);
    }
    let Some(start) = last_top_level_start(&img, pw, ph) else {
        return Ok(Vec::new());
    };
    let points = trace_outer(&img, pw, start)?;
    Ok(points.into_iter().map(|[x, y]| [x - 1, y - 1]).collect())
}

/// Start pixel (raster-first pixel) of the last 8-connected component whose
/// surrounding background is the region connected to the image border.
fn last_top_level_start(img: &[bool], pw: usize, ph: usize) -> Option<usize> {
    let mut outside = vec![false; img.len()];
    let mut stack = vec![0usize];
    outside[0] = true;
    while let Some(idx) = stack.pop() {
        for n in neighbors4(idx, pw, ph) {
            if !img[n] && !outside[n] {
                outside[n] = true;
                stack.push(n);
            }
        }
    }
    let mut seen = vec![false; img.len()];
    let mut chosen = None;
    for start in 0..img.len() {
        if !img[start] || seen[start] {
            continue;
        }
        // The left neighbour of a component's raster-first pixel is background
        // belonging to the region that encloses the component.
        if outside[start - 1] {
            chosen = Some(start);
        }
        seen[start] = true;
        stack.push(start);
        while let Some(idx) = stack.pop() {
            for d in 0..8 {
                let n = offset(idx, pw, d);
                if img[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
    }
    chosen
}

/// Neighbour of `idx` in chain direction `d`; callers only pass foreground
/// pixels of the zero-padded image, so the neighbour is always in bounds.
fn offset(idx: usize, pw: usize, d: usize) -> usize {
    (idx as isize + DX[d] + DY[d] * pw as isize) as usize
}

/// OpenCV `icvFetchContour` for an outer border with `CHAIN_APPROX_SIMPLE`:
/// a point is emitted whenever the outgoing chain direction changes, with the
/// start pixel compared against the closing step into it.
fn trace_outer(img: &[bool], pw: usize, i0: usize) -> Result<Vec<[i32; 2]>, UnclosedContour> {
    let point = |idx: usize| [(idx % pw) as i32, (idx / pw) as i32];
    let first = (0..8)
        .map(|k| (11 - k) & 7)
        .find(|&d| img[offset(i0, pw, d)]);
    // Search runs clockwise from the left neighbour, which is background for an
    // outer border, so `None` means a single-pixel component.
    let Some(mut s) = first.filter(|&d| d != 4) else {
        return Ok(vec![point(i0)]);
    };
    let i1 = offset(i0, pw, s);
    let mut prev_s = s ^ 4;
    let mut i3 = i0;
    let mut out = Vec::new();
    // Each border pixel is left at most 4 times (once per side).
    for _ in 0..4 * img.len() {
        let next = (1..=8)
            .map(|k| (s + k) & 7)
            .find(|&d| img[offset(i3, pw, d)]);
        let Some(dir) = next else {
            return Err(UnclosedContour);
        };
        s = dir;
        let i4 = offset(i3, pw, s);
        if s != prev_s {
            out.push(point(i3));
            prev_s = s;
        }
        if i4 == i0 && i3 == i1 {
            return Ok(out);
        }
        i3 = i4;
        s = (s + 4) & 7;
    }
    Err(UnclosedContour)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(rows: &[&str]) -> Vec<[i32; 2]> {
        let w = rows[0].len();
        let mask: Vec<bool> = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| b == b'1'))
            .collect();
        first_contour(&mask, w, rows.len()).expect("contour closes")
    }

    // Expected values: cv2.findContours(m, RETR_TREE, CHAIN_APPROX_SIMPLE)[0][0]

    #[test]
    fn simple_shapes() {
        assert_eq!(
            first(&["1111", "0001", "1111"]),
            [[0, 0], [2, 0], [3, 1], [2, 2], [0, 2], [3, 2], [3, 0]]
        );
        assert_eq!(
            first(&["0110", "1111", "1111", "0110"]),
            [
                [1, 0],
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [3, 2],
                [3, 1],
                [2, 0]
            ]
        );
        assert_eq!(first(&["1"]), [[0, 0]]);
        assert_eq!(first(&["1111"]), [[0, 0], [3, 0]]);
        assert_eq!(first(&["100", "010", "001"]), [[0, 0], [2, 2]]);
        assert!(first(&["000"]).is_empty());
    }

    #[test]
    fn last_top_level_component_comes_first() {
        let rows = [
            "1100000000",
            "1100000000",
            "0000000000",
            "0000100000",
            "0000000000",
            "0000001111",
            "0000001111",
            "0000001111",
        ];
        assert_eq!(first(&rows), [[6, 5], [6, 7], [9, 7], [9, 5]]);
    }

    #[test]
    fn components_inside_holes_are_not_top_level() {
        let ring = [
            "111111100",
            "100000100",
            "100000100",
            "100100100",
            "100000100",
            "100000100",
            "111111100",
            "000000000",
            "000000000",
        ];
        assert_eq!(first(&ring), [[0, 0], [0, 6], [6, 6], [6, 0]]);
        let mut with_dot = ring;
        with_dot[0] = "111111101";
        assert_eq!(first(&with_dot), [[8, 0]]);
    }
}
