//! Marker-based watershed matching `skimage.segmentation.watershed` with its
//! defaults (connectivity 1, no compactness, no watershed lines).

use super::morphology::neighbors4;

/// Heap entry: priority value plus `age << 32 | pixel index`.
#[derive(Clone, Copy)]
struct Item {
    value: f64,
    key: u64,
}

impl Item {
    fn new(value: f64, age: u64, index: usize) -> Self {
        Self {
            value,
            key: age << 32 | index as u64,
        }
    }

    fn age(&self) -> u64 {
        self.key >> 32
    }

    fn index(&self) -> usize {
        (self.key & u64::from(u32::MAX)) as usize
    }
}

/// skimage's heap order: lower value first, then lower insertion age. Markers
/// all have age 0, so their relative order depends on the exact heap layout,
/// which is why `Heap` mirrors skimage's sift-up/sift-down step for step.
fn smaller(a: &Item, b: &Item) -> bool {
    if a.value != b.value {
        a.value < b.value
    } else {
        a.age() < b.age()
    }
}

struct Heap(Vec<Item>);

impl Heap {
    fn push(&mut self, item: Item) {
        let heap = &mut self.0;
        let mut child = heap.len();
        heap.push(item);
        while child > 0 {
            let parent = (child - 1) / 2;
            if !smaller(&item, &heap[parent]) {
                break;
            }
            heap[child] = heap[parent];
            child = parent;
        }
        heap[child] = item;
    }

    fn pop(&mut self) -> Option<Item> {
        let heap = &mut self.0;
        let last = heap.pop()?;
        if heap.is_empty() {
            return Some(last);
        }
        let top = heap[0];
        let len = heap.len();
        let mut i = 0;
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            if l >= len {
                break;
            }
            let mut smallest = if smaller(&heap[l], &last) { l } else { i };
            let current = if smallest == l { heap[l] } else { last };
            if r < len && smaller(&heap[r], &current) {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            heap[i] = heap[smallest];
            i = smallest;
        }
        heap[i] = last;
        Some(top)
    }
}

/// Flood `markers` over `image` inside `mask`. Markers outside the mask are
/// dropped first (skimage multiplies markers by the mask); pixels are labelled
/// when pushed, as in skimage's fast path. A pushed pixel's priority is the
/// flood level `max(value, parent priority)`: scikit-image 0.26 behaves this
/// way (verified against it on tie-heavy random surfaces), and it differs from
/// a raw-value queue whenever the flood descends into a lower pixel.
pub(crate) fn watershed(
    image: &[f64],
    markers: &[u32],
    mask: &[bool],
    width: usize,
    height: usize,
) -> Vec<u32> {
    let mut output: Vec<u32> = markers
        .iter()
        .zip(mask)
        .map(|(&m, &inside)| if inside { m } else { 0 })
        .collect();
    let mut heap = Heap(Vec::new());
    for (index, _) in output.iter().enumerate().filter(|(_, l)| **l != 0) {
        heap.push(Item::new(image[index], 0, index));
    }
    let mut age = 1u64;
    while let Some(elem) = heap.pop() {
        let label = output[elem.index()];
        for n in neighbors4(elem.index(), width, height) {
            if !mask[n] || output[n] != 0 {
                continue;
            }
            age += 1;
            output[n] = label;
            let level = if image[n] < elem.value {
                elem.value
            } else {
                image[n]
            };
            heap.push(Item::new(level, age, n));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(image: &[f64], markers: &[u32], mask: &[u8], width: usize) -> Vec<u32> {
        let mask: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
        watershed(image, markers, &mask, width, image.len() / width)
    }

    #[test]
    fn flood_level_priority_matches_skimage() {
        // skimage.segmentation.watershed(img, markers): a raw-value queue would
        // give the bottom-left pixel to marker 2.
        let image = [-0.5, -1.0, 0.0, -0.5, 0.0, -1.0];
        let got = run(&image, &[0, 0, 2, 0, 0, 1], &[1; 6], 3);
        assert_eq!(got, [2, 2, 2, 1, 1, 1]);
    }

    #[test]
    fn plateau_ties_and_mask_match_skimage() {
        // watershed(zeros((3, 5)), markers, mask=mask)
        let markers = [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4];
        let mask = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1];
        let got = run(&[0.0; 15], &markers, &mask, 5);
        assert_eq!(got, [1, 1, 1, 2, 2, 1, 1, 0, 4, 4, 0, 1, 4, 4, 4]);
    }

    #[test]
    fn equal_valued_markers_follow_heap_layout() {
        // watershed([[1,2,1,1],[1,1,2,1],[2,1,1,1]], markers=[[5,0,0,7],[0,0,0,0],[0,0,9,9]])
        let image = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0];
        let markers = [5, 0, 0, 7, 0, 0, 0, 0, 0, 0, 9, 9];
        let got = run(&image, &markers, &[1; 12], 4);
        assert_eq!(got, [5, 5, 7, 7, 5, 5, 9, 9, 5, 9, 9, 9]);
    }
}
