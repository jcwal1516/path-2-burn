//! HoVer-Net's watershed decoding of nucleus probability, horizontal/vertical
//! distance and type maps into nucleus instances, shared by HoVer-Net and
//! CellViT.
//!
//! Pure-Rust port of CellViT's `DetectionCellPostProcessor(nr_types=6,
//! magnification=40)` (`export/cellvit_ref/postprocess.py`), whose decoding at
//! 40x is HoVer-Net's `process` line for line. It reproduces the OpenCV / SciPy
//! / scikit-image defaults it relies on so results match the Python reference
//! instance-for-instance.

mod contours;
mod filters;
mod morphology;
mod watershed;
pub(crate) use morphology::{fill_holes, remove_small_labels};

use crate::runtime::read_f32;
use crate::store::ModelError;
use burn::prelude::Tensor;
use burn::tensor::activation::softmax;
use filters::Axis;

/// Minimum marker size at 40x (`object_size`).
const OBJECT_SIZE: usize = 10;
/// Sobel kernel size at 40x (`k_size`).
const SOBEL_KSIZE: usize = 21;
/// Minimum size of a foreground blob before marker extraction.
const MIN_BLOB_SIZE: usize = 10;
/// Gradient-energy threshold separating marker cores from boundaries.
const ENERGY_THRESHOLD: f64 = 0.4;

/// Per-pixel network outputs for one `width x height` tile (row-major).
#[derive(Clone, Copy, Debug)]
pub struct HvMaps<'a> {
    pub width: usize,
    pub height: usize,
    /// Softmax foreground probability, `[H, W]`.
    pub nuclei_prob: &'a [f32],
    /// Horizontal then vertical distance maps, `[2, H, W]`.
    pub hv: &'a [f32],
    /// Arg-max nuclei type, `[H, W]`.
    pub type_pred: &'a [u8],
}

/// The maps [`decode`] needs, computed on the device from one tile's network
/// outputs so that only these come back to the CPU.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceMaps {
    pub width: usize,
    pub height: usize,
    /// P(nucleus): the softmax of the two-class nucleus logits, `[H, W]`.
    pub nuclei_prob: Vec<f32>,
    /// Horizontal then vertical distance maps, `[2, H, W]`.
    pub hv: Vec<f32>,
    /// Arg-max nucleus type, `[H, W]`; all 0 for networks without types.
    pub type_pred: Vec<u8>,
}

impl DeviceMaps {
    pub fn maps(&self) -> HvMaps<'_> {
        HvMaps {
            width: self.width,
            height: self.height,
            nuclei_prob: &self.nuclei_prob,
            hv: &self.hv,
            type_pred: &self.type_pred,
        }
    }
}

/// [`DeviceMaps`] from one tile's `[1, 2, H, W]` nucleus logits, `[1, 2, H, W]`
/// HV maps and, if the network classifies, `[1, types, H, W]` type logits.
pub(crate) fn device_maps(
    binary: Tensor<4>,
    hv: Tensor<4>,
    types: Option<Tensor<4>>,
    model: &'static str,
) -> Result<DeviceMaps, ModelError> {
    let [_, _, height, width] = binary.dims();
    let nuclei_prob = softmax(binary, 1).slice([0..1, 1..2, 0..height, 0..width]);
    let type_pred = match types {
        Some(types) => read_f32(types.argmax(1).float(), model)?
            .into_iter()
            .map(|t| t as u8)
            .collect(),
        None => vec![0; width * height],
    };
    Ok(DeviceMaps {
        width,
        height,
        nuclei_prob: read_f32(nuclei_prob, model)?,
        hv: read_f32(hv, model)?,
        type_pred,
    })
}

/// One decoded nucleus.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedCell {
    /// Instance id, as stored in [`Decoded::instance_map`].
    pub id: u32,
    /// Centroid `[x, y]` in pixels.
    pub centroid: [f64; 2],
    /// Bounding box `[x, y, w, h]` in pixels.
    pub bbox: [u32; 4],
    /// Outer contour `[x, y]` points (OpenCV `CHAIN_APPROX_SIMPLE`).
    pub contour: Vec<[i32; 2]>,
    pub cell_type: u8,
    /// Fraction of instance pixels voting for `cell_type`.
    pub type_prob: f32,
}

/// Watershed instance map (`[H, W]`, 0 = background) plus per-cell metadata
/// in ascending id order. Instances whose contour has fewer than 3 points stay
/// in the map but get no [`DecodedCell`], as in the Python reference.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub instance_map: Vec<u32>,
    pub cells: Vec<DecodedCell>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PostprocessError {
    #[error("image dimensions must be non-zero, got {width}x{height}")]
    EmptyImage { width: usize, height: usize },
    #[error("image dimensions {width}x{height} overflow usize")]
    DimensionOverflow { width: usize, height: usize },
    #[error("{name} has {actual} elements, expected {expected}")]
    LengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("class count must be in 1..=256, got {0}")]
    InvalidClassCount(usize),
    #[error("contour tracing did not close for instance {0}")]
    UnclosedContour(u32),
}

/// Pixel count of a `width x height` plane times `channels`, validated.
fn plane_len(channels: usize, width: usize, height: usize) -> Result<usize, PostprocessError> {
    if width == 0 || height == 0 {
        return Err(PostprocessError::EmptyImage { width, height });
    }
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(PostprocessError::DimensionOverflow { width, height })
}

fn check_len(name: &'static str, actual: usize, expected: usize) -> Result<(), PostprocessError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PostprocessError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
}

/// Foreground probability `softmax(binary_logits, axis=0)[1]` from `[2, H, W]` logits.
pub fn nuclei_prob_from_logits(
    binary_logits: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, PostprocessError> {
    check_len(
        "binary_logits",
        binary_logits.len(),
        plane_len(2, width, height)?,
    )?;
    let (bg, fg) = binary_logits.split_at(binary_logits.len() / 2);
    Ok(bg
        .iter()
        .zip(fg)
        .map(|(&b, &f)| {
            let m = if f > b { f } else { b };
            let (eb, ef) = ((b - m).exp(), (f - m).exp());
            ef / (eb + ef)
        })
        .collect())
}

/// Per-pixel arg-max over the `classes` channels of `[C, H, W]` type logits.
/// The first maximum wins ties and NaN wins, like `np.argmax`.
pub fn type_argmax(
    type_logits: &[f32],
    classes: usize,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, PostprocessError> {
    if !(1..=256).contains(&classes) {
        return Err(PostprocessError::InvalidClassCount(classes));
    }
    check_len(
        "type_logits",
        type_logits.len(),
        plane_len(classes, width, height)?,
    )?;
    let n = type_logits.len() / classes;
    let mut best = type_logits[..n].to_vec();
    let mut arg = vec![0u8; n];
    for (c, plane) in type_logits.chunks_exact(n).enumerate().skip(1) {
        for ((b, a), &v) in best.iter_mut().zip(arg.iter_mut()).zip(plane) {
            if v > *b || (v.is_nan() && !b.is_nan()) {
                *b = v;
                *a = c as u8;
            }
        }
    }
    Ok(arg)
}

/// Runs the HoVer-Net watershed decoding and per-instance extraction.
pub fn decode(maps: &HvMaps<'_>) -> Result<Decoded, PostprocessError> {
    let (width, height) = (maps.width, maps.height);
    let n = plane_len(1, width, height)?;
    // The watershed packs pixel index and queue age into 32 bits each.
    if n >= u32::MAX as usize {
        return Err(PostprocessError::DimensionOverflow { width, height });
    }
    check_len("nuclei_prob", maps.nuclei_prob.len(), n)?;
    check_len("hv", maps.hv.len(), plane_len(2, width, height)?)?;
    check_len("type_pred", maps.type_pred.len(), n)?;

    let blob = foreground_mask(maps.nuclei_prob, width, height);
    let (h_map, v_map) = maps.hv.split_at(n);
    let energy = gradient_energy(h_map, v_map, &blob, width, height);
    let markers = markers(&energy, &blob, width, height);
    let surface = watershed_surface(&energy, &blob, width, height);
    let instance_map = watershed::watershed(&surface, &markers, &blob, width, height);
    let cells = extract_cells(&instance_map, maps.type_pred, width)?;
    Ok(Decoded {
        instance_map,
        cells,
    })
}

/// `prob >= 0.5`, minus 4-connected blobs smaller than [`MIN_BLOB_SIZE`].
fn foreground_mask(prob: &[f32], width: usize, height: usize) -> Vec<bool> {
    let raw: Vec<bool> = prob.iter().map(|&p| p >= 0.5).collect();
    let (mut labels, count) = morphology::label4(&raw, width, height);
    morphology::remove_small_labels(&mut labels, count, MIN_BLOB_SIZE);
    labels.into_iter().map(|l| l != 0).collect()
}

/// `overall` in the reference: `max(1 - norm(sobel_h), 1 - norm(sobel_v))`
/// inside the foreground, 0 outside. NumPy promotes it to f64 when the int
/// mask is subtracted, so the `>= 0.4` test must run in f64.
fn gradient_energy(
    h_map: &[f32],
    v_map: &[f32],
    blob: &[bool],
    width: usize,
    height: usize,
) -> Vec<f64> {
    let edge = |map: &[f32], axis| -> Vec<f32> {
        let dir = filters::normalize_minmax_f32(map);
        let sobel = filters::sobel(&dir, width, height, axis, SOBEL_KSIZE);
        let norm = filters::normalize_minmax_f64_to_f32(&sobel);
        norm.into_iter().map(|v| 1.0 - v).collect()
    };
    let (sh, sv) = (edge(h_map, Axis::X), edge(v_map, Axis::Y));
    sh.iter()
        .zip(&sv)
        .zip(blob)
        .map(|((&a, &b), &fg)| {
            let m = if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                a.max(b)
            };
            let e = f64::from(m) - if fg { 0.0 } else { 1.0 };
            if e < 0.0 { 0.0 } else { e }
        })
        .collect()
}

/// `-GaussianBlur((1 - overall) * blb, (3, 3), 0)` in f64.
fn watershed_surface(energy: &[f64], blob: &[bool], width: usize, height: usize) -> Vec<f64> {
    let dist: Vec<f64> = energy
        .iter()
        .zip(blob)
        .map(|(&e, &fg)| (1.0 - e) * if fg { 1.0 } else { 0.0 })
        .collect();
    filters::gaussian_blur3(&dist, width, height)
        .into_iter()
        .map(|v| -v)
        .collect()
}

/// Marker labels: low-energy foreground cores, hole-filled, opened with a 5x5
/// ellipse, labelled and stripped of markers smaller than [`OBJECT_SIZE`].
fn markers(energy: &[f64], blob: &[bool], width: usize, height: usize) -> Vec<u32> {
    let core: Vec<bool> = energy
        .iter()
        .zip(blob)
        .map(|(&e, &fg)| {
            let boundary = e >= ENERGY_THRESHOLD;
            fg && !boundary
        })
        .collect();
    let filled = morphology::fill_holes(&core, width, height);
    let opened = morphology::open_ellipse5(&filled, width, height);
    let (mut labels, count) = morphology::label4(&opened, width, height);
    morphology::remove_small_labels(&mut labels, count, OBJECT_SIZE);
    labels
}

/// Bounding box `[rmin, rmax) x [cmin, cmax)` and pixel count of one instance.
#[derive(Clone, Copy)]
struct Extent {
    rmin: usize,
    rmax: usize,
    cmin: usize,
    cmax: usize,
    area: usize,
}

impl Extent {
    fn width(&self) -> usize {
        self.cmax - self.cmin
    }

    fn height(&self) -> usize {
        self.rmax - self.rmin
    }
}

fn instance_extents(instance_map: &[u32], width: usize) -> Vec<Option<Extent>> {
    let max_id = instance_map.iter().copied().max().unwrap_or(0) as usize;
    let mut extents: Vec<Option<Extent>> = vec![None; max_id + 1];
    for (idx, &id) in instance_map.iter().enumerate().filter(|(_, id)| **id != 0) {
        let (x, y) = (idx % width, idx / width);
        let e = extents[id as usize].get_or_insert(Extent {
            rmin: y,
            rmax: y + 1,
            cmin: x,
            cmax: x + 1,
            area: 0,
        });
        e.rmax = y + 1;
        e.cmin = e.cmin.min(x);
        e.cmax = e.cmax.max(x + 1);
        e.area += 1;
    }
    extents
}

/// Per-instance bbox, centroid, first contour and type, in ascending id order.
fn extract_cells(
    instance_map: &[u32],
    type_pred: &[u8],
    width: usize,
) -> Result<Vec<DecodedCell>, PostprocessError> {
    let mut cells = Vec::new();
    for (id, extent) in instance_extents(instance_map, width)
        .into_iter()
        .enumerate()
    {
        let Some(e) = extent else { continue };
        let id = id as u32;
        let crop: Vec<bool> = (e.rmin..e.rmax)
            .flat_map(|y| {
                instance_map[y * width + e.cmin..y * width + e.cmax]
                    .iter()
                    .map(move |&v| v == id)
            })
            .collect();
        let contour = contours::first_contour(&crop, e.width(), e.height())
            .map_err(|_| PostprocessError::UnclosedContour(id))?;
        if contour.len() < 3 {
            continue;
        }
        let (cell_type, type_prob) = vote_type(&crop, type_pred, width, &e);
        let origin = [e.cmin as i32, e.rmin as i32];
        cells.push(DecodedCell {
            id,
            centroid: centroid(&crop, &e),
            bbox: [
                e.cmin as u32,
                e.rmin as u32,
                e.width() as u32,
                e.height() as u32,
            ],
            contour: contour
                .into_iter()
                .map(|[x, y]| [x + origin[0], y + origin[1]])
                .collect(),
            cell_type,
            type_prob,
        });
    }
    Ok(cells)
}

/// Pixels of the crop as image coordinates `(x, y)`.
fn crop_pixels<'a>(crop: &'a [bool], e: &'a Extent) -> impl Iterator<Item = (usize, usize)> + 'a {
    let cw = e.width();
    crop.iter()
        .enumerate()
        .filter(|(_, v)| **v)
        .map(move |(i, _)| (e.cmin + i % cw, e.rmin + i / cw))
}

/// `cv2.moments` centroid of the crop (exact integer sums in crop coordinates)
/// offset by the bbox origin afterwards, exactly as the reference does.
fn centroid(crop: &[bool], e: &Extent) -> [f64; 2] {
    let (m10, m01) = crop_pixels(crop, e).fold((0u64, 0u64), |(sx, sy), (x, y)| {
        (sx + (x - e.cmin) as u64, sy + (y - e.rmin) as u64)
    });
    let m00 = e.area as f64;
    [
        m10 as f64 / m00 + e.cmin as f64,
        m01 as f64 / m00 + e.rmin as f64,
    ]
}

/// Majority type over the instance pixels; background (0) only wins when it
/// is the sole type. Ties go to the lower id (`np.unique` + stable sort).
fn vote_type(crop: &[bool], type_pred: &[u8], width: usize, e: &Extent) -> (u8, f32) {
    let mut counts = [0usize; 256];
    for (x, y) in crop_pixels(crop, e) {
        counts[type_pred[y * width + x] as usize] += 1;
    }
    let best = |skip_zero: bool| {
        (0..counts.len())
            .filter(|&t| counts[t] > 0 && !(skip_zero && t == 0))
            .reduce(|b, t| if counts[t] > counts[b] { t } else { b })
    };
    let chosen = match best(false) {
        Some(0) => best(true).unwrap_or(0),
        other => other.unwrap_or(0),
    };
    let prob = counts[chosen] as f64 / (e.area as f64 + 1e-6);
    (chosen as u8, prob as f32)
}
