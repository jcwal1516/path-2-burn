//! Parity of the Rust HoVer-Net decoding with CellViT's Python postprocessor.
//!
//! Fixture tiles come from `export/reference_cellvit.py`; each complete tile
//! (one with `cells.json`) is decoded from the raw torch outputs and compared
//! with the Python instance map and per-cell metadata.

mod support;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use path_2_burn::hover::{
    Decoded, DecodedCell, HvMaps, PostprocessError, decode, nuclei_prob_from_logits, type_argmax,
};
use support::{fixture_dirs, read_le};

const TILE: usize = 1024;

fn fixtures_dir() -> PathBuf {
    let dir = support::root().join("fixtures").join("cellvit");
    assert!(
        dir.is_dir(),
        "missing CellViT fixtures at {}; generate them with \
         `export/.venv/bin/python export/reference_cellvit.py --checkpoint CKPT --tiles fixtures/tiles --out fixtures/cellvit`",
        dir.display()
    );
    dir
}

/// Tile directories whose `cells.json` has been written (the last file the
/// exporter writes), sorted by name.
fn complete_tiles() -> Vec<PathBuf> {
    fixture_dirs(
        &fixtures_dir(),
        "cells.json",
        "run export/reference_cellvit.py",
    )
}

/// One Python cell; floats are parsed from the raw JSON text because
/// serde_json's default float parser is not always correctly rounded.
struct PyCell {
    id: u32,
    centroid: [f64; 2],
    bbox: [u32; 4],
    contour: Vec<[i32; 2]>,
    cell_type: u8,
    type_prob: f64,
}

/// Raw text of every value stored under `"key":`, in document order.
fn raw_values<'a>(text: &'a str, key: &str) -> Vec<&'a str> {
    let pattern = format!("\"{key}\":");
    text.match_indices(&pattern)
        .map(|(at, _)| {
            let rest = text[at + pattern.len()..].trim_start();
            let end = if rest.starts_with('[') {
                rest.find(']').map(|i| i + 1)
            } else {
                rest.find([',', '}'])
            };
            rest[..end.expect("terminated JSON value")].trim()
        })
        .collect()
}

fn parse_f64(raw: &str) -> f64 {
    raw.parse()
        .unwrap_or_else(|e| panic!("bad float {raw:?}: {e}"))
}

fn load_py_cells(tile: &Path) -> Vec<PyCell> {
    let text = fs::read_to_string(tile.join("cells.json")).expect("cells.json readable");
    let json: serde_json::Value = serde_json::from_str(&text).expect("cells.json is valid JSON");
    let cells = json.as_array().expect("cells.json is an array");
    let centroids = raw_values(&text, "centroid");
    let probs = raw_values(&text, "type_prob");
    assert_eq!(
        (centroids.len(), probs.len()),
        (cells.len(), cells.len()),
        "unexpected cells.json layout"
    );
    let int = |v: &serde_json::Value| v.as_i64().expect("integer field");
    cells
        .iter()
        .zip(centroids.iter().zip(&probs))
        .map(|(c, (centroid, prob))| {
            let xy: Vec<f64> = centroid
                .trim_matches(['[', ']'])
                .split(',')
                .map(|v| parse_f64(v.trim()))
                .collect();
            let b = &c["bbox"];
            let (rmin, cmin, rmax, cmax) =
                (int(&b[0][0]), int(&b[0][1]), int(&b[1][0]), int(&b[1][1]));
            PyCell {
                id: int(&c["id"]) as u32,
                centroid: [xy[0], xy[1]],
                bbox: [
                    cmin as u32,
                    rmin as u32,
                    (cmax - cmin) as u32,
                    (rmax - rmin) as u32,
                ],
                contour: c["contour"]
                    .as_array()
                    .expect("contour array")
                    .iter()
                    .map(|p| [int(&p[0]) as i32, int(&p[1]) as i32])
                    .collect(),
                cell_type: int(&c["type"]) as u8,
                type_prob: parse_f64(prob),
            }
        })
        .collect()
}

fn decode_tile(tile: &Path) -> (Decoded, Duration) {
    let binary = read_le(&tile.join("nuclei_binary_map.bin"), f32::from_le_bytes);
    let hv = read_le(&tile.join("hv_map.bin"), f32::from_le_bytes);
    let types = read_le(&tile.join("nuclei_type_map.bin"), f32::from_le_bytes);
    let classes = types.len() / (TILE * TILE);
    let nuclei_prob =
        nuclei_prob_from_logits(&binary, TILE, TILE).expect("binary map is [1,2,H,W]");
    let type_pred = type_argmax(&types, classes, TILE, TILE).expect("type map is [1,C,H,W]");
    let maps = HvMaps {
        width: TILE,
        height: TILE,
        nuclei_prob: &nuclei_prob,
        hv: &hv,
        type_pred: &type_pred,
    };
    let start = Instant::now();
    let decoded = decode(&maps).expect("decode succeeds on fixture tile");
    (decoded, start.elapsed())
}

/// Fraction of foreground pixels (in either map) whose labels correspond
/// one-to-one: each label's majority partner in the other map points back.
fn relabelled_pixel_agreement(py: &[i32], rs: &[u32]) -> f64 {
    let mut pairs: HashMap<(i32, u32), usize> = HashMap::new();
    for (&p, &r) in py.iter().zip(rs).filter(|(p, r)| **p != 0 || **r != 0) {
        *pairs.entry((p, r)).or_default() += 1;
    }
    let mut best_py: HashMap<i32, (u32, usize)> = HashMap::new();
    let mut best_rs: HashMap<u32, (i32, usize)> = HashMap::new();
    for (&(p, r), &n) in &pairs {
        let bp = best_py.entry(p).or_insert((r, 0));
        if n > bp.1 {
            *bp = (r, n);
        }
        let br = best_rs.entry(r).or_insert((p, 0));
        if n > br.1 {
            *br = (p, n);
        }
    }
    let total: usize = pairs.values().sum();
    let agree: usize = pairs
        .iter()
        .filter(|((p, r), _)| *p != 0 && *r != 0 && best_py[p].0 == *r && best_rs[r].0 == *p)
        .map(|(_, n)| n)
        .sum();
    if total == 0 {
        1.0
    } else {
        agree as f64 / total as f64
    }
}

#[derive(Default)]
struct TileReport {
    py_cells: usize,
    rs_cells: usize,
    matched: usize,
    pixel_agreement: f64,
    exact_ids: bool,
    type_agree: usize,
    max_centroid_err: f64,
    contour_exact: usize,
    bbox_exact: usize,
    type_prob_exact: usize,
}

fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 { 1.0 } else { n as f64 / d as f64 }
}

fn compare_tile(py: &[PyCell], py_map: &[i32], decoded: &Decoded) -> TileReport {
    let by_id: HashMap<u32, &DecodedCell> = decoded.cells.iter().map(|c| (c.id, c)).collect();
    let dist = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let mut report = TileReport {
        py_cells: py.len(),
        rs_cells: decoded.cells.len(),
        pixel_agreement: relabelled_pixel_agreement(py_map, &decoded.instance_map),
        exact_ids: py_map
            .iter()
            .zip(&decoded.instance_map)
            .all(|(&p, &r)| i64::from(p) == i64::from(r)),
        ..TileReport::default()
    };
    for cell in py {
        let matched = by_id
            .get(&cell.id)
            .copied()
            .filter(|r| dist(r.centroid, cell.centroid) <= 1.0)
            .or_else(|| {
                decoded
                    .cells
                    .iter()
                    .find(|r| dist(r.centroid, cell.centroid) <= 1.0)
            });
        let Some(rs) = matched else { continue };
        report.matched += 1;
        report.max_centroid_err = report
            .max_centroid_err
            .max(dist(rs.centroid, cell.centroid));
        report.type_agree += usize::from(rs.cell_type == cell.cell_type);
        report.contour_exact += usize::from(rs.contour == cell.contour);
        report.bbox_exact += usize::from(rs.bbox == cell.bbox);
        report.type_prob_exact += usize::from(rs.type_prob == cell.type_prob as f32);
    }
    report
}

#[test]
fn decode_matches_python_reference_on_fixture_tiles() {
    println!(
        "{:<10} {:>6} {:>6} {:>8} {:>8} {:>6} {:>7} {:>9} {:>8} {:>9}",
        "tile",
        "py",
        "rust",
        "matched",
        "pix_agr",
        "ids=",
        "type",
        "max_c_err",
        "contour",
        "decode_ms"
    );
    let mut failures = Vec::new();
    for tile in complete_tiles() {
        let name = tile
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let py = load_py_cells(&tile);
        let py_map = read_le(&tile.join("instance_map.bin"), i32::from_le_bytes);
        let (decoded, elapsed) = decode_tile(&tile);
        let r = compare_tile(&py, &py_map, &decoded);
        println!(
            "{:<10} {:>6} {:>6} {:>8.4} {:>8.4} {:>6} {:>7.4} {:>9.2e} {:>8.4} {:>9.1}",
            name,
            r.py_cells,
            r.rs_cells,
            ratio(r.matched, r.py_cells),
            r.pixel_agreement,
            r.exact_ids,
            ratio(r.type_agree, r.matched),
            r.max_centroid_err,
            ratio(r.contour_exact, r.matched),
            elapsed.as_secs_f64() * 1e3,
        );
        // The port is bit-exact on every fixture, so anything short of identity is a
        // regression: same instance ids per pixel and identical per-cell metadata.
        let identical = r.exact_ids
            && r.py_cells == r.rs_cells
            && r.matched == r.py_cells
            && r.max_centroid_err == 0.0
            && [
                r.type_agree,
                r.contour_exact,
                r.bbox_exact,
                r.type_prob_exact,
            ]
            .iter()
            .all(|&n| n == r.matched);
        if !identical {
            failures.push(name);
        }
    }
    assert!(
        failures.is_empty(),
        "tiles not identical to the Python reference: {failures:?}"
    );
}

#[test]
fn burn_predictions_preserve_dense_tile_cells() {
    use path_2_burn::cellvit::{CellVit, NUCLEI_CLASSES, SAM_H_X40, TISSUE_CLASSES};
    use path_2_burn::select_device;

    let store = support::model_store();
    let (device, _) = select_device();
    let model = CellVit::load(&store, &device, &SAM_H_X40).expect("load CellViT");
    for name in ["tile04_0", "tile04_1"] {
        let tile = fixtures_dir().join(name);
        let prediction = model
            .predict(read_le(&tile.join("input.bin"), f32::from_le_bytes))
            .expect("CellViT prediction");
        assert!(
            prediction
                .nuclei_prob
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        );
        assert!(prediction.hv.iter().all(|v| v.is_finite()));
        assert!(
            prediction
                .type_pred
                .iter()
                .all(|&c| usize::from(c) < NUCLEI_CLASSES.len())
        );
        assert!(
            prediction
                .tissue_probs
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        );
        assert_eq!(prediction.tissue_probs.len(), TISSUE_CLASSES.len());
        assert!((prediction.tissue_probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let decoded = decode(&HvMaps {
            width: TILE,
            height: TILE,
            nuclei_prob: &prediction.nuclei_prob,
            hv: &prediction.hv,
            type_pred: &prediction.type_pred,
        })
        .expect("decode Burn prediction");
        let cells = load_py_cells(&tile);
        assert!(!cells.is_empty(), "{name}: use a positive tissue field");
        let labels = read_le(&tile.join("instance_map.bin"), i32::from_le_bytes);
        assert_eq!(labels.len(), TILE * TILE, "{name}: reference mask shape");
        let r = compare_tile(&cells, &labels, &decoded);
        println!(
            "{name}: reference={} Burn={} matched={} types={} foreground_agreement={:.8} max_centroid_error={:.6}px",
            r.py_cells, r.rs_cells, r.matched, r.type_agree, r.pixel_agreement, r.max_centroid_err
        );
        assert_eq!(r.rs_cells, r.py_cells, "{name}: cell count changed");
        assert_eq!(
            r.matched, r.py_cells,
            "{name}: missing centroid match within one pixel"
        );
        assert_eq!(r.type_agree, r.py_cells, "{name}: cell type changed");
        // Permit small boundary changes from float32 backend reductions while
        // preserving the cells. CPU decoding of fixed maps remains bit-exact above.
        assert!(
            r.pixel_agreement >= 0.9999,
            "{name}: foreground agreement {} < 0.9999",
            r.pixel_agreement
        );
    }
}

/// Synthetic tile: two touching nuclei, one nucleus at the border whose marker
/// does not survive the opening, and a blob below the minimum size.
fn synthetic_maps() -> (usize, usize, Vec<f32>, Vec<f32>, Vec<u8>) {
    let (w, h) = (40usize, 28usize);
    let nuclei = [
        (12.0f32, 14.0f32, 7.0f32),
        (24.0, 13.0, 8.0),
        (37.0, 3.0, 5.0),
        (5.0, 25.0, 1.2),
    ];
    let mut prob = vec![0.1f32; w * h];
    let mut hv = vec![0.0f32; 2 * w * h];
    let mut best = vec![f32::INFINITY; w * h];
    for (cx, cy, r) in nuclei {
        for (i, b) in best.iter_mut().enumerate() {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let d = ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt() / r;
            if d <= 1.0 && d < *b {
                prob[i] = 0.9;
                hv[i] = (x - cx) / r;
                hv[w * h + i] = (y - cy) / r;
                *b = d;
            }
        }
    }
    let types = (0..w * h)
        .map(|i| match (i % w, i / w) {
            (x, y) if x < 18 => {
                if y < 12 {
                    0
                } else {
                    2
                }
            }
            (x, y) if x < 32 => {
                if y < 10 {
                    3
                } else if y > 16 {
                    4
                } else {
                    0
                }
            }
            _ => 5,
        })
        .collect();
    (w, h, prob, hv, types)
}

#[test]
fn decode_matches_python_reference_on_synthetic_tile() {
    let (w, h, prob, hv, types) = synthetic_maps();
    let decoded = decode(&maps(w, h, &prob, &hv, &types)).expect("synthetic maps decode");
    // DetectionCellPostProcessor(nr_types=6, magnification=40) on the same maps.
    let expected_rows = [
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000001000000000000000",
        "0000000000000000000001111111000000000000",
        "0000000000002000000111111111110000000000",
        "0000000002222222001111111111111000000000",
        "0000000022222222201111111111111000000000",
        "0000000222222222221111111111111100000000",
        "0000002222222222221111111111111100000000",
        "0000002222222222221111111111111100000000",
        "0000002222222222221111111111111110000000",
        "0000022222222222221111111111111100000000",
        "0000002222222222221111111111111100000000",
        "0000002222222222221111111111111100000000",
        "0000002222222222221111111111111000000000",
        "0000000222222222221111111111111000000000",
        "0000000022222222200111111111110000000000",
        "0000000002222222000001111111000000000000",
        "0000000000002000000000001000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000",
    ];
    let expected_map: Vec<u32> = expected_rows
        .iter()
        .flat_map(|r| r.bytes().map(|b| u32::from(b - b'0')))
        .collect();
    assert_eq!(decoded.instance_map, expected_map);
    let expected = [
        DecodedCell {
            id: 1,
            centroid: [24.3015873015873, 13.0],
            bbox: [18, 5, 15, 17],
            contour: vec![
                [24, 5],
                [23, 6],
                [21, 6],
                [20, 7],
                [19, 7],
                [18, 8],
                [18, 18],
                [19, 19],
                [20, 19],
                [21, 20],
                [23, 20],
                [24, 21],
                [25, 20],
                [27, 20],
                [28, 19],
                [29, 19],
                [30, 18],
                [30, 17],
                [31, 16],
                [31, 14],
                [32, 13],
                [31, 12],
                [31, 10],
                [30, 9],
                [30, 8],
                [29, 7],
                [28, 7],
                [27, 6],
                [25, 6],
            ],
            // Background (0) has the most votes; types 3 and 4 tie, the lower id wins.
            cell_type: 3,
            type_prob: 0.238_095_24, // Python: 0.23809523683547493
        },
        DecodedCell {
            id: 2,
            centroid: [11.652482269503546, 14.0],
            bbox: [5, 7, 13, 15],
            contour: vec![
                [12, 7],
                [11, 8],
                [9, 8],
                [6, 11],
                [6, 13],
                [5, 14],
                [6, 15],
                [6, 17],
                [9, 20],
                [11, 20],
                [12, 21],
                [13, 20],
                [15, 20],
                [17, 18],
                [17, 10],
                [15, 8],
                [13, 8],
            ],
            cell_type: 2,
            type_prob: 0.716_312_05, // Python: 0.7163120516573613
        },
    ];
    assert_eq!(decoded.cells, expected);
}

#[test]
fn logit_helpers_match_numpy() {
    // softmax(binary, axis=0)[1] in float32 for [2, 1, 3] logits.
    let binary = [0.5f32, -2.0, 3.0, 0.5, 1.0, -4.0];
    let prob = nuclei_prob_from_logits(&binary, 3, 1).expect("valid logits");
    let expected_bits = [1056964608u32, 1064557542, 980341661];
    assert_eq!(
        prob.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
        expected_bits
    );
    // np.argmax(types, axis=0) for [3, 1, 4]: first maximum wins ties, first NaN wins outright.
    let nan = f32::NAN;
    let types = [
        1.0f32, 0.0, nan, 2.0, 1.0, nan, 1.0, 5.0, 0.0, 5.0, nan, 5.0,
    ];
    assert_eq!(
        type_argmax(&types, 3, 4, 1).expect("valid logits"),
        [0, 1, 0, 1]
    );
}

fn maps<'a>(
    width: usize,
    height: usize,
    prob: &'a [f32],
    hv: &'a [f32],
    types: &'a [u8],
) -> HvMaps<'a> {
    HvMaps {
        width,
        height,
        nuclei_prob: prob,
        hv,
        type_pred: types,
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let mismatch = |name, expected, actual| PostprocessError::LengthMismatch {
        name,
        expected,
        actual,
    };
    assert_eq!(
        decode(&maps(2, 2, &[0.0; 3], &[0.0; 8], &[0; 4])),
        Err(mismatch("nuclei_prob", 4, 3))
    );
    assert_eq!(
        decode(&maps(2, 2, &[0.0; 4], &[0.0; 4], &[0; 4])),
        Err(mismatch("hv", 8, 4))
    );
    assert_eq!(
        decode(&maps(2, 2, &[0.0; 4], &[0.0; 8], &[0; 5])),
        Err(mismatch("type_pred", 4, 5))
    );
    assert_eq!(
        decode(&maps(0, 2, &[], &[], &[])),
        Err(PostprocessError::EmptyImage {
            width: 0,
            height: 2
        })
    );
    assert_eq!(
        decode(&maps(usize::MAX, 2, &[], &[], &[])),
        Err(PostprocessError::DimensionOverflow {
            width: usize::MAX,
            height: 2
        })
    );
    assert_eq!(
        nuclei_prob_from_logits(&[0.0; 3], 1, 2),
        Err(mismatch("binary_logits", 4, 3))
    );
    assert_eq!(
        type_argmax(&[0.0; 4], 0, 2, 2),
        Err(PostprocessError::InvalidClassCount(0))
    );
    assert_eq!(
        type_argmax(&[0.0; 4], 257, 2, 2),
        Err(PostprocessError::InvalidClassCount(257))
    );
    assert_eq!(
        type_argmax(&[0.0; 7], 2, 2, 2),
        Err(mismatch("type_logits", 8, 7))
    );
}

#[test]
fn decode_handles_degenerate_inputs() {
    // The Python reference cannot run on 1-pixel-high maps (np.squeeze drops the
    // axis); the port just returns an empty decoding.
    let decoded = decode(&maps(1, 1, &[0.9], &[0.0, 0.0], &[1])).expect("1x1 decodes");
    assert_eq!(
        decoded,
        Decoded {
            instance_map: vec![0],
            cells: vec![]
        }
    );
    // Uniform maps hit the zero-range normalisation path; the reference finds no
    // markers and returns an all-background map.
    let (w, h) = (16, 16);
    let (prob, hv, types) = (vec![0.9; w * h], vec![0.25; 2 * w * h], vec![2; w * h]);
    let decoded = decode(&maps(w, h, &prob, &hv, &types)).expect("uniform maps decode");
    assert_eq!(
        decoded,
        Decoded {
            instance_map: vec![0; w * h],
            cells: vec![]
        }
    );
}
