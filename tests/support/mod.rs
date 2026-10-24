//! Helpers shared by the parity tests and examples. Tests include it with
//! `mod support;`, examples with `#[path = "../tests/support/mod.rs"] mod support;`.
#![allow(dead_code, unused_imports, unused_macros)] // each test binary uses a subset

use std::path::{Path, PathBuf};

/// The crate root, which holds `weights/` and `fixtures/`.
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The local weights directory, `weights/`, with its manifest.
pub fn model_store() -> path_2_burn::ModelStore {
    path_2_burn::ModelStore::open(root().join("weights"))
        .unwrap_or_else(|e| panic!("open weights/: {e}"))
}

/// A fixture of whole RGB u8 tiles of `tile_bytes` each, and how many it holds.
pub fn read_tiles(path: &Path, tile_bytes: usize) -> (Vec<u8>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        !bytes.is_empty() && bytes.len() % tile_bytes == 0,
        "{}: {} bytes is not a whole number of {tile_bytes}-byte tiles",
        path.display(),
        bytes.len()
    );
    let count = bytes.len() / tile_bytes;
    (bytes, count)
}

/// Subdirectories of `dir` that contain `marker`, sorted. Panics, naming `how`
/// to generate them, when there are none.
pub fn fixture_dirs(dir: &Path, marker: &str, how: &str) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{} missing ({e}); {how}", dir.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.join(marker).is_file())
        .collect();
    dirs.sort();
    assert!(
        !dirs.is_empty(),
        "no fixtures with {marker} in {}; {how}",
        dir.display()
    );
    dirs
}

/// A file of little-endian `N`-byte values.
pub fn read_le<const N: usize, T>(path: &Path, from: fn([u8; N]) -> T) -> Vec<T> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        bytes.len() % N,
        0,
        "{} is not a whole number of {N}-byte values",
        path.display()
    );
    bytes
        .chunks_exact(N)
        .map(|chunk| from(chunk.try_into().expect("chunk of N bytes")))
        .collect()
}

pub fn read_f32(path: &Path) -> Vec<f32> {
    read_le(path, f32::from_le_bytes)
}

pub fn read_i32(path: &Path) -> Vec<i32> {
    read_le(path, i32::from_le_bytes)
}

/// Checks every value against `atol + rtol * |reference|`, after checking
/// that both sides hold `expected` (nonzero) values and are finite. Prints the
/// largest and mean differences and returns the largest.
pub fn assert_close(
    name: &str,
    actual: &[f32],
    reference: &[f32],
    expected: usize,
    atol: f32,
    rtol: f32,
) -> f32 {
    assert!(expected > 0, "{name}: no values to compare");
    assert_eq!(actual.len(), expected, "{name}: output length");
    assert_eq!(reference.len(), expected, "{name}: reference length");
    let (mut worst, mut sum) = (0f32, 0f64);
    let mut failures = 0usize;
    let mut first_failure = None;
    for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
        assert!(
            a.is_finite() && r.is_finite(),
            "{name}[{i}]: non-finite output/reference ({a}, {r})"
        );
        let diff = (a - r).abs();
        worst = worst.max(diff);
        sum += f64::from(diff);
        let allowed = atol + rtol * r.abs();
        if diff > allowed {
            failures += 1;
            first_failure.get_or_insert((i, a, r, allowed));
        }
    }
    let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!(
        "    {name:<18} max|d| {worst:>10.6}  mean|d| {:>10.7}  (|ref|max {scale:.3})",
        sum / reference.len() as f64
    );
    assert_eq!(
        failures, 0,
        "{name}: values exceed atol={atol}, rtol={rtol}; first (index, output, reference, allowed): {first_failure:?}"
    );
    worst
}

/// [`assert_close`] with the absolute budget `atol_scale` times the largest
/// reference value (at least 1): float32 error in a deep network grows with its
/// activations, not with each value. Torch's own MPS run is the yardstick.
pub fn assert_close_scaled(
    name: &str,
    actual: &[f32],
    reference: &[f32],
    expected: usize,
    atol_scale: f32,
    rtol: f32,
) -> f32 {
    assert_close(
        name,
        actual,
        reference,
        expected,
        atol_scale * scale(&[reference]),
        rtol,
    )
}

/// The largest magnitude in `tensors`, at least 1.
pub fn scale(tensors: &[&[f32]]) -> f32 {
    tensors
        .iter()
        .flat_map(|values| values.iter())
        .fold(1f32, |m, v| m.max(v.abs()))
}

/// One `#[ignore]`d test per `test => entry`, each running `check(&entry)`;
/// for parity tests that need weights the repository does not ship.
macro_rules! parity_tests {
    ($check:ident, $reason:literal, $($test:ident => $entry:path),* $(,)?) => {
        $(
            #[test]
            #[ignore = $reason]
            fn $test() {
                $check(&$entry);
            }
        )*
    };
}
pub(crate) use parity_tests;

/// CellViT's float32 backend budgets. PyTorch/MPS against the CPU references
/// reaches 0.00292 binary-logit, 0.000158 type, 0.000042 HV and 0.000017 tissue
/// error on the PanNuke fixtures.
pub mod cellvit {
    use path_2_burn::cellvit::{CellVitRaw, NUCLEI_CLASSES, TISSUE_CLASSES};
    use std::path::Path;

    pub const RTOL: f32 = 1e-4;
    pub const BINARY_ATOL: f32 = 5e-3;
    pub const TYPE_ATOL: f32 = 5e-4;
    pub const HV_ATOL: f32 = 1e-4;
    pub const TISSUE_ATOL: f32 = 1e-4;

    /// A fixture tile's torch outputs, one `.bin` file per field.
    pub fn read_reference(tile: &Path) -> CellVitRaw {
        let read = |name: &str| super::read_f32(&tile.join(format!("{name}.bin")));
        CellVitRaw {
            nuclei_binary_map: read("nuclei_binary_map"),
            hv_map: read("hv_map"),
            nuclei_type_map: read("nuclei_type_map"),
            tissue_types: read("tissue_types"),
        }
    }

    /// Checks all four outputs of a `pixels`-pixel tile, each within its
    /// budget, and returns the largest difference.
    pub fn assert_raw_close(burn: &CellVitRaw, torch: &CellVitRaw, pixels: usize) -> f32 {
        let types = NUCLEI_CLASSES.len() * pixels;
        [
            (
                "nuclei_binary_map",
                &burn.nuclei_binary_map,
                &torch.nuclei_binary_map,
                2 * pixels,
                BINARY_ATOL,
            ),
            ("hv_map", &burn.hv_map, &torch.hv_map, 2 * pixels, HV_ATOL),
            (
                "nuclei_type_map",
                &burn.nuclei_type_map,
                &torch.nuclei_type_map,
                types,
                TYPE_ATOL,
            ),
            (
                "tissue_types",
                &burn.tissue_types,
                &torch.tissue_types,
                TISSUE_CLASSES.len(),
                TISSUE_ATOL,
            ),
        ]
        .into_iter()
        .fold(0f32, |worst, (name, burn, torch, count, atol)| {
            worst.max(super::assert_close(name, burn, torch, count, atol, RTOL))
        })
    }
}
