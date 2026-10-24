//! Shared by tests/instanseg_parity.rs and examples/parity_instanseg.rs, which
//! also include `tests/support` as `support`.

use std::collections::HashMap;
use std::path::PathBuf;

pub const SIDE: usize = 1024;

pub fn fixtures_dir() -> PathBuf {
    crate::support::root().join("fixtures").join("instanseg")
}

/// Complete fixture tiles, or a panic that says how to make them.
pub fn fixture_tiles() -> Vec<PathBuf> {
    crate::support::fixture_dirs(
        &fixtures_dir(),
        "meta.json",
        "run export/reference_instanseg.py",
    )
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LabelParity {
    pub reference_instances: usize,
    pub rust_instances: usize,
    /// Reference instances with a Rust instance at IoU >= 0.5.
    pub matched: usize,
    pub mean_matched_iou: f64,
    /// Foreground IoU (any label vs any label).
    pub foreground_iou: f64,
    /// Pixels whose label value is identical.
    pub identical_pixels: f64,
}

impl LabelParity {
    pub fn matched_fraction(&self) -> f64 {
        if self.reference_instances == 0 {
            1.0
        } else {
            self.matched as f64 / self.reference_instances as f64
        }
    }
}

pub fn compare_labels(reference: &[i32], rust: &[u32]) -> LabelParity {
    assert_eq!(reference.len(), rust.len());
    let mut area_ref: HashMap<i32, u64> = HashMap::new();
    let mut area_rust: HashMap<u32, u64> = HashMap::new();
    let mut overlap: HashMap<(i32, u32), u64> = HashMap::new();
    let (mut fg_and, mut fg_or, mut identical) = (0u64, 0u64, 0u64);
    for (&a, &b) in reference.iter().zip(rust) {
        identical += u64::from(a >= 0 && a as u32 == b);
        if a > 0 {
            *area_ref.entry(a).or_default() += 1;
        }
        if b > 0 {
            *area_rust.entry(b).or_default() += 1;
        }
        if a > 0 && b > 0 {
            *overlap.entry((a, b)).or_default() += 1;
            fg_and += 1;
        }
        fg_or += u64::from(a > 0 || b > 0);
    }
    let mut best: HashMap<i32, f64> = HashMap::new();
    for (&(a, b), &inter) in &overlap {
        let iou = inter as f64 / (area_ref[&a] + area_rust[&b] - inter) as f64;
        let entry = best.entry(a).or_default();
        *entry = entry.max(iou);
    }
    let matched: Vec<f64> = best.values().copied().filter(|&iou| iou >= 0.5).collect();
    LabelParity {
        reference_instances: area_ref.len(),
        rust_instances: area_rust.len(),
        matched: matched.len(),
        mean_matched_iou: if matched.is_empty() {
            1.0
        } else {
            matched.iter().sum::<f64>() / matched.len() as f64
        },
        foreground_iou: if fg_or == 0 {
            1.0
        } else {
            fg_and as f64 / fg_or as f64
        },
        identical_pixels: identical as f64 / reference.len() as f64,
    }
}
