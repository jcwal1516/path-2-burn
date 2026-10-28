//! Cellpose's 2D flow dynamics, quality filter and hole filling on saved flows.

mod support;

use path_2_burn::cellpose::{DynamicsOptions, compute_masks};
use support::{fixture_dirs, read_f32, read_le, root};

#[test]
#[ignore = "needs mask fixtures from export/reference_cellpose.py masks"]
fn masks_match_cellpose_on_network_and_synthetic_flows() {
    for fixture in fixture_dirs(
        &root().join("fixtures/cellpose_sam/masks"),
        "flows.bin",
        "run export/reference_cellpose.py masks",
    ) {
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
        let height = meta["height"].as_u64().unwrap() as usize;
        let width = meta["width"].as_u64().unwrap() as usize;
        let flow = read_f32(&fixture.join("flows.bin"));
        for (file, options) in [
            (
                "unfiltered.bin",
                DynamicsOptions {
                    flow_threshold: 0.0,
                    min_size: 0,
                    ..DynamicsOptions::default()
                },
            ),
            ("masks.bin", DynamicsOptions::default()),
        ] {
            let actual = compute_masks(&flow, height, width, &options).unwrap();
            let expected: Vec<u32> = read_le(&fixture.join(file), u32::from_le_bytes);
            let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            assert_eq!(actual.len(), expected.len());
            println!(
                "{} {file}: {} masks, {different} different pixels",
                fixture.file_name().unwrap().to_string_lossy(),
                actual.iter().max().unwrap()
            );
            assert_eq!(different, 0, "mask parity");
        }
    }
}
