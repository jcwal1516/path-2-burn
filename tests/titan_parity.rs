//! TITAN slide embeddings against its upstream float32 CPU implementation.

mod support;

use path_2_burn::{select_device, titan::Titan};
use support::{assert_close, fixture_dirs, model_store, read_f32, read_le, root};

// On the full demo and edge cases, torch MPS vs CPU reaches 3.1e-6;
// Burn's largest difference is 4e-6. Keep a small float32 backend margin.
const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

#[test]
#[ignore = "needs TITAN weights and fixtures from export/reference_titan.py"]
fn slide_embeddings_match_upstream() {
    let (device, _) = select_device();
    let model = Titan::load(&model_store(), &device).expect("load TITAN");
    for fixture in fixture_dirs(
        &root().join("fixtures/titan"),
        "embedding.bin",
        "run export/reference_titan.py reference",
    ) {
        let features = read_f32(&fixture.join("features.bin"));
        let coords: Vec<i64> = read_le(&fixture.join("coords.bin"), i64::from_le_bytes);
        assert_eq!(coords.len() % 2, 0);
        let coords: Vec<[i64; 2]> = coords.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(fixture.join("meta.json")).expect("read metadata"),
        )
        .expect("parse metadata");
        let patch_size = meta["patch_size_lv0"].as_u64().expect("patch size") as u32;
        let embedding = model
            .encode_slide_from_patch_features(features, &coords, patch_size)
            .expect("slide embedding");
        println!("{}", fixture.file_name().unwrap().to_string_lossy());
        assert_close(
            "embedding",
            &embedding,
            &read_f32(&fixture.join("embedding.bin")),
            768,
            ATOL,
            RTOL,
        );
    }
}
