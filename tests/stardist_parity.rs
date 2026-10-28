//! StarDist network outputs against OpenCV and polygon selection against StarDist.

mod support;

use path_2_burn::{
    select_device,
    stardist::{self, NmsOptions, StarDist, StarDistModel, StarDistRaw},
};
use support::{assert_close, fixture_dirs, model_store, read_f32, read_i32, root};

fn check(spec: &StarDistModel) {
    let (device, _) = select_device();
    let model = StarDist::load(&model_store(), &device, spec).expect("load StarDist");
    for fixture in fixture_dirs(
        &root().join("fixtures/stardist").join(spec.name),
        "input.bin",
        "run export/reference_stardist.py reference",
    ) {
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
        let height = meta["crop_yxhw"][2].as_u64().unwrap() as usize;
        let width = meta["crop_yxhw"][3].as_u64().unwrap() as usize;
        let mut input = read_f32(&fixture.join("raw_input.bin"));
        spec.normalize(&mut input, height, width).unwrap();
        assert_close(
            "normalization",
            &input,
            &read_f32(&fixture.join("input.bin")),
            input.len(),
            1e-6,
            1e-6,
        );
        let actual = model.predict(input, height, width).unwrap();
        let reference = StarDistRaw {
            height: height / spec.grid,
            width: width / spec.grid,
            grid: spec.grid,
            probability: read_f32(&fixture.join("probability.bin")),
            distances: read_f32(&fixture.join("distances.bin")),
        };
        let count = reference.height * reference.width;
        println!(
            "{} {}",
            spec.name,
            fixture.file_name().unwrap().to_string_lossy()
        );
        assert_close(
            "probability",
            &actual.probability,
            &reference.probability,
            count,
            1e-4,
            1e-4,
        );
        assert_close(
            "distances",
            &actual.distances,
            &reference.distances,
            count * 32,
            1e-3,
            1e-4,
        );
        // Isolate postprocessing from network roundoff: use the exact OpenCV arrays.
        let instances = reference.instances(&NmsOptions::default()).unwrap();
        let points: Vec<i32> = instances
            .iter()
            .flat_map(|p| [p.center_yx[0] as i32, p.center_yx[1] as i32])
            .collect();
        assert_eq!(
            points,
            read_i32(&fixture.join("points.bin")),
            "NMS selected points"
        );
        let scores: Vec<f32> = instances.iter().map(|p| p.probability).collect();
        assert_eq!(
            scores,
            read_f32(&fixture.join("scores.bin")),
            "NMS selected scores"
        );
    }
}

support::parity_tests! {
    check,
    "needs exported StarDist weights and OpenCV/NMS fixtures",
    he => stardist::HE,
    fluorescence => stardist::FLUORESCENCE,
    fluorescence_paper => stardist::FLUORESCENCE_PAPER,
}
