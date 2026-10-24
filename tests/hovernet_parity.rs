//! Burn vs hover_net for HoVer-Net's published checkpoints: all three branches
//! on fixture tiles at 40x.
//!
//! Download a checkpoint from vqdang/hover_net, list it in weights/manifest.json
//! under its name in `path_2_burn::hovernet`, and run
//! `export/reference_hovernet.py` for it. Ignored by default:
//!
//!   cargo test --release --test hovernet_parity -- --ignored pannuke

mod support;

use path_2_burn::hovernet::{self, HoverNet, HoverNetWeights};
use path_2_burn::select_device;
use support::{assert_close, model_store, read_f32, read_tiles, root, scale};

/// Float32 through a ResNet-50 and dense decoders. The error in every branch
/// grows with the shared encoder's activations, so the absolute budget scales
/// with the checkpoint's largest output over all branches:
/// |burn - torch| <= ATOL_SCALE * max|torch| + RTOL * |torch|. On the CoNSeP
/// typed checkpoint (type logits up to 86), torch on MPS differs from these CPU
/// references by up to 0.0077 (nucleus logits) and 0.0014 (HV, within ±1).
const ATOL_SCALE: f32 = 2e-4;
const RTOL: f32 = 1e-4;

fn check(weights: &HoverNetWeights) {
    let fixtures = root().join("fixtures").join(weights.name);
    let side = weights.mode.input_size();
    let tile_bytes = side * side * 3;
    let (rgb, count) = read_tiles(&fixtures.join("rgb.bin"), tile_bytes);

    let (device, _) = select_device();
    let model = HoverNet::load(&model_store(), &device, weights)
        .unwrap_or_else(|e| panic!("load {}: {e}", weights.name));
    let (mut np, mut hv, mut tp) = (Vec::new(), Vec::new(), Vec::new());
    for tile in rgb.chunks_exact(tile_bytes) {
        let raw = model
            .forward_raw(model.normalize_rgb(tile).expect("input"))
            .expect("forward");
        np.extend(raw.np);
        hv.extend(raw.hv);
        tp.extend(raw.tp);
    }
    let plane = weights.mode.output_size().pow(2);
    let reference = |branch: &str| read_f32(&fixtures.join(format!("{branch}.bin")));
    let (np_ref, hv_ref) = (reference("np"), reference("hv"));
    let tp_ref = weights.types.map(|_| reference("tp")).unwrap_or_default();
    let atol = ATOL_SCALE * scale(&[&np_ref, &hv_ref, &tp_ref]);
    assert_close("np", &np, &np_ref, count * 2 * plane, atol, RTOL);
    assert_close("hv", &hv, &hv_ref, count * 2 * plane, atol, RTOL);
    if let Some(types) = weights.types {
        assert_close("tp", &tp, &tp_ref, count * types * plane, atol, RTOL);
    } else {
        assert!(
            tp.is_empty(),
            "a segmentation-only checkpoint has no type branch"
        );
    }
}

support::parity_tests! {
    check,
    "needs the checkpoint in weights/ and fixtures from export/reference_hovernet.py",
    pannuke => hovernet::PANNUKE,
    monusac => hovernet::MONUSAC,
    consep => hovernet::CONSEP,
    consep_notype => hovernet::CONSEP_NOTYPE,
    kumar => hovernet::KUMAR,
    cpm17 => hovernet::CPM17,
}
