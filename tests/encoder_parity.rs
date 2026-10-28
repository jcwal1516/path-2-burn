//! Burn vs torch for the tile encoders in `path_2_burn::encoders`.
//!
//! `export/reference_encoders.py download` fetches each checkpoint with your own
//! Hugging Face access into weights/<name>/ and lists it in weights/manifest.json;
//! `reference` writes fixtures/encoders/: N tiles at 0.5 µm/px (two crops
//! of each fixtures/tiles PNG) and, per model, every output token and the pooled
//! embedding from torch float32 on the CPU.
//! The weights are large and mostly gated, so these tests are ignored by default:
//!
//!   cargo test --release --test encoder_parity -- --ignored [name]

mod support;

use path_2_burn::encoders::{self, Encoder, TileEncoder};
use path_2_burn::select_device;
use std::path::Path;
use support::{assert_close_scaled, model_store, read_f32, read_tiles, root};

/// Float32 error after up to 40 blocks grows with the activations, so the
/// absolute budget is a fraction of the largest reference value:
/// |burn - torch| <= ATOL_SCALE * max|torch| + RTOL * |torch|. Torch on MPS
/// differs from these CPU references by up to 0.0064 on Virchow2, whose tokens
/// reach 50; Burn on the GPU by up to 0.0058.
/// CONCH v1.5 on four tiles drifts by 8.1e-5 (tokens) / 1.1e-5 (embedding)
/// in torch MPS; Burn on all 20 tiles stays below 5e-5 / 8e-6.
/// On four tiles, torch MPS reaches 1.94e-4 / 1.43e-5 for CONCH v1 and
/// 4.58e-5 / 1.01e-5 for PLIP (tokens / embedding).
const ATOL_SCALE: f32 = 2e-4;
const RTOL: f32 = 1e-3;
/// Tiles per forward pass.
const BATCH: usize = 4;

fn check(encoder: &Encoder) {
    let fixtures = root().join("fixtures").join("encoders");
    let tile_bytes = encoder.tile * encoder.tile * 3;
    let tile_file = if encoder.tile == 224 {
        "tiles.bin".to_string()
    } else {
        format!("tiles_{}.bin", encoder.tile)
    };
    let (tiles, count) = read_tiles(&fixtures.join(tile_file), tile_bytes);

    let (device, _) = select_device();
    let model = TileEncoder::load(&model_store(), &device, encoder)
        .unwrap_or_else(|e| panic!("load {}: {e}", encoder.name));
    let (mut tokens, mut embeddings) = (Vec::new(), Vec::new());
    for batch in tiles.chunks(BATCH * tile_bytes) {
        let input = model.normalize_rgb(batch).expect("normalize tiles");
        tokens.extend(model.tokens(input.clone()).expect("tokens"));
        embeddings.extend(model.embed(input).expect("embedding"));
    }
    let vit = &encoder.vit;
    let grid = encoder.tile / vit.patch_size;
    let reference = fixtures.join(encoder.name);
    println!("{} on {count} tiles", encoder.name);
    let token_count = count * (vit.prefix_tokens() + grid * grid) * vit.embed_dim;
    compare(
        "tokens",
        &tokens,
        &reference.join("tokens.bin"),
        token_count,
    );
    let embedding_count = count * encoder.embedding_dim();
    compare(
        "embedding",
        &embeddings,
        &reference.join("embedding.bin"),
        embedding_count,
    );
}

fn compare(name: &str, burn: &[f32], reference: &Path, expected: usize) {
    assert_close_scaled(name, burn, &read_f32(reference), expected, ATOL_SCALE, RTOL);
}

support::parity_tests! {
    check,
    "needs the checkpoint in weights/ and fixtures from export/reference_encoders.py",
    uni => encoders::UNI,
    uni2_h => encoders::UNI2_H,
    virchow => encoders::VIRCHOW,
    virchow2 => encoders::VIRCHOW2,
    h_optimus_0 => encoders::H_OPTIMUS_0,
    h_optimus_1 => encoders::H_OPTIMUS_1,
    prov_gigapath => encoders::PROV_GIGAPATH,
    phikon => encoders::PHIKON,
    phikon_v2 => encoders::PHIKON_V2,
    midnight_12k => encoders::MIDNIGHT_12K,
    conch_v1_5 => encoders::CONCH_V1_5,
    conch_v1 => encoders::CONCH_V1,
    plip => encoders::PLIP,
}
