//! Pathology models in Burn.
//!
//! CellViT, HoVer-Net, Cellpose-SAM, InstanSeg and the tile encoders in
//! [`encoders`] are written directly in Burn and load their published PyTorch
//! checkpoints by the original parameter names.
//! Weights live outside the crate, listed in a `manifest.json` (see
//! [`ModelStore`]); this crate carries the architectures, their typed wrappers
//! and model-specific decoding.

pub mod cellpose;
pub mod cellvit;
pub mod encoders;
pub mod hover;
pub mod hovernet;
pub mod instanseg;
mod ops;
mod preprocess;
mod runtime;
mod store;

pub use burn::prelude::Device;
pub use runtime::{
    Checkpoint, DeviceChoice, REQUIRED_GPU_BUFFER_BYTES, default_device, load_pytorch_weights,
    read_f32, select_device, tensor_from_f32,
};
pub use store::{MODEL_DIR_ENV, ModelError, ModelStore};
