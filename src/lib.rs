//! Pathology models in Burn.
//!
//! CellViT, HoVer-Net, Cellpose-SAM, the tile encoders in [`encoders`] and
//! [`titan`] load their published PyTorch checkpoints by the original parameter
//! names. InstanSeg and [`stardist`] load tensors extracted from their original
//! TorchScript and frozen TensorFlow models. Every network runs natively in Burn.
//! Weights live outside the crate, listed in a `manifest.json` (see
//! [`ModelStore`]); this crate carries the architectures, their typed wrappers
//! and model-specific decoding.

pub mod cellpose;
pub mod cellvit;
pub mod encoders;
pub mod hover;
pub mod hovernet;
pub mod instanseg;
mod open_clip;
mod ops;
mod preprocess;
mod runtime;
pub mod stardist;
mod store;
pub mod titan;

pub use burn::prelude::Device;
pub use runtime::{
    Checkpoint, DeviceChoice, REQUIRED_GPU_BUFFER_BYTES, default_device, load_pytorch_weights,
    read_f32, select_device, tensor_from_f32,
};
pub use store::{MODEL_DIR_ENV, ModelError, ModelStore};
