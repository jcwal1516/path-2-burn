//! Device selection, weight loading and tensor readback shared by all models.

use crate::store::{ModelError, ModelStore};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_store::{
    ApplyResult, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use std::fmt;

/// Largest single buffer the models allocate: InstanSeg's 64-channel
/// full-resolution activation (64 x 1024 x 1024 f32). A GPU that cannot hold it
/// fails mid-inference, so it is not used.
pub const REQUIRED_GPU_BUFFER_BYTES: u64 = 64 * 1024 * 1024 * 4;

/// Kernels cubecl batches into one GPU submission (its default is 32). The
/// viewer draws on the same GPU, and its frames wait behind each submission;
/// at 8, CellViT-SAM-H tiles take as long as at 32 while the slowest viewer
/// frame drops from about 0.5 s to about 0.1 s. `CUBECL_WGPU_MAX_TASKS`
/// overrides it.
pub const WGPU_TASKS_MAX: usize = 8;

/// Which device the models run on, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceChoice {
    /// `BURN_DEVICE` names the backend.
    Override(String),
    /// A hardware GPU (Metal on macOS, wgpu elsewhere).
    Gpu { name: String },
    /// No usable GPU: the pure-Rust CPU backend, which is much slower.
    Cpu { reason: String },
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Override(name) => write!(f, "BURN_DEVICE={name}"),
            Self::Gpu { name } => write!(f, "GPU ({name})"),
            Self::Cpu { reason } => write!(f, "CPU, because {reason}"),
        }
    }
}

/// What probing the GPU found.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuProbe {
    /// A GPU other than a software rasterizer (llvmpipe, lavapipe, WARP).
    hardware: bool,
    name: String,
    /// Largest single buffer the device allows.
    max_buffer_bytes: u64,
}

/// [`select_device`]'s device.
pub fn default_device() -> Device {
    select_device().0
}

/// The GPU when it can run the models, else the pure-Rust CPU backend.
/// `BURN_DEVICE` overrides the choice.
///
/// burn's own default takes wgpu whenever it is compiled in, so a machine
/// without a GPU driver lands on a software rasterizer that cannot hold the
/// models' buffers and fails mid-inference.
pub fn select_device() -> (Device, DeviceChoice) {
    if let Ok(name) = std::env::var("BURN_DEVICE") {
        return (Device::default(), DeviceChoice::Override(name));
    }
    match choose(probe_gpu()) {
        gpu @ DeviceChoice::Gpu { .. } => (Device::default(), gpu),
        cpu => (Device::flex(), cpu),
    }
}

/// Initializes the wgpu device burn would use, so the probe sees its real
/// limits; burn then reuses that client. Panics inside the driver stack are
/// reported as errors.
fn probe_gpu() -> Result<GpuProbe, String> {
    use cubecl_runtime::runtime::Runtime;
    use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

    std::panic::catch_unwind(|| {
        if !<WgpuRuntime as Runtime>::is_available() {
            return Ok(GpuProbe {
                hardware: false,
                name: String::new(),
                max_buffer_bytes: 0,
            });
        }
        init_wgpu_runtime()?;
        let client = <WgpuRuntime as Runtime>::client(&WgpuDevice::default());
        let properties = client.properties();
        Ok(GpuProbe {
            hardware: true,
            name: properties.identity.name.clone(),
            max_buffer_bytes: properties.memory.max_page_size,
        })
    })
    .map_err(|panic| {
        panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic".to_string())
    })?
}

/// Registers the default wgpu device with [`WGPU_TASKS_MAX`] before anything
/// creates its client lazily with cubecl's defaults. Runs once per process.
fn init_wgpu_runtime() -> Result<(), String> {
    use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuInitError};
    use std::sync::OnceLock;

    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let tasks_max = tasks_max_override(std::env::var("CUBECL_WGPU_MAX_TASKS").ok());
        let options = RuntimeOptions::with_tasks_max(tasks_max).map_err(|e| e.to_string())?;
        match cubecl_wgpu::try_init_setup::<AutoGraphicsApi>(&WgpuDevice::default(), options) {
            // Already registered by an earlier caller: burn reuses that client.
            Ok(_) | Err(WgpuInitError::Registration { .. }) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    })
    .clone()
}

/// The task count to pass cubecl: ours, unless the environment names one
/// (cubecl then reads and validates it).
fn tasks_max_override(env: Option<String>) -> Option<usize> {
    env.is_none().then_some(WGPU_TASKS_MAX)
}

fn choose(probe: Result<GpuProbe, String>) -> DeviceChoice {
    let mib = |bytes: u64| bytes / (1024 * 1024);
    match probe {
        Err(message) => DeviceChoice::Cpu {
            reason: format!("GPU initialization failed: {message}"),
        },
        Ok(gpu) if !gpu.hardware => DeviceChoice::Cpu {
            reason: "no hardware GPU was found (only a software rasterizer, or none)".to_string(),
        },
        Ok(gpu) if gpu.max_buffer_bytes < REQUIRED_GPU_BUFFER_BYTES => DeviceChoice::Cpu {
            reason: format!(
                "{} allows {} MiB per buffer and the models need {} MiB",
                gpu.name,
                mib(gpu.max_buffer_bytes),
                mib(REQUIRED_GPU_BUFFER_BYTES)
            ),
        },
        Ok(gpu) => DeviceChoice::Gpu { name: gpu.name },
    }
}

/// How a PyTorch checkpoint's tensors map onto a Burn module.
pub struct Checkpoint<'a> {
    /// Key of the state dict inside a pickled checkpoint dict; `None` when the
    /// file is the state dict itself (always so for safetensors).
    pub state_dict_key: Option<&'a str>,
    /// (regex, replacement) pairs from checkpoint keys to module paths, applied
    /// in order: `nn.Sequential` indices that cannot be Rust field names, or
    /// another library's naming.
    pub renames: &'a [(&'a str, &'a str)],
    /// Checkpoint tensors the module has no counterpart for, by exact key.
    pub unused: &'a [&'a str],
}

/// Load a verified PyTorch checkpoint (pickled `.pth`/`.bin`, or
/// `.safetensors`) by its original parameter names. Like
/// `load_state_dict(strict=True)`, every module tensor must be present and
/// every checkpoint tensor used, apart from those listed in
/// [`Checkpoint::unused`] and BatchNorm's `num_batches_tracked` counter, so a
/// misnamed field cannot go unnoticed.
///
/// This is how every model here loads, and it works the same for a module of
/// your own, such as a private fine-tune.
pub fn load_pytorch_weights<M: ModuleSnapshot>(
    mut module: M,
    store: &ModelStore,
    model: &str,
    checkpoint: &Checkpoint,
) -> Result<M, ModelError> {
    let path = store.weights(model)?;
    let load_error = |message: String| ModelError::Load {
        model: model.to_string(),
        message,
    };
    let result = if path.extension().is_some_and(|ext| ext == "safetensors") {
        if let Some(key) = checkpoint.state_dict_key {
            return Err(load_error(format!(
                "{} is a safetensors file, which has no nested state dict '{key}'",
                path.display()
            )));
        }
        let mut file = checkpoint.renames.iter().fold(
            SafetensorsStore::from_file(&path).with_from_adapter(PyTorchToBurnAdapter),
            |file, (from, to)| file.with_key_remapping(from, *to),
        );
        module
            .load_from(&mut file)
            .map_err(|e| load_error(e.to_string()))?
    } else {
        let file = PytorchStore::from_file(&path);
        let file = match checkpoint.state_dict_key {
            Some(key) => file.with_top_level_key(key),
            None => file,
        };
        let mut file = checkpoint
            .renames
            .iter()
            .fold(file, |file, (from, to)| file.with_key_remapping(from, *to));
        module
            .load_from(&mut file)
            .map_err(|e| load_error(e.to_string()))?
    };
    let unused: Vec<&String> = result
        .unused
        .iter()
        .filter(|path| {
            !path.ends_with(".num_batches_tracked") && !checkpoint.unused.contains(&path.as_str())
        })
        .collect();
    check_applied(model, &result, &unused)?;
    Ok(module)
}

fn check_applied(model: &str, result: &ApplyResult, unused: &[&String]) -> Result<(), ModelError> {
    if result.missing.is_empty() && result.errors.is_empty() && unused.is_empty() {
        return Ok(());
    }
    Err(ModelError::Load {
        model: model.to_string(),
        message: format!(
            "{} missing tensors (first: {:?}), {} unused (first: {:?}), {} errors (first: {:?})",
            result.missing.len(),
            result.missing.first().map(|m| &m.0),
            unused.len(),
            unused.first(),
            result.errors.len(),
            result.errors.first().map(|e| e.to_string()),
        ),
    })
}

/// A tensor of `shape` from flat values, or an input error naming `model` when
/// the count does not match.
pub fn tensor_from_f32<const D: usize>(
    data: Vec<f32>,
    shape: [usize; D],
    device: &Device,
    model: &'static str,
) -> Result<Tensor<D>, ModelError> {
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(ModelError::InvalidInput {
            model,
            message: format!(
                "expected {expected} values for shape {shape:?}, got {}",
                data.len()
            ),
        });
    }
    Ok(Tensor::from_data(TensorData::new(data, shape), device))
}

/// A `[batch, channels, height, width]` tensor from flat values, the batch
/// being however many whole items `input` holds (at least one).
pub(crate) fn batch_from_f32(
    input: Vec<f32>,
    [channels, height, width]: [usize; 3],
    device: &Device,
    model: &'static str,
) -> Result<Tensor<4>, ModelError> {
    let batch = input.len() / (channels * height * width);
    if batch == 0 {
        return Err(ModelError::InvalidInput {
            model,
            message: format!(
                "expected at least one {channels}x{height}x{width} input, got {} values",
                input.len()
            ),
        });
    }
    tensor_from_f32(input, [batch, channels, height, width], device, model)
}

/// A tensor's values as float32, or an output error naming `model`.
pub fn read_f32<const D: usize>(
    tensor: Tensor<D>,
    model: &'static str,
) -> Result<Vec<f32>, ModelError> {
    let output = |message: String| ModelError::Output { model, message };
    let data = tensor
        .try_into_data()
        .map_err(|e| output(e.to_string()))?
        .try_cast(burn::tensor::DType::F32)
        .map_err(|e| output(format!("{e:?}")))?;
    data.as_slice::<f32>()
        .map(<[f32]>::to_vec)
        .map_err(|e| output(format!("{e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(hardware: bool, max_buffer_bytes: u64) -> Result<GpuProbe, String> {
        Ok(GpuProbe {
            hardware,
            name: "Test GPU".to_string(),
            max_buffer_bytes,
        })
    }

    #[test]
    fn hardware_gpu_with_room_is_used() {
        assert_eq!(
            choose(gpu(true, 2 << 30)),
            DeviceChoice::Gpu {
                name: "Test GPU".to_string()
            }
        );
        assert!(matches!(
            choose(gpu(true, REQUIRED_GPU_BUFFER_BYTES)),
            DeviceChoice::Gpu { .. }
        ));
    }

    #[test]
    fn software_rasterizer_falls_back_to_cpu() {
        // llvmpipe: a "Cpu" wgpu adapter with 128 MiB storage bindings.
        assert!(matches!(
            choose(gpu(false, 128 << 20)),
            DeviceChoice::Cpu { .. }
        ));
    }

    #[test]
    fn gpu_too_small_for_the_largest_buffer_falls_back_to_cpu() {
        let choice = choose(gpu(true, 128 << 20));
        assert!(matches!(choice, DeviceChoice::Cpu { .. }), "{choice}");
        assert!(choice.to_string().contains("128 MiB"), "{choice}");
    }

    #[test]
    fn wgpu_batches_default_to_ours_unless_the_environment_sets_them() {
        assert_eq!(tasks_max_override(None), Some(WGPU_TASKS_MAX));
        assert_eq!(tasks_max_override(Some("32".to_string())), None);
    }

    #[test]
    fn failed_gpu_initialization_falls_back_to_cpu_with_the_reason() {
        let choice = choose(Err("driver lost".to_string()));
        assert_eq!(
            choice.to_string(),
            "CPU, because GPU initialization failed: driver lost"
        );
    }
}
