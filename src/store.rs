//! Locating and verifying weight files.
//!
//! A model directory holds `manifest.json` (written by the scripts in `export/`)
//! mapping each model name to its checkpoint file and SHA-256. Weights are hashed
//! once per process before first use.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Environment variable that overrides model directory discovery.
pub const MODEL_DIR_ENV: &str = "PATH_2_BURN_MODEL_DIR";
const MANIFEST: &str = "manifest.json";

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("no model directory with {MANIFEST} found (searched: {searched})")]
    NoModelDir { searched: String },
    #[error("read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Manifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("model '{0}' is not listed in the model manifest")]
    UnknownModel(String),
    #[error(
        "weights for '{model}' failed verification: expected sha256 {expected}, found {actual}"
    )]
    ChecksumMismatch {
        model: String,
        expected: String,
        actual: String,
    },
    #[error("load weights for '{model}': {message}")]
    Load { model: String, message: String },
    #[error("invalid input for '{model}': {message}")]
    InvalidInput {
        model: &'static str,
        message: String,
    },
    #[error("read output of '{model}': {message}")]
    Output {
        model: &'static str,
        message: String,
    },
}

impl ModelError {
    /// An I/O failure on `path`, for `map_err`.
    pub(crate) fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Self + '_ {
        move |source| Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestEntry {
    file: String,
    sha256: String,
}

/// A verified directory of model weights.
#[derive(Debug)]
pub struct ModelStore {
    dir: PathBuf,
    entries: HashMap<String, ManifestEntry>,
    verified: Mutex<HashSet<String>>,
}

impl ModelStore {
    /// Open a directory containing `manifest.json`.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, ModelError> {
        let dir = dir.into();
        let path = dir.join(MANIFEST);
        let raw = std::fs::read_to_string(&path).map_err(ModelError::io(&path))?;
        let entries =
            serde_json::from_str(&raw).map_err(|source| ModelError::Manifest { path, source })?;
        Ok(Self {
            dir,
            entries,
            verified: Mutex::new(HashSet::new()),
        })
    }

    /// Open the first candidate directory that has a manifest.
    pub fn discover(candidates: &[PathBuf]) -> Result<Self, ModelError> {
        for dir in candidates {
            if dir.join(MANIFEST).is_file() {
                return Self::open(dir.clone());
            }
        }
        let searched = candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Err(ModelError::NoModelDir { searched })
    }

    /// Search order: `$PATH_2_BURN_MODEL_DIR`, then `~/.path-2-burn/models`. An
    /// application with its own layout passes its directories to [`discover`](Self::discover).
    pub fn default_candidates() -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        if let Some(dir) = std::env::var_os(MODEL_DIR_ENV) {
            dirs.push(PathBuf::from(dir));
        }
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            dirs.push(PathBuf::from(home).join(".path-2-burn").join("models"));
        }
        dirs
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn contains(&self, model: &str) -> bool {
        self.entries.contains_key(model)
    }

    /// Path to a model's weights, hashed against the manifest on first request.
    pub fn weights(&self, model: &str) -> Result<PathBuf, ModelError> {
        let entry = self
            .entries
            .get(model)
            .ok_or_else(|| ModelError::UnknownModel(model.to_string()))?;
        let path = self.dir.join(&entry.file);
        let mut verified = self
            .verified
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !verified.contains(model) {
            let actual = sha256_file(&path)?;
            if !actual.eq_ignore_ascii_case(&entry.sha256) {
                return Err(ModelError::ChecksumMismatch {
                    model: model.to_string(),
                    expected: entry.sha256.clone(),
                    actual,
                });
            }
            verified.insert(model.to_string());
        }
        Ok(path)
    }
}

fn sha256_file(path: &Path) -> Result<String, ModelError> {
    let mut file = std::fs::File::open(path).map_err(ModelError::io(path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = file.read(&mut buf).map_err(ModelError::io(path))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("path-2-burn-store-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn verifies_weights_against_manifest() {
        let dir = temp_dir("ok");
        std::fs::write(dir.join("m.bpk"), b"weights").unwrap();
        let digest = sha256_file(&dir.join("m.bpk")).unwrap();
        std::fs::write(
            dir.join(MANIFEST),
            format!(r#"{{"m":{{"file":"m.bpk","sha256":"{digest}"}}}}"#),
        )
        .unwrap();
        let store = ModelStore::open(&dir).unwrap();
        assert_eq!(store.weights("m").unwrap(), dir.join("m.bpk"));
        assert!(matches!(
            store.weights("other"),
            Err(ModelError::UnknownModel(_))
        ));
    }

    #[test]
    fn rejects_tampered_weights() {
        let dir = temp_dir("bad");
        std::fs::write(dir.join("m.bpk"), b"weights").unwrap();
        std::fs::write(
            dir.join(MANIFEST),
            r#"{"m":{"file":"m.bpk","sha256":"00"}}"#,
        )
        .unwrap();
        let store = ModelStore::open(&dir).unwrap();
        assert!(matches!(
            store.weights("m"),
            Err(ModelError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn discover_reports_searched_dirs() {
        let missing = temp_dir("missing").join("nope");
        let err = ModelStore::discover(std::slice::from_ref(&missing)).unwrap_err();
        assert!(err.to_string().contains(&missing.display().to_string()));
    }
}
