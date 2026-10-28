//! Tile foundation models: Vision Transformers that turn a tissue tile into one
//! embedding vector, loaded unchanged from their published checkpoints.
//!
//! Each model is an [`Encoder`] entry: where its weights are published, its
//! architecture, the input it was trained on and how its embedding is pooled.
//! [`TileEncoder`] runs any entry. The weights are not distributed with this
//! crate; `MODELS-NOTICE.md` lists where to download each one and its terms.

mod clip;
mod conch;
mod conch_v1;

use crate::preprocess::rgb_to_nchw;
use crate::runtime::{Checkpoint, batch_from_f32, load_pytorch_weights, read_f32};
use crate::store::{ModelError, ModelStore};
use burn::prelude::*;
use vit_burn::{MlpKind, PosResample, VisionTransformer, VitConfig, naming};

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Which library saved the checkpoint, and so how its keys are named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Naming {
    Timm,
    /// transformers `Dinov2Model`.
    HfDinov2,
    /// transformers `ViTModel`.
    HfVit,
    /// transformers `CLIPVisionModel` with a projection.
    HfClip,
}

/// How the output tokens become one embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// The class token.
    ClassToken,
    /// The class token, then the mean of the patch tokens (register tokens
    /// excluded): twice the token width.
    ClassAndPatchMean,
    /// CONCH v1.5's learned query over all tokens, then LayerNorm (768 values).
    ConchAttention,
    /// CONCH v1's unprojected, unnormalized 512-dimensional features.
    ConchV1,
    /// CLIP's post-normalized class token and learned 512-dimensional projection.
    ClipProjection,
}

/// One published tile encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Encoder {
    /// The model's name in the weights manifest.
    pub name: &'static str,
    /// Where the checkpoint is published.
    pub upstream: &'static str,
    pub vit: VitConfig,
    pub naming: Naming,
    /// Tile side in pixels.
    pub tile: usize,
    /// Microns per pixel the tiles should be read at.
    pub mpp: f64,
    /// Normalization of RGB in [0, 1].
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub pooling: Pooling,
}

impl Encoder {
    pub fn embedding_dim(&self) -> usize {
        match self.pooling {
            Pooling::ClassToken => self.vit.embed_dim,
            Pooling::ClassAndPatchMean => 2 * self.vit.embed_dim,
            Pooling::ConchAttention => 768,
            Pooling::ConchV1 | Pooling::ClipProjection => 512,
        }
    }

    fn checkpoint(&self) -> Checkpoint<'static> {
        let (renames, unused): (&'static [(&str, &str)], &'static [&str]) = match self.naming {
            Naming::Timm => (&[], &[]),
            Naming::HfDinov2 => (naming::HF_DINOV2, naming::HF_DINOV2_UNUSED),
            Naming::HfVit => (naming::HF_VIT, naming::HF_VIT_UNUSED),
            Naming::HfClip => (clip::RENAMES, &[]),
        };
        Checkpoint {
            state_dict_key: None,
            renames,
            unused,
        }
    }
}

/// UNI (Mahmood Lab): ViT-L/16 trained with DINOv2 on Mass-100K at 20x.
pub const UNI: Encoder = Encoder {
    name: "uni",
    upstream: "https://huggingface.co/MahmoodLab/UNI",
    vit: VitConfig {
        layer_scale: true,
        ..VitConfig::LARGE_PATCH16_224
    },
    naming: Naming::Timm,
    tile: 224,
    mpp: 0.5,
    mean: IMAGENET_MEAN,
    std: IMAGENET_STD,
    pooling: Pooling::ClassToken,
};

/// UNI2-h (Mahmood Lab): ViT-H/14 with SwiGLU and eight register tokens.
pub const UNI2_H: Encoder = Encoder {
    name: "uni2_h",
    upstream: "https://huggingface.co/MahmoodLab/UNI2-h",
    vit: VitConfig {
        embed_dim: 1536,
        depth: 24,
        num_heads: 24,
        patch_size: 14,
        mlp_hidden: 8192,
        mlp: MlpKind::SwiGlu,
        reg_tokens: 8,
        no_embed_class: true,
        layer_scale: true,
        ..VitConfig::BASE_PATCH16_224
    },
    ..UNI
};

/// Virchow (Paige): ViT-H/14 with SwiGLU; the class token and the patch mean.
pub const VIRCHOW: Encoder = Encoder {
    name: "virchow",
    upstream: "https://huggingface.co/paige-ai/Virchow",
    vit: VitConfig {
        mlp_hidden: 6832,
        mlp: MlpKind::SwiGlu,
        layer_scale: true,
        ..VitConfig::HUGE_PATCH14_224
    },
    pooling: Pooling::ClassAndPatchMean,
    ..UNI
};

/// Virchow2 (Paige): Virchow with four register tokens, which the position
/// embedding also covers.
pub const VIRCHOW2: Encoder = Encoder {
    name: "virchow2",
    upstream: "https://huggingface.co/paige-ai/Virchow2",
    vit: VitConfig {
        reg_tokens: 4,
        ..VIRCHOW.vit
    },
    ..VIRCHOW
};

/// H-optimus-0 (Bioptimus): DINOv2's ViT-g/14 with four registers, at 224 px.
pub const H_OPTIMUS_0: Encoder = Encoder {
    name: "h_optimus_0",
    upstream: "https://huggingface.co/bioptimus/H-optimus-0",
    vit: VitConfig {
        img_size: 224,
        ..VitConfig::GIANT_PATCH14_REG4_DINOV2
    },
    mean: [0.707_223, 0.578_729, 0.703_617],
    std: [0.211_883, 0.230_117, 0.177_517],
    ..UNI
};

/// H-optimus-1 (Bioptimus): H-optimus-0's architecture, trained on more data.
pub const H_OPTIMUS_1: Encoder = Encoder {
    name: "h_optimus_1",
    upstream: "https://huggingface.co/bioptimus/H-optimus-1",
    ..H_OPTIMUS_0
};

/// Prov-GigaPath tile encoder (Providence, Microsoft): DINOv2's ViT-g with
/// 16 px patches.
pub const PROV_GIGAPATH: Encoder = Encoder {
    name: "prov_gigapath",
    upstream: "https://huggingface.co/prov-gigapath/prov-gigapath",
    vit: VitConfig {
        patch_size: 16,
        img_size: 224,
        ..VitConfig::GIANT_PATCH14_DINOV2
    },
    ..UNI
};

/// `vit` as transformers saves and runs it: query, key and value as separate
/// layers, and its own position-embedding resize.
const fn transformers(vit: VitConfig) -> VitConfig {
    VitConfig {
        fused_qkv: false,
        pos_resample: PosResample::Bicubic,
        ..vit
    }
}

/// Phikon (Owkin): ViT-B/16 trained with iBOT, saved by transformers.
pub const PHIKON: Encoder = Encoder {
    name: "phikon",
    upstream: "https://huggingface.co/owkin/phikon",
    vit: transformers(VitConfig::BASE_PATCH16_224),
    naming: Naming::HfVit,
    ..UNI
};

/// Phikon-v2 (Owkin): ViT-L/16 trained with DINOv2, saved by transformers.
pub const PHIKON_V2: Encoder = Encoder {
    name: "phikon_v2",
    upstream: "https://huggingface.co/owkin/phikon-v2",
    vit: transformers(VitConfig {
        layer_scale: true,
        ..VitConfig::LARGE_PATCH16_224
    }),
    naming: Naming::HfDinov2,
    ..UNI
};

/// Midnight-12k (kaiko.ai): DINOv2's ViT-g/14, saved by transformers with its
/// 518 px position embedding, which 224 px tiles resample. The card states no
/// resolution; 0.5 µm/px matches the other encoders.
pub const MIDNIGHT_12K: Encoder = Encoder {
    name: "midnight_12k",
    upstream: "https://huggingface.co/kaiko-ai/midnight",
    vit: transformers(VitConfig::GIANT_PATCH14_DINOV2),
    naming: Naming::HfDinov2,
    mean: [0.5; 3],
    std: [0.5; 3],
    pooling: Pooling::ClassAndPatchMean,
    ..UNI
};

/// CONCH v1.5 (Mahmood Lab): ViT-L/16 at 448 px with attention pooling.
pub const CONCH_V1_5: Encoder = Encoder {
    name: "conch_v1_5",
    upstream: "https://huggingface.co/MahmoodLab/TITAN",
    vit: VitConfig {
        pos_resample: PosResample::Dino,
        ..UNI.vit
    },
    tile: 448,
    pooling: Pooling::ConchAttention,
    ..UNI
};

/// CONCH v1: `encode_image(proj_contrast=False, normalize=False)`.
pub const CONCH_V1: Encoder = Encoder {
    name: "conch_v1",
    upstream: "https://huggingface.co/MahmoodLab/CONCH",
    vit: VitConfig {
        img_size: 448,
        ..VitConfig::BASE_PATCH16_224
    },
    tile: 448,
    mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
    std: [0.268_629_54, 0.261_302_58, 0.275_777_11],
    pooling: Pooling::ConchV1,
    ..UNI
};

/// PLIP's CLIP ViT-B/32: projected image features, without L2 normalization.
pub const PLIP: Encoder = Encoder {
    name: "plip",
    upstream: "https://huggingface.co/vinid/plip",
    vit: VitConfig {
        patch_size: 32,
        mlp: MlpKind::QuickGelu,
        fused_qkv: false,
        norm_eps: 1e-5,
        ..VitConfig::BASE_PATCH16_224
    },
    naming: Naming::HfClip,
    tile: 224,
    pooling: Pooling::ClipProjection,
    ..CONCH_V1
};

/// Every tile encoder, for listing and tests.
pub const ALL: [&Encoder; 13] = [
    &UNI,
    &UNI2_H,
    &VIRCHOW,
    &VIRCHOW2,
    &H_OPTIMUS_0,
    &H_OPTIMUS_1,
    &PROV_GIGAPATH,
    &PHIKON,
    &PHIKON_V2,
    &MIDNIGHT_12K,
    &CONCH_V1_5,
    &CONCH_V1,
    &PLIP,
];

enum Network {
    Vit(VisionTransformer),
    Conch(conch::Conch),
    ConchV1(conch_v1::ConchV1),
    Clip(clip::Clip),
}

impl Network {
    fn tokens(&self, input: Tensor<4>) -> Tensor<3> {
        match self {
            Self::Vit(model) => model.forward(input),
            Self::Conch(model) => model.trunk.forward(input),
            Self::ConchV1(model) => model.tokens(input),
            Self::Clip(model) => model.tokens(input),
        }
    }
}

/// A tile encoder's network on one device.
pub struct TileEncoder {
    encoder: Encoder,
    model: Network,
    device: Device,
}

impl TileEncoder {
    pub fn load(
        store: &ModelStore,
        device: &Device,
        encoder: &Encoder,
    ) -> Result<Self, ModelError> {
        let naming = encoder.checkpoint();
        let unused = match encoder.pooling {
            Pooling::ConchV1 => conch_v1::unused_keys(),
            Pooling::ClipProjection => clip::unused_keys(),
            _ => Vec::new(),
        };
        let unused: Vec<&str> = unused.iter().map(String::as_str).collect();
        let checkpoint = Checkpoint {
            unused: if unused.is_empty() {
                naming.unused
            } else {
                &unused
            },
            ..naming
        };
        let model = match encoder.pooling {
            Pooling::ConchV1 => Network::ConchV1(load_pytorch_weights(
                conch_v1::ConchV1::new(&encoder.vit, device),
                store,
                encoder.name,
                &checkpoint,
            )?),
            Pooling::ClipProjection => Network::Clip(load_pytorch_weights(
                clip::Clip::new(&encoder.vit, device),
                store,
                encoder.name,
                &checkpoint,
            )?),
            Pooling::ConchAttention => Network::Conch(load_pytorch_weights(
                conch::Conch::new(&encoder.vit, device),
                store,
                encoder.name,
                &checkpoint,
            )?),
            _ => Network::Vit(load_pytorch_weights(
                encoder.vit.init(device),
                store,
                encoder.name,
                &checkpoint,
            )?),
        };
        Ok(Self {
            encoder: *encoder,
            model,
            device: device.clone(),
        })
    }

    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// HWC RGB u8 tiles, `tile x tile` each and concatenated, to the
    /// normalized NCHW batch: `(x / 255 - mean) / std`.
    pub fn normalize_rgb(&self, rgb: &[u8]) -> Result<Vec<f32>, ModelError> {
        let Encoder {
            name,
            tile,
            mean,
            std,
            ..
        } = self.encoder;
        rgb_to_nchw(rgb, tile, None, name, |c, x| {
            (x as f32 / 255.0 - mean[c]) / std[c]
        })
    }

    /// One embedding of [`Encoder::embedding_dim`] values per tile, for a
    /// normalized NCHW batch. The whole batch runs at once, so the caller
    /// bounds its size.
    pub fn embed(&self, input: Vec<f32>) -> Result<Vec<f32>, ModelError> {
        let tokens = self.forward(input)?;
        read_f32(self.pool(tokens), self.encoder.name)
    }

    /// Every output token, `[batch, prefix + patches, embed]` flattened, for
    /// parity checks and dense (per-patch) tasks. PLIP returns CLIP's last hidden
    /// state before the class token's post-norm; other entries include final norm.
    pub fn tokens(&self, input: Vec<f32>) -> Result<Vec<f32>, ModelError> {
        let tokens = self.forward(input)?;
        read_f32(tokens, self.encoder.name)
    }

    fn forward(&self, input: Vec<f32>) -> Result<Tensor<3>, ModelError> {
        let Encoder { name, tile, .. } = self.encoder;
        let input = batch_from_f32(input, [3, tile, tile], &self.device, name)?;
        Ok(self.model.tokens(input))
    }

    fn pool(&self, tokens: Tensor<3>) -> Tensor<2> {
        let [batch, count, embed] = tokens.dims();
        let class = tokens
            .clone()
            .slice([0..batch, 0..1, 0..embed])
            .reshape([batch, embed]);
        match (self.encoder.pooling, &self.model) {
            (Pooling::ConchAttention, Network::Conch(model)) => model.pool(tokens),
            (Pooling::ConchV1, Network::ConchV1(model)) => model.pool(tokens),
            (Pooling::ClipProjection, Network::Clip(model)) => model.pool(tokens),
            (Pooling::ClassToken, _) => class,
            (Pooling::ClassAndPatchMean, _) => {
                let prefix = self.encoder.vit.prefix_tokens();
                let patches = tokens
                    .slice([0..batch, prefix..count, 0..embed])
                    .mean_dim(1)
                    .reshape([batch, embed]);
                Tensor::cat(vec![class, patches], 1)
            }
            _ => unreachable!("pooling needs the matching learned head"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_consistent() {
        let mut names: Vec<&str> = ALL.iter().map(|e| e.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL.len(), "duplicate encoder names");
        for encoder in ALL {
            let vit = &encoder.vit;
            assert!(
                vit.class_token,
                "{}: pooling needs a class token",
                encoder.name
            );
            assert_eq!(vit.embed_dim % vit.num_heads, 0, "{}", encoder.name);
            assert_eq!(encoder.tile % vit.patch_size, 0, "{}", encoder.name);
            assert_eq!(
                vit.fused_qkv,
                encoder.naming == Naming::Timm,
                "{}: transformers checkpoints keep q, k and v separate",
                encoder.name
            );
            assert!(encoder.upstream.starts_with("https://"), "{}", encoder.name);
        }
    }

    #[test]
    fn architectures_match_the_model_cards() {
        // Output widths the cards document.
        let widths: Vec<(&str, usize)> = ALL.iter().map(|e| (e.name, e.embedding_dim())).collect();
        assert_eq!(
            widths,
            [
                ("uni", 1024),
                ("uni2_h", 1536),
                ("virchow", 2560),
                ("virchow2", 2560),
                ("h_optimus_0", 1536),
                ("h_optimus_1", 1536),
                ("prov_gigapath", 1536),
                ("phikon", 768),
                ("phikon_v2", 1024),
                ("midnight_12k", 3072),
                ("conch_v1_5", 768),
                ("conch_v1", 512),
                ("plip", 512),
            ]
        );
        // Position-embedding rows: Virchow2 covers its registers, UNI2-h does not.
        let rows = |e: &Encoder| {
            let prefix = if e.vit.no_embed_class {
                0
            } else {
                e.vit.prefix_tokens()
            };
            prefix + e.vit.grid() * e.vit.grid()
        };
        assert_eq!(rows(&VIRCHOW2), 5 + 256);
        assert_eq!(rows(&UNI2_H), 256);
        assert_eq!(rows(&PHIKON_V2), 197);
        assert_eq!(rows(&MIDNIGHT_12K), 1 + 37 * 37);
        assert_eq!(rows(&CONCH_V1_5), 197);
        assert_eq!(CONCH_V1_5.tile, 448);
    }
}
