//! CellViT: the networks and their checkpoint entries. [`crate::hover`]
//! decodes the predicted maps into nuclei.

mod model;
mod native;

pub use model::{
    CellVit, CellVitPrediction, CellVitRaw, CellVitWeights, NUCLEI_CLASSES, SAM_H_X40,
    TISSUE_CLASSES, VIT256_X40,
};
pub use native::Backbone;
