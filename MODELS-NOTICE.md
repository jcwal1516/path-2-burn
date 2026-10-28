# Model weights notice

The weight files installed next to this notice come from third-party
checkpoints: unchanged PyTorch checkpoints, weights-only copies of them, or
safetensors and JSON extracted from them. They keep the terms of their sources.
These packages are built for local use. Check every license below before you
share a package that contains them.

| Weights | Source | Terms |
|---|---|---|
| `CellViT-SAM-H-x40-AMP.pth` | Unchanged TIO-IKIM CellViT / CellViT++ checkpoint, PanNuke nuclei and tissue heads | SAM-derived weights: Apache-2.0. CellViT segmentation-head code: Apache-2.0 with Commons Clause (no selling) and required citation. |
| `CellViT-256-x40.pth` | Unchanged TIO-IKIM CellViT checkpoint (Google Drive link in [TIO-IKIM/CellViT](https://github.com/TIO-IKIM/CellViT)'s README), PanNuke heads on HIPT's ViT-S/16 | HIPT-derivative weights: Apache-2.0 with Commons Clause (Mahmood Lab). CellViT segmentation-head code: as above. |
| `cellpose_sam/cpsam` | Unchanged [mouseland/cellpose-sam](https://huggingface.co/mouseland/cellpose-sam) checkpoint, fetched by `export/reference_cellpose.py download` | BSD-3-Clause |
| `hovernet_*` | Unchanged vqdang/hover_net checkpoints (README, Google Drive), you download them | Code MIT. The PanNuke weights carry PanNuke's CC BY-NC-SA 4.0; check each dataset's terms for the others. |
| `instanseg_net.safetensors`, `instanseg_postprocess.json` | Extracted from your copy of InstanSeg `brightfield_nuclei` 0.1.1 (instanseg-torch) by `export/export_instanseg.py` | Apache-2.0 |
| `stardist_*/model.safetensors` | Convolution tensors extracted from the three [QuPath StarDist graphs](https://github.com/qupath/models/tree/main/stardist), trained by Uwe Schmidt and Martin Weigert | BSD-3-Clause; see upstream model and training-data notices |

## Tile encoders

These checkpoints are not distributed with path-2-burn. `export/reference_encoders.py
download` fetches each one unchanged from its Hugging Face page at a pinned revision,
using your own account: gated models need you to accept their terms on the page first.

| Model | Upstream | File | Terms | Access |
|---|---|---|---|---|
| UNI | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | `pytorch_model.bin` | CC-BY-NC-ND 4.0 | Gated; institutional email |
| UNI2-h | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | `pytorch_model.bin` | CC-BY-NC-ND 4.0 | Gated; institutional email |
| Virchow | [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) | `model.safetensors` | Apache-2.0, with Paige's use guidelines | Gated |
| Virchow2 | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) | `model.safetensors` | CC-BY-NC-ND 4.0 | Gated; institutional email |
| H-optimus-0 | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | `pytorch_model.bin` | Apache-2.0 | Gated |
| H-optimus-1 | [bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | `model.safetensors` | CC-BY-NC-ND 4.0 | Gated; manual approval |
| Prov-GigaPath (tile) | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | `pytorch_model.bin` | Apache-2.0 | Gated |
| Phikon | [owkin/phikon](https://huggingface.co/owkin/phikon) | `model.safetensors` | Owkin non-commercial | Public |
| Phikon-v2 | [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2) | `model.safetensors` | Owkin non-commercial | Public |
| Midnight-12k | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) | `model.safetensors` | MIT | Public |
| CONCH v1.5 | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) | `conch_v1_5_pytorch_model.bin` | CC-BY-NC-ND 4.0; non-commercial academic research; no redistribution of weights or derived datasets | Gated; institutional email |
| CONCH v1 | [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) | `pytorch_model.bin` | CC-BY-NC-ND 4.0; non-commercial academic research; no redistribution of weights or derived datasets | Gated; institutional email |
| PLIP | [vinid/plip](https://huggingface.co/vinid/plip) | `pytorch_model.bin` | No explicit weight license listed; see the model card's research-use and deployment restrictions | Public |

Non-commercial and no-derivatives terms apply to the weights and to what you build from
them; they do not change the license of this crate's code. Each model card asks for a
citation; see the card.

## Slide encoders

| Model | Upstream | File | Terms | Access |
|---|---|---|---|---|
| TITAN | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) | `model.safetensors` | CC-BY-NC-ND 4.0; non-commercial academic research; no redistribution of weights or derived datasets | Gated; institutional email |

TITAN's slide encoder loads its vision weights from the original checkpoint;
the unused text tower, captioning pooler and vision-language projection remain
in the file. Each user obtains their own checkpoint under the upstream terms.

## Required citations

- Hörst, F. et al. (2023). CellViT: Vision Transformers for precise cell
  segmentation and classification. https://doi.org/10.48550/ARXIV.2306.15350
- Hörst, F. et al. (2025). CellViT++: Energy-Efficient and Adaptive Cell
  Segmentation and Classification Using Foundation Models.
  https://doi.org/10.48550/ARXIV.2501.05269
- Graham, S. et al. (2019). HoVer-Net: Simultaneous segmentation and
  classification of nuclei in multi-tissue histology images.
  https://doi.org/10.1016/j.media.2019.101563
- Pachitariu, M., Rariden, M. & Stringer, C. (2025). Cellpose-SAM: superhuman
  generalization for cellular segmentation. https://doi.org/10.1101/2025.04.28.651001
- Goldsborough, T. et al. (2024). InstanSeg: an embedding-based instance
  segmentation algorithm optimized for accurate, efficient and portable cell
  segmentation. https://doi.org/10.48550/arXiv.2408.15954
- Ding, T., Wagner, S. J., Song, A. H., Chen, R. J. et al. (2025).
  A multimodal whole-slide foundation model for pathology. Nature Medicine.
  https://doi.org/10.1038/s41591-025-03982-3
- Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. (2018).
  Cell Detection with Star-convex Polygons. MICCAI.
  https://doi.org/10.1007/978-3-030-00934-2_30
- Lu, M. Y., Chen, B., Williamson, D. F. K. et al. (2024).
  A visual-language foundation model for computational pathology. Nature Medicine.
  https://doi.org/10.1038/s41591-024-02856-4
- Huang, Z., Bianchi, F., Yuksekgonul, M., Montine, T. J. & Zou, J. (2023).
  A visual–language foundation model for pathology image analysis using medical
  Twitter. Nature Medicine. https://doi.org/10.1038/s41591-023-02504-3
