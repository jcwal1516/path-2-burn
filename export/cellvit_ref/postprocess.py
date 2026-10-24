"""Minimal CellViT++ postprocessor extracted from the vendored HoverNet pipeline.

Adapted from:
  backend/segmentation/cellvit_model/post_proc_cellvit.py
  backend/segmentation/cellvit_model/post_proc_tools.py

Original source: https://github.com/TIO-IKIM/CellViT (MIT + Apache 2.0)
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed


# ---------------------------------------------------------------------------
# Utilities (from post_proc_tools.py)
# ---------------------------------------------------------------------------


def get_bounding_box(img: np.ndarray) -> list[int]:
    """Get bounding box [rmin, rmax, cmin, cmax] for a binary mask."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def remove_small_objects(
    pred: np.ndarray, min_size: int = 64, connectivity: int = 1,
) -> np.ndarray:
    """Remove connected components smaller than *min_size*."""
    out = pred
    if min_size == 0:
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    component_sizes = np.bincount(ccs.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    return out


# ---------------------------------------------------------------------------
# DetectionCellPostProcessor
# ---------------------------------------------------------------------------


class DetectionCellPostProcessor:
    """HoverNet-style watershed postprocessor for CellViT++ prediction maps.

    Converts combined (type, nuclei_prob, hv_x, hv_y) maps into an instance
    segmentation map and per-instance metadata (bbox, centroid, contour, type).
    """

    def __init__(
        self,
        nr_types: int | None = None,
        magnification: Literal[20, 40] = 40,
        gt: bool = False,
    ) -> None:
        self.nr_types = nr_types
        self.magnification = magnification
        self.gt = gt

        if magnification == 40:
            self.object_size = 10
            self.k_size = 21
        elif magnification == 20:
            self.object_size = 3
            self.k_size = 11
        else:
            raise NotImplementedError(f"Unknown magnification: {magnification}")

        if gt:
            self.object_size = 100
            self.k_size = 21

    # ----- public API -----

    def post_process_cell_segmentation(
        self, pred_map: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Post-process one image tile.

        Args:
            pred_map: Shape (H, W, 4) — [type_argmax, nuclei_prob, hv_x, hv_y].

        Returns:
            (instance_map, inst_info_dict) where instance_map is (H, W) int
            and inst_info_dict maps instance_id -> {bbox, centroid, contour,
            type, type_prob}.
        """
        if self.nr_types is not None:
            pred_type = pred_map[..., :1].astype(np.int32)
            pred_inst = pred_map[..., 1:]
        else:
            pred_inst = pred_map

        pred_inst = np.squeeze(pred_inst)
        pred_inst = self._proc_np_hv(
            pred_inst, object_size=self.object_size, ksize=self.k_size,
        )

        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        inst_info_dict: dict = {}

        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[rmin:rmax, cmin:cmax].astype(np.uint8)

            inst_moment = cv2.moments(inst_map)
            if inst_moment["m00"] == 0:
                continue

            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
            )
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue

            inst_centroid = np.array([
                inst_moment["m10"] / inst_moment["m00"],
                inst_moment["m01"] / inst_moment["m00"],
            ])
            inst_contour[:, 0] += inst_bbox[0][1]  # X offset
            inst_contour[:, 1] += inst_bbox[0][0]  # Y offset
            inst_centroid[0] += inst_bbox[0][1]
            inst_centroid[1] += inst_bbox[0][0]

            inst_info_dict[inst_id] = {
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

        # Assign cell types from the type prediction channel
        if self.nr_types is not None:
            for inst_id in list(inst_info_dict.keys()):
                rmin, cmin, rmax, cmax = inst_info_dict[inst_id]["bbox"].flatten()
                inst_map_crop = (pred_inst[rmin:rmax, cmin:cmax] == inst_id)
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
                inst_type = inst_type_crop[inst_map_crop]
                type_list, type_pixels = np.unique(inst_type, return_counts=True)
                type_list = sorted(
                    zip(type_list, type_pixels), key=lambda x: x[1], reverse=True,
                )
                chosen_type = type_list[0][0]
                if chosen_type == 0 and len(type_list) > 1:
                    chosen_type = type_list[1][0]
                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[chosen_type] / (np.sum(inst_map_crop) + 1e-6)
                inst_info_dict[inst_id]["type"] = int(chosen_type)
                inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        return pred_inst, inst_info_dict

    # ----- private -----

    def _proc_np_hv(
        self, pred: np.ndarray, object_size: int = 10, ksize: int = 21,
    ) -> np.ndarray:
        """Watershed instance segmentation from nuclei probability + HV maps."""
        pred = np.array(pred, dtype=np.float32)

        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1

        h_dir = cv2.normalize(
            h_dir_raw, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
        )

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - cv2.normalize(
            sobelh, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
        )
        sobelv = 1 - cv2.normalize(
            sobelv, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        return watershed(dist, markers=marker, mask=blb)
