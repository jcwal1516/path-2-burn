"""Extract StarDist's QuPath graph weights and write OpenCV/NMS references.

Obtain the three frozen graphs from https://github.com/qupath/models/tree/main/stardist.
`export --graphs DIR --weights WEIGHTS` extracts unchanged convolution tensors
into safetensors and registers them. `reference --graphs DIR --out OUT` uses the
public H&E and fluorescence samples bundled with the StarDist Python package.
TensorFlow is not needed. Generated weights and outputs are never distributed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import stardist
from csbdeep.utils import normalize
from safetensors.numpy import save_file
from stardist.data import test_image_he_2d, test_image_nuclei_2d
from stardist.nms import non_maximum_suppression

from common import register, write_json

# name: (published graph, channels, prediction grid spacing)
MODELS = {
    "stardist_he": ("he_heavy_augment.pb", 3, 2),
    "stardist_fluorescence": ("dsb2018_heavy_augment.pb", 1, 2),
    "stardist_fluorescence_paper": ("dsb2018_paper.pb", 1, 1),
}
PROB_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def export(graphs: Path, weights: Path) -> None:
    for name, (file, _, _) in MODELS.items():
        net = cv2.dnn.readNetFromTensorflow(str(graphs / file))
        tensors = {}
        for layer_name in net.getLayerNames():
            layer = net.getLayer(net.getLayerId(layer_name))
            if layer.type == "Convolution":
                weight, bias = layer.blobs
                prefix = layer_name.removesuffix("/convolution")
                tensors[f"{prefix}.weight"] = np.ascontiguousarray(weight)
                tensors[f"{prefix}.bias"] = np.ascontiguousarray(bias).reshape(-1)
            elif layer.type == "Deconvolution":
                # QuPath places grid-2 probabilities at even pixels with zeros
                # between them. This fixed unit kernel has no learned parameter.
                if len(layer.blobs) != 1 or not np.array_equal(layer.blobs[0], np.ones((1, 1, 1, 1))):
                    raise ValueError(f"unexpected fixed upsampling kernel: {layer_name}")
            elif layer.blobs:
                raise ValueError(f"unhandled tensors: {layer_name} ({layer.type})")
        folder = weights / name
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "model.safetensors"
        save_file(tensors, target)
        register(weights, name, target)
        print(f"{name}: {len(tensors)} convolution tensors", flush=True)


def reference(graphs: Path, out: Path) -> None:
    for name, (file, channels, grid) in MODELS.items():
        net = cv2.dnn.readNetFromTensorflow(str(graphs / file))
        image = test_image_he_2d() if channels == 3 else test_image_nuclei_2d()[..., None]
        for index, (y, x, height, width) in enumerate([(32, 32, 128, 128), (128, 224, 128, 160)]):
            # CSBDeep's usual per-channel 1st / 99.8th percentile normalization,
            # left unclipped. Values may be outside [0, 1].
            tile = image[y:y+height, x:x+width]
            if tile.shape != (height, width, channels):
                raise ValueError("reference crop is outside the sample image")
            normalized = normalize(tile, 1, 99.8, axis=(0, 1)).astype(np.float32)
            nchw = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
            net.setInput(nchw)
            output = net.forward()
            if output.shape != (1, 33, height, width):
                raise ValueError(f"unexpected {name} output shape: {output.shape}")
            # Undo only the QuPath export's fixed output upsampling. The network
            # itself predicts on grid 2 (or grid 1 for the original paper model).
            probability = output[0, 0, ::grid, ::grid].copy()
            distances = output[0, 1:, ::grid, ::grid].copy()
            points, scores, _ = non_maximum_suppression(
                np.maximum(distances.transpose(1, 2, 0), 1e-3), probability,
                grid=(grid, grid), b=2, prob_thresh=PROB_THRESHOLD, nms_thresh=NMS_THRESHOLD,
            )
            folder = out / "stardist" / name / str(index)
            folder.mkdir(parents=True, exist_ok=True)
            tile.transpose(2, 0, 1).astype("<f4").tofile(folder / "raw_input.bin")
            nchw.astype("<f4").tofile(folder / "input.bin")
            probability.astype("<f4").tofile(folder / "probability.bin")
            distances.astype("<f4").tofile(folder / "distances.bin")
            points.astype("<i4").tofile(folder / "points.bin")
            scores.astype("<f4").tofile(folder / "scores.bin")
            write_json(folder / "meta.json", {
                "graph": file, "sample": "stardist.data H&E" if channels == 3 else "stardist.data nuclei",
                "crop_yxhw": [y, x, height, width], "channels": channels, "grid": grid,
                "prob_threshold": PROB_THRESHOLD, "nms_threshold": NMS_THRESHOLD,
                "opencv": cv2.__version__, "stardist": stardist.__version__,
            })
            print(f"{name}/{index}: {height}x{width}, {len(points)} polygons", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("export")
    extract.add_argument("--graphs", type=Path, required=True)
    extract.add_argument("--weights", type=Path, required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--graphs", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.graphs, args.weights)
    else:
        reference(args.graphs, args.out)


if __name__ == "__main__":
    main()
