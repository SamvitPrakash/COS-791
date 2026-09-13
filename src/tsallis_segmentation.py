from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .tsallis_segmentation import apply_multilevel_thresholds, tsallis_fitness


def _parse_thresholds(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("No thresholds were provided.")
    return [int(v) for v in values]


def _labels_to_uint8(labels: np.ndarray) -> np.ndarray:
    max_label = int(labels.max())
    if max_label == 0:
        return np.zeros_like(labels, dtype=np.uint8)
    return np.round((labels.astype(np.float64) / max_label) * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Tsallis multilevel objective and save segmented output."
    )
    parser.add_argument("--image", required=True, help="Path to a grayscale image.")
    parser.add_argument(
        "--thresholds",
        required=True,
        help="Comma-separated threshold list, e.g. 60,120,180",
    )
    parser.add_argument("--q", type=float, default=0.8, help="Tsallis q parameter.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for segmented image.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load grayscale image: {image_path}")

    thresholds = _parse_thresholds(args.thresholds)
    score = tsallis_fitness(image=image, thresholds=thresholds, q=args.q)
    print(f"Tsallis fitness (maximize) = {score:.10f}")
    print(f"Tsallis cost (minimize)    = {-score:.10f}")

    labels = apply_multilevel_thresholds(image=image, thresholds=thresholds)
    segmented = _labels_to_uint8(labels)

    output_path = Path(args.output) if args.output else image_path.with_name(
        f"{image_path.stem}_tsallis_segmented{image_path.suffix}"
    )
    cv2.imwrite(str(output_path), segmented)
    print(f"Saved segmented image to: {output_path}")


if __name__ == "__main__":
    main()