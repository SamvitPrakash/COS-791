from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.kapurs_entropy.kapur_segmentation import kapur_threshold
from src.otsus_variance.otsus_variance import otsu_threshold
from src.tsallis_entropy.tsallis_segmentation import tsallis_threshold


def process_image(image_path: Path, q: float, output_dir: Path | None = None) -> None:

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(f"[SKIP] Could not read image: {image_path}")
        return

    # Calculate thresholds
    kapur_t = kapur_threshold(image)
    otsu_t = otsu_threshold(image)
    tsallis_t = tsallis_threshold(image, q=q)

    # Apply thresholds
    _, kapur_segmented = cv2.threshold(
        image,
        kapur_t,
        255,
        cv2.THRESH_BINARY
    )

    _, otsu_segmented = cv2.threshold(
        image,
        otsu_t,
        255,
        cv2.THRESH_BINARY
    )

    _, tsallis_segmented = cv2.threshold(
        image,
        tsallis_t,
        255,
        cv2.THRESH_BINARY
    )

    # Determine output directory
    out_dir = (
        output_dir
        if output_dir is not None
        else image_path.parent
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Generate output filenames
    kapur_out = (
        out_dir
        / f"{image_path.stem}_kapur_segmented{image_path.suffix}"
    )

    otsu_out = (
        out_dir
        / f"{image_path.stem}_otsu_segmented{image_path.suffix}"
    )

    tsallis_out = (
        out_dir
        / f"{image_path.stem}_tsallis_segmented{image_path.suffix}"
    )

    # Save segmented images
    cv2.imwrite(
        str(kapur_out),
        kapur_segmented
    )

    cv2.imwrite(
        str(otsu_out),
        otsu_segmented
    )

    cv2.imwrite(
        str(tsallis_out),
        tsallis_segmented
    )

    # Report results
    print(f"Processing '{image_path.name}'")
    print(f"  -> Kapur Threshold: {kapur_t}")
    print(f"  -> Otsu Threshold: {otsu_t}")
    print(f"  -> Tsallis Threshold (q={q}): {tsallis_t}")
    print(f"  -> Saved: {kapur_out}")
    print(f"  -> Saved: {otsu_out}")
    print(f"  -> Saved: {tsallis_out}\n")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run Kapur, Otsu, and Tsallis thresholding "
            "on one image or all PNG images in a directory."
        )
    )

    parser.add_argument(
        "--image",
        help="Path to one PNG image."
    )

    parser.add_argument(
        "--input-dir",
        help="Directory containing PNG images."
    )

    parser.add_argument(
        "--q",
        type=float,
        default=0.8,
        help="Tsallis q parameter."
    )

    parser.add_argument(
        "--output-dir",
        help="Optional output directory."
    )

    args = parser.parse_args()

    if bool(args.image) == bool(args.input_dir):
        raise ValueError(
            "Provide exactly one of --image or --input-dir."
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else None
    )

    # Process a single image
    if args.image:
        process_image(
            Path(args.image),
            q=args.q,
            output_dir=output_dir
        )
        return

    # Process all PNG files in a directory
    input_dir = Path(args.input_dir)

    png_files = sorted(
        input_dir.glob("*.png")
    )

    if not png_files:
        print(
            f"No .png files found in: {input_dir}"
        )
        return

    print(
        f"Found {len(png_files)} .png file(s). "
        "Starting processing...\n"
    )

    for image_path in png_files:

        # Skip previously generated segmentation outputs
        if "_segmented" in image_path.name:
            continue

        process_image(
            image_path,
            q=args.q,
            output_dir=output_dir
        )

    print(
        "All files have been processed successfully!"
    )


if __name__ == "__main__":
    main()