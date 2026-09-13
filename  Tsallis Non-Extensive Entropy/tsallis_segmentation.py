import cv2
import numpy as np
import glob
import os


def tsallis_threshold(image: np.ndarray, q: float = 0.8) -> int:
    """Calculates optimal bi-level threshold using Tsallis non-extensive entropy."""
    if q <= 0:
        raise ValueError("q must be > 0")

    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    prob = hist / hist.sum()

    eps = 1e-12
    max_entropy = -np.inf
    optimal_t = 0

    for t in range(255):
        p0 = float(np.sum(prob[: t + 1]))
        p1 = float(np.sum(prob[t + 1 :]))

        if p0 <= eps or p1 <= eps:
            continue

        bg = prob[: t + 1] / p0
        fg = prob[t + 1 :] / p1

        bg = bg[bg > 0]
        fg = fg[fg > 0]

        if abs(q - 1.0) <= 1e-12:
            s0 = -np.sum(bg * np.log(bg))
            s1 = -np.sum(fg * np.log(fg))
            total_entropy = s0 + s1
        else:
            s0 = (1.0 - np.sum(np.power(bg, q))) / (q - 1.0)
            s1 = (1.0 - np.sum(np.power(fg, q))) / (q - 1.0)
            total_entropy = s0 + s1 + (1.0 - q) * s0 * s1

        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_t = t

    return optimal_t


if __name__ == "__main__":
    q = 0.8

    # Find all .png files in the folder where the script is executed
    png_files = glob.glob("*.png")

    if not png_files:
        print("Error: No .png files found in this folder.")
    else:
        print(f"Found {len(png_files)} .png file(s). Starting Tsallis processing (q={q})...\\n")

        for file_path in png_files:
            # Skip files that were already segmented by any method.
            if "_segmented" in file_path:
                continue

            print(f"Processing '{file_path}'...")

            # Read image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"  -> Failed to read {file_path}")
                continue

            # Calculate threshold
            best_t = tsallis_threshold(img, q=q)
            print(f"  -> Tsallis Threshold (q={q}): {best_t}")

            # Apply threshold
            _, segmented = cv2.threshold(img, best_t, 255, cv2.THRESH_BINARY)

            # Generate output filename and save
            file_name, extension = os.path.splitext(file_path)
            output_name = f"{file_name}_tsallis_segmented{extension}"

            cv2.imwrite(output_name, segmented)
            print(f"  -> Saved output as '{output_name}'\\n")

        print("All files have been processed successfully!")
