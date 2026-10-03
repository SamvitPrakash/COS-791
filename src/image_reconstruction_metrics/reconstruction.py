import numpy as np


def threshold_regions(image: np.ndarray, thresholds) -> np.ndarray:
    """Assign each pixel a region index (0..K-1) from K-1 thresholds.

    Follows the repo convention of cv2.THRESH_BINARY: a pixel belongs to the
    higher region when its intensity is strictly greater than a threshold.
    """
    t = np.sort(np.atleast_1d(np.asarray(thresholds, dtype=np.int64)))
    return np.searchsorted(t, image, side="left")


def reconstruct_image(
    image: np.ndarray,
    thresholds,
    mode: str = "mean",
) -> np.ndarray:
    """Convert thresholds into a segmented pixel intensity map.

    mode="mean":   each region is filled with the mean intensity of the
                   original pixels it contains (used for PSNR/SSIM/U).
    mode="levels": each region is filled with evenly spaced grey levels
                   from 0 to 255 (matches the binary 0/255 output in main.py
                   when there is a single threshold).
    """
    if mode not in ("mean", "levels"):
        raise ValueError("mode must be 'mean' or 'levels'")

    regions = threshold_regions(image, thresholds)
    k = int(np.atleast_1d(thresholds).size) + 1

    if mode == "levels":
        palette = np.round(np.linspace(0, 255, k)).astype(np.uint8)
        return palette[regions]

    reconstructed = np.zeros(image.shape, dtype=np.uint8)
    for j in range(k):
        mask = regions == j
        if np.any(mask):
            reconstructed[mask] = np.uint8(round(float(image[mask].mean())))
    return reconstructed