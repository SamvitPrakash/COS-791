import numpy as np

from src.image_reconstruction_metrics.reconstruction import threshold_regions


def uniformity(image: np.ndarray, thresholds) -> float:
    """Calculate the Uniformity Metric U (Levine & Nazif) for a segmentation.

    U = 1 - (2 * (K - 1) / (N * (f_max - f_min)^2)) * sum_j sum_{i in R_j} (f_i - mu_j)^2

    where K-1 is the number of thresholds, N the pixel count, R_j region j and
    mu_j its mean intensity. Result lies in [0, 1]; higher means more uniform.
    """
    f = image.astype(np.float64)
    f_range = float(f.max() - f.min())

    if f_range == 0.0:
        return 1.0

    regions = threshold_regions(image, thresholds)
    num_thresholds = int(np.atleast_1d(thresholds).size)

    within = 0.0
    for j in range(num_thresholds + 1):
        vals = f[regions == j]
        if vals.size:
            within += float(np.sum((vals - vals.mean()) ** 2))

    return float(1.0 - (2.0 * num_thresholds * within) / (f.size * f_range ** 2))