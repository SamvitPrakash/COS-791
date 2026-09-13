import cv2
import numpy as np


def otsu_threshold(image: np.ndarray) -> int:
    """Calculate optimal threshold using Otsu's between-class variance."""

    hist = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    prob = hist / hist.sum()

    max_variance = -np.inf
    optimal_t = 0

    cumulative_prob = np.cumsum(prob)
    cumulative_mean = np.cumsum(
        np.arange(256) * prob
    )

    total_mean = cumulative_mean[-1]

    for t in range(255):
        w0 = cumulative_prob[t]
        w1 = 1.0 - w0

        if w0 <= 0 or w1 <= 0:
            continue

        mu0 = cumulative_mean[t] / w0
        mu1 = (total_mean - cumulative_mean[t]) / w1

        between_variance = (
            w0
            * w1
            * (mu0 - mu1) ** 2
        )

        if between_variance > max_variance:
            max_variance = between_variance
            optimal_t = t

    return int(optimal_t)