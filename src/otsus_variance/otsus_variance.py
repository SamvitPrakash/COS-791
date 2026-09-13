from __future__ import annotations

import numpy as np

__all__ = [
    "compute_normalised_histogram",
    "otsu_variance",
    "otsu_fitness",
]


def compute_normalised_histogram(image: np.ndarray, n_bins: int = 256) -> np.ndarray:
    image = np.asarray(image)
    hist, _ = np.histogram(image.ravel(), bins=n_bins, range=(0, n_bins))
    return hist.astype(np.float64) / image.size


def _sorted_unique_thresholds(thresholds, n_bins: int) -> np.ndarray:
    t = np.round(np.asarray(thresholds, dtype=np.float64)).astype(int)
    t = np.clip(t, 1, n_bins - 2)
    return np.unique(t)


def otsu_variance(thresholds, hist_probs: np.ndarray) -> float:
    hist_probs = np.asarray(hist_probs, dtype=np.float64)
    n_bins = hist_probs.size
    levels = np.arange(n_bins, dtype=np.float64)

    t = _sorted_unique_thresholds(thresholds, n_bins)

    mu_t = float(np.sum(levels * hist_probs))

    cum_w = np.concatenate(([0.0], np.cumsum(hist_probs)))
    cum_s = np.concatenate(([0.0], np.cumsum(levels * hist_probs)))

    edges = np.concatenate(([0], t, [n_bins]))

    variance = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo >= hi:
            continue
        w_k = cum_w[hi] - cum_w[lo]
        if w_k <= 0.0:
            continue
        mu_k = (cum_s[hi] - cum_s[lo]) / w_k
        variance += w_k * (mu_k - mu_t) ** 2

    return float(variance)


def otsu_fitness(thresholds, image: np.ndarray, n_bins: int = 256) -> float:
    hist_probs = compute_normalised_histogram(image, n_bins=n_bins)
    return otsu_variance(thresholds, hist_probs)
