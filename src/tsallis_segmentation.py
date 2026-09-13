import cv2
import numpy as np


def tsallis_threshold(image: np.ndarray, q: float = 0.8) -> int:
    """Calculate optimal bi-level threshold using Tsallis non-extensive entropy."""
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