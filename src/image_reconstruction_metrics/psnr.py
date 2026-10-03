import numpy as np


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (dB) between two images."""
    if original.shape != reconstructed.shape:
        raise ValueError("Images must have the same shape")

    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff ** 2))

    if mse == 0.0:
        return float("inf")

    return float(10.0 * np.log10((max_val ** 2) / mse))