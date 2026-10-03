import cv2
import numpy as np


def ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_val: float = 255.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Calculate mean Structural Similarity Index (Wang et al., 2004).

    Uses a Gaussian sliding window (11x11, sigma=1.5) and the standard
    constants K1=0.01, K2=0.03. Only fully valid window positions are averaged.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Images must have the same shape")

    x = original.astype(np.float64)
    y = reconstructed.astype(np.float64)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    def blur(a: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            a, (window_size, window_size), sigma,
            borderType=cv2.BORDER_REFLECT,
        )

    mu_x = blur(x)
    mu_y = blur(y)

    sigma_x = blur(x * x) - mu_x ** 2
    sigma_y = blur(y * y) - mu_y ** 2
    sigma_xy = blur(x * y) - mu_x * mu_y

    ssim_map = (
        ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2))
        / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    )

    pad = window_size // 2
    if ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        ssim_map = ssim_map[pad:-pad, pad:-pad]

    return float(ssim_map.mean())