import os
import cv2
import numpy as np
from src.lade import LateAcceptanceDE

# 1. Otsu Variance objective function
def compute_otsu(thresholds, p_hist):
    t = np.sort(np.rint(thresholds).astype(int))
    t = np.clip(t, 1, 254)
    if len(np.unique(t)) < len(t):
        return -1e9
    
    boundaries = [0] + list(t) + [256]
    levels = np.arange(256)
    mu_T = np.sum(levels * p_hist)
    sigma_b = 0.0
    
    for i in range(len(boundaries) - 1):
        c_start, c_end = boundaries[i], boundaries[i+1]
        w_k = np.sum(p_hist[c_start:c_end])
        if w_k > 1e-12:
            mu_k = np.sum(levels[c_start:c_end] * p_hist[c_start:c_end]) / w_k
            sigma_b += w_k * ((mu_k - mu_T) ** 2)
            
    return float(sigma_b)

def apply_segmentation(image, thresholds):
    """Segment image into discrete class levels based on thresholds."""
    segmented = np.zeros_like(image)
    boundaries = [0] + list(thresholds) + [256]
    step = 255 // (len(thresholds))
    for i in range(len(boundaries) - 1):
        mask = (image >= boundaries[i]) & (image < boundaries[i+1])
        segmented[mask] = i * step
    return segmented

def main():
    img_path = os.path.join("data", "CHAOS_DATA", "IMG-0003-00010.png")
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}, please check path.")
        return

    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    p_hist = hist.astype(np.float64) / img.size

    # Test for K=5 thresholds
    K = 5
    print(f"Running LADE optimization for K = {K}...")
    optimizer = LateAcceptanceDE(
        objective_func=lambda th: compute_otsu(th, p_hist),
        K=K,
        max_fes=10000,
        pop_size=30,
        F=0.5,
        CR=0.8,
        L=50
    )

    best_thresholds, best_score = optimizer.optimize()
    print("Optimization finished successfully!")
    print(f"Optimal Thresholds: {best_thresholds}")
    print(f"Best Objective Score: {best_score:.4f}")

    # Generate and save output image to check visual results
    os.makedirs("results", exist_ok=True)
    segmented_img = apply_segmentation(img, best_thresholds)
    out_path = os.path.join("results", f"test_LADE_K{K}_segmented.png")
    cv2.imwrite(out_path, segmented_img)
    print(f"Saved segmented test output to: {out_path}")

if __name__ == "__main__":
    main()
