from src.otsus_variance.otsus_variance import compute_normalised_histogram, otsu_variance
from pathlib import Path
import cv2
import numpy as np

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"

    image_path = DATA_DIR / "BDS500" / "img1.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    hist = compute_normalised_histogram(image)

    def fitness(candidate_vector):
        return otsu_variance(candidate_vector, hist)

    candidate = [80, 130, 190]
    score = fitness(candidate)
    print(f"Thresholds {candidate} -> variance {score:.4f}")


if __name__ == "__main__":
    main()