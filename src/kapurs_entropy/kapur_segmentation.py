import cv2
import numpy as np
import glob
import os

def kapur_threshold(image: np.ndarray) -> int:
    """Calculates optimal threshold using Kapur's Maximum Entropy method."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    prob = hist / hist.sum()

    eps = 1e-12
    cum_prob = np.cumsum(prob)
    
    max_entropy = -np.inf
    optimal_t = 0

    for t in range(255):
        p0 = cum_prob[t]
        p1 = 1.0 - p0

        if p0 <= eps or p1 <= eps:
            continue

        prob_bg = prob[: t + 1]
        prob_fg = prob[t + 1 :]

        bg_nonzero = prob_bg[prob_bg > 0]
        fg_nonzero = prob_fg[prob_fg > 0]

        h0 = -np.sum((bg_nonzero / p0) * np.log(bg_nonzero / p0))
        h1 = -np.sum((fg_nonzero / p1) * np.log(fg_nonzero / p1))

        total_entropy = h0 + h1

        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_t = t

    return optimal_t

if __name__ == "__main__":
    # Find all .png files in the folder where the script is located
    png_files = glob.glob("*.png")

    if not png_files:
        print("Error: No .png files found in this folder.")
    else:
        print(f"Found {len(png_files)} .png file(s). Starting processing...\n")
        
        for file_path in png_files:
            # Skip files that we already processed previously
            if "_segmented" in file_path:
                continue
                
            print(f"Processing '{file_path}'...")
            
            # Read image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"  -> Failed to read {file_path}")
                continue

            # Calculate threshold
            best_t = kapur_threshold(img)
            print(f"  -> Kapur Threshold: {best_t}")

            # Apply threshold
            _, segmented = cv2.threshold(img, best_t, 255, cv2.THRESH_BINARY)

            # Generate output filename and save
            file_name, extension = os.path.splitext(file_path)
            output_name = f"{file_name}_segmented{extension}"
            
            cv2.imwrite(output_name, segmented)
            print(f"  -> Saved output as '{output_name}'\n")
            
        print("All files have been processed successfully!")