import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_FOLDER = 'images'
MASK_FOLDER = 'masks'
OUTPUT_FOLDER = 'filtered_masks'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def filter_mask(mask):
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours and filter by area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # adjust threshold as needed
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask

for filename in os.listdir(IMAGE_FOLDER):
    name, ext = os.path.splitext(filename)
    mask_path = os.path.join(MASK_FOLDER, f"{name}_mask.png")
    image_path = os.path.join(IMAGE_FOLDER, filename)

    if not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path)
    filtered = filter_mask(mask)

    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_filtered.png")
    cv2.imwrite(output_path, filtered)

    # Optional: Show the result
    # plt.imshow(filtered, cmap='gray')
    # plt.title(f"Filtered Mask: {name}")
    # plt.show()

print("✅ Mask filtering complete. Filtered masks saved in 'filtered_masks/' folder.")
