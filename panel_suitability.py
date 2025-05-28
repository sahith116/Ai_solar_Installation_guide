import os
import cv2
import numpy as np

# Input & output folders
MASK_FOLDER = "filtered_masks"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "panel_suitability_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Panel dimensions in pixels (adjust based on image resolution)
PANEL_WIDTH = 100
PANEL_HEIGHT = 160

def check_panel_fit(mask, panel_width, panel_height):
    suitable_mask = np.zeros_like(mask)
    for y in range(0, mask.shape[0] - panel_height, panel_height):
        for x in range(0, mask.shape[1] - panel_width, panel_width):
            panel_area = mask[y:y+panel_height, x:x+panel_width]
            if np.all(panel_area == 255):
                suitable_mask[y:y+panel_height, x:x+panel_width] = 255
    return suitable_mask

# Process each mask
for filename in os.listdir(MASK_FOLDER):
    if filename.endswith(".png"):
        mask_path = os.path.join(MASK_FOLDER, filename)
        image_path = os.path.join(IMAGE_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)

        # Check where panels can be placed
        suitable = check_panel_fit(mask, PANEL_WIDTH, PANEL_HEIGHT)

        # Overlay panel positions in green
        overlay = image.copy()
        overlay[suitable == 255] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        cv2.imwrite(output_path, blended)

print("✅ Panel suitability estimation complete. Results saved in 'panel_suitability_output/'")
