import os
import cv2
import numpy as np

# Folder with filtered masks
FILTERED_MASKS_FOLDER = "filtered_masks"

# Output CSV file to store area estimations
OUTPUT_CSV = "rooftop_areas.csv"

# Estimated ground resolution (in meters per pixel)
# Adjust this value based on your image source for real-world accuracy
GROUND_RESOLUTION_M = 0.3  # example: 0.3 meters per pixel (high-res satellite image)

def calculate_area_from_mask(mask_path, ground_resolution):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    white_pixel_count = np.sum(mask == 255)
    area_m2 = white_pixel_count * (ground_resolution ** 2)
    return area_m2

def main():
    if not os.path.exists(FILTERED_MASKS_FOLDER):
        print(f"❌ Folder '{FILTERED_MASKS_FOLDER}' not found.")
        return

    with open(OUTPUT_CSV, "w") as f:
        f.write("filename,area_m2\n")
        for mask_file in os.listdir(FILTERED_MASKS_FOLDER):
            if not mask_file.lower().endswith(".png"):
                continue

            mask_path = os.path.join(FILTERED_MASKS_FOLDER, mask_file)
            area = calculate_area_from_mask(mask_path, GROUND_RESOLUTION_M)
            f.write(f"{mask_file},{area:.2f}\n")
            print(f"✅ {mask_file}: Area = {area:.2f} m²")

    print(f"\n📄 Rooftop area data saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
