import os
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.transforms.functional import to_pil_image

# Paths
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "models/sam_vit_b_01ec64.pth"

# Make sure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load SAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Process images
for image_file in os.listdir(IMAGE_FOLDER):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Generating masks for {image_file}...")
        masks = mask_generator.generate(image_rgb)

        # Create mask overlay
        mask_image = image_rgb.copy()
        for mask in masks:
            m = mask['segmentation']
            mask_image[m] = [0, 255, 0]  # Green mask

        # Save output
        output_path = os.path.join(OUTPUT_FOLDER, f"masked_{image_file}")
        cv2.imwrite(output_path, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
        print(f"Saved masked image to {output_path}")
