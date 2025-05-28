import streamlit as st
import os
from PIL import Image

# Utility to save uploaded image
def save_uploaded_file(uploaded_file):
    upload_dir = os.path.join("data", "raw")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

# Main Streamlit app
def main():
    st.set_page_config(page_title="AI Solar Analysis - Step 1", layout="centered")
    st.title("🔷 Step 1: Satellite Image Acquisition")
    st.markdown("Upload a rooftop satellite image to begin the analysis.")

    uploaded_file = st.file_uploader(
        "Upload a rooftop image (JPEG/PNG)", 
        type=["jpg", "jpeg", "png"],
        key="unique_upload_key"  # ✅ ensures no duplication error
    )

    if uploaded_file:
        st.success("Image uploaded successfully!")
        
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Rooftop Image", use_column_width=True)

        # Save it
        saved_path = save_uploaded_file(uploaded_file)
        st.info(f"Image saved to: `{saved_path}`")

if __name__ == "__main__":
    main()
