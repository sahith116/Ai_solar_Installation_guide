import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import gdown
import plotly.graph_objects as go
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import asyncio

config_dir = '/tmp/UltralyticsConfig'
os.makedirs(config_dir, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = config_dir

CLIP_LABELS = ["a rooftop", "a road", "a forest"]
YOLO_MODEL_PATH = "yolov8n.pt"
PX_PER_METER = 10
SAM_DRIVE_ID = "1lAipianp9NLedqF4xWJ-YSSgwJaTff6R"
SAM_MODEL_PATH = "sam_vit_b_01ec64.pth"

st.set_page_config(page_title="AI Solar Analysis", layout="wide")

def download_sam_model():
    if not os.path.exists(SAM_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={SAM_DRIVE_ID}"
        with st.spinner("📅 Downloading SAM model from Google Drive..."):
            gdown.download(url, SAM_MODEL_PATH, quiet=False)
    return SAM_MODEL_PATH

@st.cache_resource
def load_models():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = download_sam_model()

        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        sam_predictor = SamPredictor(sam_model)

        yolo_model = YOLO(YOLO_MODEL_PATH)

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        return sam_predictor, yolo_model, clip_model, clip_processor, device
    except Exception as e:
        st.error(f"🚨 Model loading failed: {e}")
        st.stop()

sam_predictor, yolo_model, clip_model, clip_processor, device = load_models()

def verify_scene(image):
    st.info("🔍 Verifying scene with CLIP...")

    image_input = clip_processor.images_processor(image, return_tensors="pt")
    text_input = clip_processor.tokenizer(CLIP_LABELS, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = clip_model(**text_input, **image_input)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    rooftop_prob = probs[CLIP_LABELS.index("a rooftop")].item()
    return rooftop_prob > 0.5, dict(zip(CLIP_LABELS, probs.tolist()))

def segment_rooftop(image_pil):
    st.info("🔍 Running rooftop segmentation with SAM...")
    image = np.array(image_pil)
    sam_predictor.set_image(image)
    H, W, _ = image.shape
    input_point = np.array([[W // 2, H // 2]])
    input_label = np.array([1])
    masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    return masks[0]

def detect_obstacles(image_np):
    st.info("🧠 Detecting obstacles with YOLO...")
    results = yolo_model(image_np)
    return results[0].boxes.data.cpu().numpy() if results and results[0].boxes is not None else []

def apply_obstacle_mask(mask, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        mask[y1:y2, x1:x2] = 0
    return mask

def layout_panels(mask, panel_h, panel_w):
    layout = np.zeros_like(mask)
    count = 0
    step = int(min(panel_h, panel_w) * 1.1)
    for y in range(0, mask.shape[0] - panel_h, step):
        for x in range(0, mask.shape[1] - panel_w, step):
            region = mask[y:y+panel_h, x:x+panel_w]
            if np.all(region == 1):
                layout[y:y+panel_h, x:x+panel_w] = 2
                count += 1
    return layout, count

def calculate_solar_stats(num_panels, panel_area_m2, panel_size_kw, cost_per_kw, energy_price, annual_output):
    total_area_m2 = num_panels * panel_area_m2
    solar_capacity_kw = total_area_m2 / panel_size_kw
    installation_cost = solar_capacity_kw * cost_per_kw
    annual_savings = solar_capacity_kw * annual_output * energy_price / 1000
    payback_period = installation_cost / annual_savings if annual_savings > 0 else float('inf')
    return total_area_m2, solar_capacity_kw, installation_cost, annual_savings, payback_period

def plot_savings_chart(payback_period, annual_savings):
    years = list(range(int(payback_period) + 6))
    cumulative_savings = [annual_savings * y for y in years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=cumulative_savings, mode='lines+markers', name='Cumulative Savings (₹)'))
    fig.add_vline(x=payback_period, line_dash="dash", line_color="red", annotation_text="Payback Period", annotation_position="top right")
    fig.update_layout(title="Projected Savings Over Years", xaxis_title="Years", yaxis_title="₹ Savings")
    return fig

def overlay_mask(image_pil, mask):
    image = np.array(image_pil).copy()
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] = mask * 255
    return cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

def overlay_panels(image_pil, panel_mask):
    image = np.array(image_pil).copy()
    panel_overlay = np.zeros_like(image)
    panel_overlay[:, :, 0] = (panel_mask == 2) * 255
    return cv2.addWeighted(image, 0.7, panel_overlay, 0.3, 0)

def generate_pdf_report(image, stats, fig_image, panel_overlay):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 760, "AI Rooftop Solar Analysis Report")

    c.setFont("Helvetica", 12)
    y = 730
    for key, value in stats.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    overlay_img = Image.fromarray(panel_overlay)
    overlay_buffer = io.BytesIO()
    overlay_img.save(overlay_buffer, format="PNG")
    overlay_buffer.seek(0)
    c.drawImage(ImageReader(overlay_buffer), 50, 300, width=240, height=180)

    fig_buffer = io.BytesIO()
    fig_image.write_image(fig_buffer, format="png")
    fig_buffer.seek(0)
    c.drawImage(ImageReader(fig_buffer), 310, 300, width=240, height=180)

    c.save()
    buffer.seek(0)
    return buffer

def main():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    st.title("🌞 AI Rooftop Solar Analysis with SAM, YOLO & CLIP")

    uploaded_file = st.file_uploader("Upload Rooftop Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.info("Please upload a rooftop image to begin analysis.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Verifying scene..."):
        verified, probs = verify_scene(image)
    if not verified:
        st.error("❌ Uploaded image does not appear to be a rooftop. Try another image.")
        if st.checkbox("Show CLIP probabilities"):
            st.write(probs)
        return
    st.success("✅ Scene verified as rooftop.")

    with st.spinner("Segmenting rooftop..."):
        mask = segment_rooftop(image)

    image_np = np.array(image)
    mask = mask.astype(np.uint8)

    with st.spinner("Detecting obstacles..."):
        detections = detect_obstacles(image_np)

    mask = apply_obstacle_mask(mask, detections)

    st.image(mask * 255, caption="Rooftop Mask (with obstacles filtered)", use_container_width=True)
    st.image(overlay_mask(image, mask), caption="Overlay of Mask on Image", use_container_width=True)

    st.sidebar.header("🔧 Assumptions")
    panel_length = st.sidebar.number_input("Panel Length (m)", 1.0, 3.0, 1.6, 0.1)
    panel_width = st.sidebar.number_input("Panel Width (m)", 0.5, 2.0, 1.0, 0.1)
    cost_per_kw = st.sidebar.number_input("Installation cost per kW (₹)", 10000, 200000, 70000, step=1000)
    energy_price = st.sidebar.number_input("Energy price per kWh (₹)", 1, 20, 8, step=1)
    annual_output = st.sidebar.number_input("Annual kWh output per kW", 1000, 2000, 1200, step=50)
    panel_size_kw = 6.5

    panel_h_px = int(panel_length * PX_PER_METER)
    panel_w_px = int(panel_width * PX_PER_METER)

    panel_mask, num_panels = layout_panels(mask, panel_h_px, panel_w_px)
    overlay_image = overlay_panels(image, panel_mask)
    st.image(overlay_image, caption="Solar Panel Layout", use_container_width=True)

    area_m2, solar_capacity_kw, installation_cost, annual_savings, payback_period = calculate_solar_stats(
        num_panels, panel_length * panel_width, panel_size_kw, cost_per_kw, energy_price, annual_output
    )

    st.markdown("## ☀️ Solar Feasibility Report Summary")
    st.markdown(f"- **Usable Area:** {area_m2:.2f} m²")
    st.markdown(f"- **Estimated Solar Capacity:** {solar_capacity_kw:.2f} kW")
    st.markdown(f"- **Installation Cost:** ₹{installation_cost:,.0f}")
    st.markdown(f"- **Annual Savings:** ₹{annual_savings:,.0f}")
    st.markdown(f"- **Payback Period:** {payback_period:.1f} years")

    fig = plot_savings_chart(payback_period, annual_savings)
    st.plotly_chart(fig, use_container_width=True)

    image_with_boxes = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
    st.image(image_with_boxes, caption="Detected Obstacles", use_container_width=True)

    st.markdown("---")
    summary_df = pd.DataFrame([{
        "Usable Area (m²)": area_m2,
        "Capacity (kW)": solar_capacity_kw,
        "Cost (₹)": installation_cost,
        "Savings (₹/year)": annual_savings,
        "Payback Period (years)": payback_period
    }])
    csv = summary_df.to_csv(index=False).encode()
    st.download_button("📅 Download Report (CSV)", data=csv, file_name="solar_report.csv", mime="text/csv")

    pdf_stats = {
        "Usable Area (m²)": f"{area_m2:.2f}",
        "Capacity (kW)": f"{solar_capacity_kw:.2f}",
        "Cost (₹)": f"{installation_cost:,.0f}",
        "Savings (₹/year)": f"{annual_savings:,.0f}",
        "Payback Period (years)": f"{payback_period:.1f}"
    }
    pdf_buffer = generate_pdf_report(image, pdf_stats, fig, overlay_image)
    st.download_button("📄 Download PDF Report", data=pdf_buffer, file_name="solar_analysis_report.pdf", mime="application/pdf")

    st.markdown("### 🔍 Solar Assessment Summary")
    st.markdown(f"""
    Based on the analysis, approximately **{area_m2:.1f} m²** of rooftop is usable for solar panel installation.
    This can support about **{num_panels} panels**, generating around **{solar_capacity_kw * annual_output:.0f} kWh/year**.
    Estimated annual savings are **₹{annual_savings:,.0f}**, with a payback period of **{payback_period:.1f} years**.
    For best results, install panels in **sunny, unobstructed, south-facing** areas.
    """)

if __name__ == "__main__":
    main()