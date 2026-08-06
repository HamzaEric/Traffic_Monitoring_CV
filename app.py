from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Kileleshwa Traffic Monitor",
    page_icon="🚗",
    layout="wide"
)

# Header Section
st.title("Custom YOLO Traffic Monitor")
st.caption("Trained using the Dhaka-AI Dataset")
st.markdown("---")

st.subheader("Kileleshwa Traffic Monitoring CV")
st.caption(
    "Real-time computer vision applied to Nairobi's urban transit. "
    "Processing multi-class vehicle detection for practical traffic analysis."
)

st.divider()

# --- MAIN PAGE NAVIGATION ---
# Moved from the sidebar to the main page, styled horizontally
app_mode = st.radio(
    "Select Processing Mode",
    ["Image Analysis", "Kileleshwa Video Feed"],
    horizontal=True
)

st.divider()

# --- HARDCODED MODEL SETTINGS ---
model_path = "yolo model/best.pt"
conf_threshold = 0.35
iou_threshold = 0.45

@st.cache_resource
def load_model(path_str: str):
    path = Path(path_str)
    if not path.exists():
        st.error(f"Weights file not found at: `{path_str}`. Please verify the directory path.")
        st.stop()
    return YOLO(path)

model = load_model(model_path)

# --- INVERTED CLASS MAPPING ---
# Assign directly to the underlying PyTorch model to bypass read-only wrapper
model.model.names = {
    0: 'ambulance',
    1: 'army vehicle',
    2: 'auto rickshaw',
    3: 'bicycle',
    4: 'bus',
    5: 'car',
    6: 'garbagevan',
    7: 'human hauler',
    8: 'minibus',
    9: 'minivan',
    10: 'motorbike',
    11: 'pickup',
    12: 'policecar',
    13: 'rickshaw',
    14: 'scooter',
    15: 'suv',
    16: 'taxi',
    17: 'three wheelers (CNG)',
    18: 'truck',
    19: 'van',
    20: 'wheelbarrow'
}

# ==========================================
# MODE 1: IMAGE ANALYSIS (Upload + Samples)
# ==========================================
if app_mode == "Image Analysis":

    # Session State for Sample Buttons
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    def set_sample_image(url):
        st.session_state.selected_sample = url

    st.markdown("### Quick Test Samples")
    st.caption("Click a sample to test the model immediately without uploading a local file.")

    # Sample Image URLs (Replace with direct image URLs as needed)
    sample_1 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAvdDGwjLjLQYJXvbfmrjIlTeeMoYjKAe2iy-LUw4uRTdG0Mze-lliOlas&s=10"
    sample_2 = "https://c8.alamy.com/comp/HWAWNW/nairobi-kenya-december-23-roundabout-and-kicc-area-entrance-at-city-HWAWNW.jpg"

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.image(sample_1, caption="Sample 1: Heavy Urban Traffic", use_container_width=True)
        st.button("Try Sample 1", key="btn_s1", on_click=set_sample_image, args=(sample_1,))

    with col_s2:
        st.image(sample_2, caption="Sample 2: Multi-Vehicle Junction", use_container_width=True)
        st.button("Try Sample 2", key="btn_s2", on_click=set_sample_image, args=(sample_2,))

    st.divider()

    uploaded_file = st.file_uploader("...or upload a custom traffic image", type=["jpg", "jpeg", "png"])

    input_image = None

    # Handle image source selection
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.selected_sample = None  # Clear sample choice if user uploads a file
    elif st.session_state.selected_sample is not None:
        with st.spinner("Fetching sample image..."):
            response = requests.get(st.session_state.selected_sample)
            input_image = Image.open(BytesIO(response.content)).convert("RGB")

    # Run Inference on Image
    if input_image is not None:
        with st.spinner("Running detection..."):
            results = model.predict(
                source=input_image,
                conf=conf_threshold,
                iou=iou_threshold
            )

            result = results[0]
            res_plotted = result.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Source Image")
            st.image(input_image, use_container_width=True)

        with col2:
            st.subheader("Detections")
            st.image(res_rgb, use_container_width=True)

        st.divider()
        st.subheader("Detection Metrics")

        boxes = result.boxes
        if len(boxes) > 0:
            class_ids = boxes.cls.tolist()
            class_names = [result.names[int(c)] for c in class_ids]

            counts = {}
            for name in class_names:
                counts[name] = counts.get(name, 0) + 1

            metric_cols = st.columns(len(counts))
            for idx, (cls_name, count) in enumerate(counts.items()):
                metric_cols[idx].metric(label=f"Class: {cls_name}", value=count)
        else:
            st.info("No objects detected at the current confidence threshold.")


# ==========================================
# MODE 2: KILELESHWA VIDEO FEED INFERENCE
# ==========================================
elif app_mode == "Kileleshwa Video Feed":
    video_path = "traffic feed video/kileleshwa.mp4"

    st.markdown("### Raw Video Feed")
    st.video(video_path)

    st.divider()
    st.markdown("### Real-Time Frame Inference")

    start_video = st.button("Start Live Detection Processing")
    stop_video = st.button("Stop Processing")

    if start_video:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error(f"Could not open video file at `{video_path}`. Verify the file path.")
        else:
            frame_window = st.empty()
            metrics_window = st.empty()

            while cap.isOpened() and not stop_video:
                ret, frame = cap.read()
                if not ret:
                    st.info("Video playback completed.")
                    break

                # Run inference on frame
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )

                result = results[0]
                res_plotted = result.plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                # Render processed frame live
                frame_window.image(res_rgb, caption="Live Detection Stream", use_container_width=True)

                # Render real-time class counts
                boxes = result.boxes
                if len(boxes) > 0:
                    class_ids = boxes.cls.tolist()
                    class_names = [result.names[int(c)] for c in class_ids]

                    counts = {}
                    for name in class_names:
                        counts[name] = counts.get(name, 0) + 1

                    with metrics_window.container():
                        st.subheader("Live Vehicle Counts")
                        m_cols = st.columns(max(len(counts), 1))
                        for idx, (cls_name, count) in enumerate(counts.items()):
                            m_cols[idx].metric(label=f"Class: {cls_name}", value=count)

            cap.release()
