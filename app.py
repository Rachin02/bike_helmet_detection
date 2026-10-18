import io

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="wide")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

st.title("🪖 Bike Helmet Detection")
st.write("Upload an image and the model marks riders **With Helmet** and **Without Helmet**.")

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if file:
    image = Image.open(file).convert("RGB")
    bgr = np.array(image)[:, :, ::-1]                       # YOLO expects BGR arrays

    result = model.predict(bgr, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]                   # BGR -> RGB for display

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", width="stretch")
    col2.image(annotated, caption="Prediction", width="stretch")

    names = [model.names[int(c)] for c in result.boxes.cls]
    c1, c2 = st.columns(2)
    c1.metric("With Helmet", names.count("With Helmet"))
    c2.metric("Without Helmet", names.count("Without Helmet"))

    # Download the annotated image
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("Download result image", buf.getvalue(), "helmet_result.png", "image/png")

