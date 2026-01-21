import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Desai Eye Hospital AI",
    layout="centered"
)

st.title("🏥 Desai Eye Hospital")
st.subheader("AI-Based Eye Disease Screening System")

IMG_SIZE = 224

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    glaucoma = load_model("glaucoma_model.h5")
    dr = load_model("diabetic_retinopathy_model.h5")
    return glaucoma, dr

glaucoma_model, dr_model = load_models()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ----------------
test_type = st.selectbox(
    "Select Screening Type",
    ("Glaucoma Detection", "Diabetic Retinopathy Detection")
)

uploaded_file = st.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Fundus Image", use_container_width=True)

    if st.button("Analyze Image"):
        img = preprocess(image)

        if test_type == "Glaucoma Detection":
            prediction = glaucoma_model.predict(img)[0]
        else:
            prediction = dr_model.predict(img)[0]

        confidence = float(np.max(prediction)) * 100

        st.success(f"🔍 Confidence Score: **{confidence:.2f}%**")

st.markdown("---")
st.caption(
    "⚠️ This AI system is for screening only and does not replace professional medical diagnosis."
)
