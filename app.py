import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="AI Deepfake Detector", layout="centered")

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>🧠 AI Deepfake Image Detection</h1>
    <p style='text-align: center; font-size:18px;'>
    Upload a face image to check whether it is REAL or AI-GENERATED.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "deepfake_mobilenetv2_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ---------- PREPROCESS ----------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    with col2:
        st.subheader("🔍 Prediction")

        if prediction > 0.5:
            st.success("✅ REAL Image")
            st.progress(float(prediction))
            st.caption(f"Confidence: {prediction:.2f}")
        else:
            st.error("⚠️ FAKE Image")
            st.progress(float(1 - prediction))
            st.caption(f"Confidence: {1 - prediction:.2f}")