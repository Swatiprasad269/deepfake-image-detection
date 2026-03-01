import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page config
st.set_page_config(page_title="AI Deepfake Detector", layout="centered")

st.title("🧠 AI Deepfake Image Detection")
st.write("Upload an image to check if it is REAL or FAKE.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepfake_mobilenetv2_model.h5")
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    st.write(f"Raw prediction: {prediction: .4f}")

    if prediction > 0.5:
        st.error(f"✅ REAL Image (Confidence: {prediction:.2f})")
    else:
        st.success(f"⚠️ FAKE Image (Confidence: {1-prediction:.2f})")