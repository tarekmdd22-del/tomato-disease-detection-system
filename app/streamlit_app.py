"""
Streamlit UI for the Tomato Disease Detection System.

Run:
    streamlit run app/streamlit_app.py
"""
import json
import os
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import IMG_SIZE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection")
st.caption("Upload a photo of a tomato leaf to detect possible diseases and get a confidence score.")


@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, "tomato_disease_model.keras")
    classes_path = os.path.join(MODELS_DIR, "class_names.json")
    if not (os.path.exists(model_path) and os.path.exists(classes_path)):
        return None, None
    model = tf.keras.models.load_model(model_path)
    with open(classes_path) as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model()

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if model is None:
        st.error("No trained model found. Run `python src/train_model.py --data data/train` first.")
    else:
        img = image.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)

        proba = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(proba))
        label = class_names[top_idx]
        confidence = float(proba[top_idx])
        status = "Healthy 🌱" if label.strip().lower() in {"healthy", "healthy tomato leaf", "healthy_tomato_leaf"} else "Diseased ⚠️"

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Disease", label)
        col2.metric("Confidence", f"{confidence:.1%}")
        st.write(f"**Status:** {status}")

        st.write("**Top 3 predictions:**")
        top3_idx = np.argsort(proba)[::-1][:3]
        for i in top3_idx:
            st.write(f"- {class_names[i]}: {proba[i]:.1%}")

st.divider()
st.caption("Built for the Smart Tomato Disease Detection System project.")
