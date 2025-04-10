
import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace

st.set_page_config(page_title="🧠 Mood Detector", layout="centered")
st.title("😊 Mood Detector from Your Photo")

uploaded_file = st.file_uploader("📸 Upload a face photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Your Photo", use_column_width=True)

    img_np = np.array(img)

    st.subheader("🔍 Detecting Mood...")
    try:
        result = DeepFace.analyze(img_path=img_np, actions=["emotion"], enforce_detection=False)
        emotion = result[0]["dominant_emotion"]
        st.success(f"🥳 Mood Detected: **{emotion.upper()}**")
    except Exception as e:
        st.error("⚠️ Could not detect mood. Please try another face photo.")
