# ===================================================
# 🎧 MoodMate: Dual Emotion Detection & Music Recommender
# ===================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import time
from tensorflow.keras.models import load_model
from transformers import pipeline
from collections import deque

# ==============================
# 🎨 Page Configuration
# ==============================
st.set_page_config(
    page_title="MoodMate 🎧",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #1DB954;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #999;
        margin-bottom: 2rem;
    }
    .recommend-box {
        background-color: #f9f9f9;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>MoodMate 🎧</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect your mood via webcam or text and get personalized song recommendations 🎵</div>", unsafe_allow_html=True)

# ==============================
# 📂 Load Resources
# ==============================
@st.cache_resource
def load_models():
    model_path = "C:\\Users\\adity\\OneDrive\\Desktop\\ai moodmate\\mobilenet.keras"
    model = load_model(model_path)
    nlp_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", framework="pt")
    return model, nlp_model

model, nlp_model = load_models()
music_df = pd.read_csv("C:\\Users\\adity\\Downloads\\muse_v3.csv")

labels = ["angry", "happy", "sad", "surprise", "neutral"]

emotion_map = {
    "happy": ["happy", "energetic", "upbeat", "party"],
    "sad": ["sad", "melancholic", "emotional", "calm"],
    "angry": ["aggressive", "intense", "rock", "metal"],
    "surprise": ["exciting", "curious", "fast", "adventurous"],
    "neutral": ["chill", "relaxing", "balanced", "ambient"]
}

# ==============================
# 🎵 Helper Functions
# ==============================
def preprocess_face(face_img, img_size=48, model_channels=1):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    arr = resized.astype("float32") / 255.0
    if model_channels == 1:
        arr = np.expand_dims(arr, -1)
    else:
        arr = np.stack([arr, arr, arr], -1)
    return arr

def aggregate_probs(prob_deque, labels):
    if not prob_deque:
        return None, 0.0
    sum_probs = np.sum([p for (_, p) in prob_deque], axis=0)
    sum_probs = sum_probs / (np.sum(sum_probs) + 1e-9)
    idx = np.argmax(sum_probs)
    return labels[idx], float(sum_probs[idx])

def recommend_music(emotion, n=20):
    tags = emotion_map.get(emotion, [])
    genre_cols = [col for col in ["genre", "genre2"] if col in music_df.columns]
    for col in genre_cols:
        music_df[col] = music_df[col].astype(str).str.lower()
    filtered = music_df[
        pd.concat([music_df[col].isin(tags) for col in genre_cols], axis=1).any(axis=1)
    ] if genre_cols else music_df

    if filtered.empty:
        filtered = music_df.sample(n=min(len(music_df), n))

    return filtered.sample(min(len(filtered), n))

def text_to_emotion(text):
    result = nlp_model(text)[0]
    label = result["label"].lower()
    score = result["score"]
    mapping = {
        "anger": "angry",
        "joy": "happy",
        "sadness": "sad",
        "surprise": "surprise",
        "neutral": "neutral",
        "fear": "neutral",
        "love": "happy"
    }
    mapped = mapping.get(label, "neutral")
    return mapped, score

# ==============================
# 🧠 Mode Selection
# ==============================
mode = st.radio("Choose Emotion Detection Mode:", ["🎥 Webcam Mode", "💬 Text Mode"], horizontal=True)

# ==============================
# 🎥 Webcam Emotion Detection
# ==============================
if mode == "🎥 Webcam Mode":
    st.info("Press **Start Webcam** to detect your facial emotion live.")
    start_cam = st.button("▶ Start Webcam")

    if start_cam:
        stframe = st.empty()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        interval = 2.0
        prob_deque = deque()
        frame_count = 0
        num_frames = 200

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam.")
        else:
            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                now = time.time()

                while prob_deque and (now - prob_deque[0][0] > interval):
                    prob_deque.popleft()

                for (x, y, w, h) in faces:
                    face_patch = frame[y:y+h, x:x+w]
                    inp = preprocess_face(face_patch, img_size=model.input_shape[1], model_channels=model.input_shape[-1])
                    inp_batch = np.expand_dims(inp, 0)
                    probs = model.predict(inp_batch, verbose=0)[0]
                    label = labels[np.argmax(probs)]
                    conf = np.max(probs)
                    prob_deque.append((now, probs))

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf*100:.1f}%", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                agg_label, agg_conf = aggregate_probs(prob_deque, labels)
                if agg_label:
                    cv2.putText(frame, f"Smoothed: {agg_label} {agg_conf*100:.1f}%",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            cv2.destroyAllWindows()

            if agg_label:
                st.success(f"🧠 Final Detected Emotion: **{agg_label.upper()}** ({agg_conf*100:.1f}%)")
                recs = recommend_music(agg_label)
                st.markdown("### 🎵 Recommended Songs")
                for _, row in recs.iterrows():
                    with st.container():
                        st.markdown(
                            f"<div class='recommend-box'>"
                            f"🎵 **{row.get('track','Untitled')}** by *{row.get('artist','Unknown Artist')}*  "
                            f"<br>🎧 Genre: {row.get('genre','')} / {row.get('genre2','')} "
                            f"<br>🔗 [Listen Here]({row.get('lastfm_url','')})"
                            f"</div>", unsafe_allow_html=True
                        )
            else:
                st.warning("⚠ No emotion detected. Try again.")

# ==============================
# 💬 Text-Based Emotion Detection
# ==============================
elif mode == "💬 Text Mode":
    user_text = st.text_area("💭 Type your thoughts or feelings:", placeholder="e.g., I feel energetic and ready to dance!")
    if st.button("🔍 Analyze & Recommend"):
        if user_text.strip():
            emotion, score = text_to_emotion(user_text)
            st.success(f"🧠 Detected Emotion: **{emotion.upper()}** ({score*100:.1f}%)")
            recs = recommend_music(emotion)
            st.markdown("### 🎵 Recommended Songs")
            for _, row in recs.iterrows():
                with st.container():
                    st.markdown(
                        f"<div class='recommend-box'>"
                        f"🎵 **{row.get('track','Untitled')}** by *{row.get('artist','Unknown Artist')}*  "
                        f"<br>🎧 Genre: {row.get('genre','')} / {row.get('genre2','')} "
                        f"<br>🔗 [Listen Here]({row.get('lastfm_url','')})"
                        f"</div>", unsafe_allow_html=True
                    )
        else:
            st.warning("Please enter some text to analyze.")
