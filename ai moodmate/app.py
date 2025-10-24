from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import pandas as pd
import time
from collections import deque
from tensorflow.keras.models import load_model
from transformers import pipeline
import torch

# --------------------------------------------
# 🔧 Setup Flask
# --------------------------------------------
app = Flask(__name__)

# --------------------------------------------
# 🎭 Load Face Emotion Model
# --------------------------------------------
model_path = "C:\\Users\\adity\\OneDrive\\Desktop\\ai moodmate\\mobilenet.keras"
model = load_model(model_path)
labels = ["angry", "happy", "sad", "surprise", "neutral"]
input_shape = model.input_shape
img_size = input_shape[1]
channels = input_shape[-1]
print("✅ Loaded model:", model_path, "→ expects", input_shape)

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

# --------------------------------------------
# 🎵 Load Music Dataset
# --------------------------------------------
music_df = pd.read_csv("C:\\Users\\adity\\Downloads\\muse_v3.csv")
music_df.fillna("", inplace=True)
for col in ["genre", "genre2"]:
    if col in music_df.columns:
        music_df[col] = music_df[col].astype(str).str.lower()

emotion_map = {
    "happy": ["happy", "energetic", "upbeat", "party"],
    "sad": ["sad", "melancholic", "emotional", "calm"],
    "angry": ["aggressive", "intense", "rock", "metal"],
    "surprise": ["exciting", "curious", "fast", "adventurous"],
    "neutral": ["chill", "relaxing", "balanced", "ambient"]
}

# --------------------------------------------
# 🤗 Load NLP Emotion Model
# --------------------------------------------
nlp_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    framework="pt"
)
model_to_custom_map = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
    "fear": "neutral",
    "love": "happy"
}

def text_to_emotion(text):
    result = nlp_model(text)[0]
    label = result["label"].lower()
    mapped = model_to_custom_map.get(label, "neutral")
    print(f"🧠 Text: {text} → {label} → {mapped}")
    return mapped

# --------------------------------------------
# 🎶 Music Recommender (fixed substring + fallback)
# --------------------------------------------
def recommend_music(emotion, n=20):
    emotion = emotion.lower()
    tags = emotion_map.get(emotion, [])
    if not tags:
        print(f"⚠ No tags for emotion: {emotion}")
        return music_df.sample(min(len(music_df), n)).to_dict(orient="records")

    genre_cols = [col for col in ["genre", "genre2"] if col in music_df.columns]
    # ✅ Substring match instead of exact match
    mask = pd.concat([
        music_df[col].apply(lambda x: any(tag in str(x) for tag in tags))
        for col in genre_cols
    ], axis=1).any(axis=1)

    filtered = music_df[mask]
    if filtered.empty:
        print(f"⚠ No matches for {emotion}, returning random sample.")
        filtered = music_df.sample(min(len(music_df), n))

    recs = filtered.sample(min(len(filtered), n))
    return recs[["artist", "track", "genre", "genre2", "lastfm_url"]].to_dict(orient="records")

# --------------------------------------------
# 🎥 Webcam Emotion Logic
# --------------------------------------------
last_emotion = "neutral"
stable_emotion = "neutral"
emotion_stable = False
prob_deque = deque(maxlen=30)

def gen_frames():
    global last_emotion, stable_emotion, emotion_stable
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        now = time.time()

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            inp = preprocess_face(face, img_size=img_size, model_channels=channels)
            inp_batch = np.expand_dims(inp, 0)
            probs = model.predict(inp_batch, verbose=0)[0]
            label = labels[np.argmax(probs)]
            prob_deque.append((now, probs))
            last_emotion = label

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        agg_label, agg_conf = aggregate_probs(prob_deque, labels)
        if agg_label:
            stable_emotion = agg_label
            emotion_stable = True
            cv2.putText(frame, f"Smoothed: {agg_label} ({agg_conf*100:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        frame_count += 1
        if frame_count >= 300 or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------------
# 🌐 Routes
# --------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    global stable_emotion, emotion_stable

    user_text = request.form.get("text", "").strip()
    if not user_text and not emotion_stable:
        return jsonify({"error": "Please enter text or wait for webcam emotion."})

    if user_text:
        final_emotion = text_to_emotion(user_text)
    elif emotion_stable:
        final_emotion = stable_emotion
        emotion_stable = False
    else:
        final_emotion = "neutral"

    songs = recommend_music(final_emotion)
    return jsonify({"emotion": final_emotion, "songs": songs})

@app.route('/emotion')
def get_emotion():
    global stable_emotion, emotion_stable, last_emotion
    emotion = stable_emotion if emotion_stable else last_emotion
    return jsonify({"emotion": emotion})

# --------------------------------------------
# 🚀 Run
# --------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
