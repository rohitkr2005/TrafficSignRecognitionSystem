# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import cv2

# -----------------------------
# 1. Load Model and Labels
# -----------------------------
MODEL_PATH = "gtsrb_baseline_final.h5"
LABEL_MAP_PATH = "label_map.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    # Ensure keys are strings
    return {str(k): v for k, v in label_map.items()}

model = load_model()
label_map = load_labels()

IMG_SIZE = 32  # model input size

# -----------------------------
# 2. Helper Functions
# -----------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(img_array):
    preds = model.predict(img_array, verbose=0)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_id]
    class_name = label_map.get(str(class_id), str(class_id))
    
    # Top 3 predictions
    top3_idx = preds[0].argsort()[-3:][::-1]
    top3 = [(label_map.get(str(i), str(i)), preds[0][i]) for i in top3_idx]
    return class_name, confidence, top3

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🛑 Traffic Sign Recognition")
st.write("Upload an image or use your webcam to detect traffic signs in real-time.")

option = st.radio("Select input method:", ["Upload Image", "Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(img)
        class_name, confidence, top3 = predict(img_array)

        st.success(f"Predicted Sign: **{class_name}**")
        st.info(f"Confidence: {confidence*100:.2f}%")

        st.write("Top 3 Predictions:")
        for name, conf in top3:
            st.write(f"{name}: {conf*100:.2f}%")

elif option == "Webcam":
    st.write("Click start to activate your webcam. Press 'q' to stop.")
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame from webcam.")
            break

        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_array = preprocess_image(Image.fromarray(img))
        class_name, confidence, top3 = predict(img_array)

        # Overlay prediction on frame
        cv2.putText(frame, f"{class_name} ({confidence*100:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()