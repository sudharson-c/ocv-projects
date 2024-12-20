import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit app
st.title("Real-Time Emotion Detection")
st.write("This app uses your webcam to detect emotions in real-time.")

# Sidebar options
st.sidebar.title("Settings")
run_webcam = st.sidebar.checkbox("Use Webcam", value=True)

# Emotion Detection Function
def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotions = []

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emotions.append((x, y, w, h, emotion))
    return emotions

# Webcam feed
if run_webcam:
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access the webcam.")
            break
        emotions = detect_emotions(frame)

        for (x, y, w, h, emotion) in emotions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file,)
        frame = np.array(img)

        emotions = detect_emotions(frame)

        for (x, y, w, h, emotion) in emotions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        st.image(frame, caption="Emotion Detection Result",width=500)
