import streamlit as st
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
from PIL import Image

# Initialize Segmentor
segmentor = SelfiSegmentation()

# Streamlit app
st.title("Real-Time Background Removal")
st.write("Upload an image or use your webcam to remove the background dynamically.")

# Sidebar for user input
st.sidebar.title("Settings")
bg_option = st.sidebar.selectbox("Background Type", ["White", "Black", "Custom Image"])
custom_bg = None

if bg_option == "Custom Image":
    custom_bg_file = st.sidebar.file_uploader("Upload Background Image", type=["jpg", "png"])
    if custom_bg_file:
        custom_bg = Image.open(custom_bg_file)
        custom_bg = np.array(custom_bg)

# Video capture or image upload
use_webcam = st.sidebar.checkbox("Use Webcam")
if use_webcam:
    run = st.checkbox("Run Webcam")
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Webcam not found.")
            break

        # Apply background removal
        if bg_option == "White":
            bg = (255, 255, 255)
        elif bg_option == "Black":
            bg = (0, 0, 0)
        elif bg_option == "Custom Image" and custom_bg is not None:
            bg = cv2.resize(custom_bg, (frame.shape[1], frame.shape[0]))
            if bg.shape[2] == 4:  
                bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)
        else:
            bg = (255, 255, 255)
        img_out = segmentor.removeBG(frame, bg,0.45)
        FRAME_WINDOW.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
else:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_np = np.array(img)

        # Apply background removal
        if bg_option == "White":
            bg = (255, 255, 255)
        elif bg_option == "Black":
            bg = (0, 0, 0)
        elif bg_option == "Custom Image" and custom_bg is not None:
            bg = cv2.resize(custom_bg, (img_np.shape[1], img_np.shape[0]))
            
        else:
            bg = (255, 255, 255)

        img_out = segmentor.removeBG(img_np, bg,0.45)
        st.image(img_out, caption="Background Removed", use_column_width=True)

