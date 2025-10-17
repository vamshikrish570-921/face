import cv2
import streamlit as st
from deepface import DeepFace
import numpy as np

st.set_page_config(page_title="Facial Expression Recognition", layout="wide")
st.title("😊 Real-Time Facial Expression Recognition (DeepFace)")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        cv2.putText(frame, dominant_emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception:
        pass

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
st.write("Camera stopped.")
