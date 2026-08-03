import streamlit as st
from emotion.face_detector import run_face_detection
from emotion.emotion_detector import run_emotion_detection

st.set_page_config(page_title="AI Interview Analyzer")

st.title("AI Interview Emotion & Confidence Analyzer")

st.write("AI Powered Interview Analysis System")

name = st.text_input("Enter your name")

if name:
    st.success(f"Welcome {name}")

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Face Detection"):
        run_face_detection()

with col2:
    if st.button("Start Emotion Detection"):
        run_emotion_detection()
