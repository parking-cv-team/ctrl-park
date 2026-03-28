import streamlit as st
import requests
import os
import cv2
from dotenv import load_dotenv


load_dotenv()


RTSP_URL = "rtsp://localhost:8554/live.stream"


def place_a_video():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        st.error(f"Could not open stream: {RTSP_URL}")
        return

    placeholder = st.empty()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            placeholder.image(frame, channels="BGR")
    finally:
        cap.release()


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Ctrl+Park Dashboard")

st.write("## Recent processed frames")

try:
    response = requests.get(f"{API_BASE}/analytics/recent")
    response.raise_for_status()
    rows = response.json()
    for r in rows:
        st.write(r)
except Exception as e:
    st.error(f"Could not fetch analytics: {e}")

st.write("## Add Camera Source")
with st.form("camera-form"):
    name = st.text_input("Camera name")
    uri = st.text_input("Camera URI")
    submitted = st.form_submit_button("Register")
    if submitted:
        try:
            r = requests.post(f"{API_BASE}/camera", json={"name": name, "uri": uri})
            r.raise_for_status()
            st.success("Camera registered")
        except Exception as e:
            st.error(f"Error: {e}")


place_a_video()
