import streamlit as st
import requests
import os
import cv2
from dotenv import load_dotenv


load_dotenv()


RTSP_URL = "rtsp://localhost:8554/live.stream"



def camera_button():
    try:
        response = requests.get(f"{API_BASE}/analytics/cameras")
        response.raise_for_status()
        rows = response.json()
        
        # Create the dropdown
        option = st.selectbox(
            '## Choose a camera to see occupancy:',tuple([i["name"] for i in rows])
        )

        # Create the button
        if st.button('Confirm Selection'):
            # Logic to execute when button is clicked
            st.success(f'You confirmed: {option}')
            camera = list(filter(lambda x: x["name"]==option, rows))[0]
            number_of_cars(camera)
            draw_table(camera)
            
        else:
            st.info('Please select an option and click the button.')
        
    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")
    
    

@st.fragment(run_every=10)
def draw_table(camera):
    try:
        response = requests.get(f"{API_BASE}/analytics/zones",params={"camera_id":camera["id"]})
        response.raise_for_status()
        rows = response.json()
        banana = { i["zone"]:i["occupancy"] for i in rows}
        st.table(banana)

    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")


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

@st.fragment(run_every=10)
def number_of_cars(camera):
    try:
        response = requests.get(f"{API_BASE}/analytics/cameras/recent",params={"camera_id":camera["id"]})
        response.raise_for_status()
        st.write(f"Number of cars in the last minute seen by {camera["name"]}: {response.text}")

    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")
    pass


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Ctrl+Park Dashboard")



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


#place_a_video()
camera_button()