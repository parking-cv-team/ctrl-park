import streamlit as st
import requests
import os
import cv2
import base64
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

        # create the other button to shock scatterplot of tracked items
        request_tracking_plots(list(filter(lambda x: x["name"]==option, rows))[0]['id'])

    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")

# logic to make API request and display the plot
def request_tracking_plots(camera_id):
    if st.button("Show tracking map"):
        frame = get_first_frame()
        payload = {"camera_id": camera_id}
        if frame is not None:
            _, buf = cv2.imencode(".png", frame)
            payload["frame"] = base64.b64encode(buf).decode("utf-8")
        r = requests.post(f"{API_BASE}/analytics/trajectory_analysis", json=payload)

        if r.status_code == 200:
            st.image(r.content, caption="Tracking Scatterplot", width="stretch")
        else:
            st.error(f"Failed to Fetch Plot: {r.status_code}")

# get first frame of RTSP stream
def get_first_frame():
    # this should be the logic to obtain the first frame of a RTSP stream, to check if it actually works
    # if it doesn't, replace with a working logic and uhhhhhhh
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        return None

    try:
        for i in range(30):
            ret, frame = cap.read()
            
        while True:
            ret, frame = cap.read()
            if ret:
                break
    finally:
        cap.release()

    if ret:
        return frame
    else:
        return None


@st.fragment(run_every=10)
def draw_table(camera):
    try:
        response = requests.get(f"{API_BASE}/analytics/zones",params={"camera_id":camera["id"]})
        response.raise_for_status()
        rows = response.json()
        zone = { str(i["zone"]):i["occupancy"] for i in rows}
        st.table(zone)

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

