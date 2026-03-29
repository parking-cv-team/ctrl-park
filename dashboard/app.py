import streamlit as st
import requests
import os
import cv2
import base64
from dotenv import load_dotenv
import numpy as np
from db.models import Zone

#current dashboard structure:
# 
#   body:
#       camera_form()
#       camera_button():
#           number_of_cars()
#           button_to_draw_zones()
#           camera_video()
#           occupancy_table()



load_dotenv()

RTSP_URL = "rtsp://localhost:8554/live.stream"
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
_ZONE_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (0, 165, 255),  # orange
    (255, 0, 0),  # blue
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 128),  # purple
]


def camera_button():
    
    try:
        response = requests.get(f"{API_BASE}/analytics/cameras")
        response.raise_for_status()
        rows = response.json()

        if "confirmed_camera" not in st.session_state:
            st.session_state.confirmed_camera = None
        if "show_zones" not in st.session_state:
            st.session_state.show_zones = False

        option = st.selectbox(
            '## Choose a camera to see occupancy:', tuple([i["name"] for i in rows])
        )

        if st.button('Confirm Selection'):
            st.session_state.confirmed_camera = option
            st.session_state.show_zones = False
        


        if st.session_state.confirmed_camera is not None:
            camera = list(filter(lambda x: x["name"]==st.session_state.confirmed_camera,rows))[0]
            st.success(f'You confirmed: {st.session_state.confirmed_camera}')


            number_of_cars(camera)

            if st.button('Show zones'):
                st.session_state.show_zones = not st.session_state.show_zones

            cap, placeholder, zones = place_a_video(camera)
            draw_table(camera)
            continue_video(cap, placeholder, zones)
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

    
def get_zones_to_draw(camera):
    try:
        response = requests.get(f"{API_BASE}/analytics/zones/poly",params={"camera_id":camera["id"]})
        response.raise_for_status()
        rows = response.json()
        return list(rows)

    except Exception as e:
        st.error(f"Could not fetch zones to draw: {e}")

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

def place_a_video(camera):
    uri = camera["uri"]
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        st.error(f"Could not open stream: {uri}")
        return

    placeholder = st.empty()
    zones = get_zones_to_draw(camera)
    return (cap,placeholder,zones)

def draw_zones(
    canvas: np.ndarray,
    completed: list[Zone]) -> np.ndarray:
    """Render zones onto a copy of canvas."""
    out = canvas.copy()
    overlay = canvas.copy()

    for idx, zone in enumerate(completed):
        color = _ZONE_COLORS[idx % len(_ZONE_COLORS)]
        pts = zone.polygon.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        centroid = zone.polygon.mean(axis=0).astype(int)
        cv2.putText(
            out,
            str(zone.name),
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    
    return out

def continue_video(cap,placeholder,zones=[]):
    zones=[Zone(id=int(z["id"]), name=str(z["name"]), polygon=np.array(z["polygon"], dtype=np.int32), camera_id=["camera_id"]) for z in zones]

    while True:
        ret, frame = cap.read()
        if len(zones)!=0:
            frame = draw_zones(frame,zones)
        if not ret:
            break
        placeholder.image(frame, channels="BGR")
    cap.release()

@st.fragment(run_every=10)
def number_of_cars(camera):
    try:
        response = requests.get(f"{API_BASE}/analytics/cameras/recent",params={"camera_id":camera["id"]})
        response.raise_for_status()
        st.write(f"### Number of cars in the last minute seen by {camera["name"]}: {response.text}")

    except Exception as e:
        st.error(f"Could not fetch the number of cars: {e}")
    pass

def camera_form():
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

def body():
    st.title("Ctrl+Park Dashboard")
    camera_form()
    camera_button()



body()
