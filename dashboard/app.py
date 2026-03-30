import streamlit as st
import requests
import os
import cv2
import base64
from dotenv import load_dotenv
import pandas as pd 
import seaborn as sns
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import numpy as np
from db.models import Zone

# current dashboard structure:
#
#   body:
#       camera_form()
#       camera_button():
#           number_of_cars()
#           button_to_draw_zones()
#           camera_video()
#           trajectory+heatmap()
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


        option = st.selectbox(
            "## Choose a camera to see related data:", tuple([i["name"] for i in rows])
        )

        if st.button("Confirm Selection"):
            st.session_state.confirmed_camera = list(filter(lambda x: x["name"] == option, rows))[0]
            st.session_state.show_zones = False
            st.session_state.show_tracking_map = False

    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")


def camera_selected():
    camera = st.session_state.confirmed_camera

    if st.session_state.confirmed_camera is not None:
        
        st.success(f"You confirmed: {st.session_state.confirmed_camera}")

        number_of_cars(camera)

        
        cap, placeholder, zones = place_a_video(camera)

        # bottoni per visualizzare le zone e visualizzare le heatmaps
        col_bot = st.columns(2)
        with col_bot[0]:
            if st.button("Show zones"):
                st.session_state.show_zones = not st.session_state.show_zones
        with col_bot[1]:
            if st.button("Show tracking map"):
                st.session_state.show_tracking_map = (
                    not st.session_state.show_tracking_map
                )
        
        if st.session_state.show_tracking_map:
            st.text("## Tracking Plots and Heatmaps")
            request_tracking_plots(camera["id"])
        
        # tabella occupazioni
        draw_table(camera)

        # create button to create main metrics report (KPI, time series blablabla)
        st.text("## Metrics Report and Time Series")
        cols = st.columns(2)
        with cols[0]:
            # setto il tempo iniziale a 24 ore prima
            default_dt = datetime.now() - timedelta(days=1)
            ti = st.datetime_input("Time Start",value=default_dt)
        with cols[1]:
            tf = st.datetime_input("Time End")

        request_report(camera['id'], ti, tf)
        
        request_timeseries(camera['id'], ti, tf)
        
        veicoli_fuori(camera)

        continue_video(cap, placeholder, zones)
    else:
        st.info("Please select an option and click the button.")


def veicoli_fuori(camera):
    try:
        response = requests.get(
            f"{API_BASE}/analytics/cameras/recent/outside_zones", params={"camera_id": camera["id"]}
        )
        response.raise_for_status()
        data = response.json()
        seen = set()
        unique = []

        for x in data:
            if x["event_type"]=="departure":
                seen.add(x)
            if x["tracker_id"] in seen:
                continue
            unique.append(x)
            seen.add(x["tracker_id"])
        
        for x in unique:
            st.write(f"Found {x["class"]} outside zones at x:{x["cx"]} y:{x["cy"]} with tracking id: {x["tracker_id"]} at timestamp: {x["time"]}")


    except Exception as e:
        st.error(f"Could not get cars outside parking zones: {e}")
    pass

# logic to make API request and display the plot
def request_tracking_plots(camera_id):
    # if st.button("Show tracking map"):
    frame = get_first_frame()
    payload = {"camera_id": camera_id}
    if frame is not None:
        _, buf = cv2.imencode(".png", frame)
        payload["frame"] = base64.b64encode(buf).decode("utf-8")  # type: ignore
    r = requests.post(f"{API_BASE}/analytics/trajectory_analysis", json=payload)

    if r.status_code == 200:
        st.image(r.content, caption="Tracking Scatterplot", width="stretch")  # type: ignore
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

# logic to make API request to get main metrics
def request_report(camera_id, t_i, t_f):
    if st.button("Get Metrics Report"):
        # step 1: KPIs
        r = requests.get(f"{API_BASE}/analytics/metrics_report/kpi", params={'camera_id': camera_id,
                                                                         't_start': t_i,
                                                                         't_end': t_f})
        
        if r.status_code == 200:
            r_s = r.content.decode()
            r_j = json.loads(r_s)

            st.dataframe(pd.DataFrame(r_j['total_tracked_by_class']))
            st.dataframe(pd.DataFrame(r_j['avg_confidence_by_class']))
            st.dataframe(pd.DataFrame(r_j['total_zones']))
            st.dataframe(pd.DataFrame(r_j['max_occupations']))
            st.dataframe(pd.DataFrame(r_j['avg_occupations']))
            st.dataframe(pd.DataFrame(r_j['avg_track_time']))
            st.dataframe(pd.DataFrame(r_j['avg_confidence']))
            st.dataframe(pd.DataFrame(r_j['n_departures']))
            st.dataframe(pd.DataFrame(r_j['n_tracked_det']))

        else:
            st.error(f"Could not fetch summary... {r.status_code}")            


def request_timeseries(camera_id, t_i, t_f):
    # TODO: see if i can map the timestamps better, integers make no sense...
    
    if st.button("Get Timeseries Report"):
        # step 1: KPIs
        r = requests.get(f"{API_BASE}/analytics/metrics_report/timeseries", params={'camera_id': camera_id,
                                                                         't_start': t_i,
                                                                         't_end': t_f})
        
        if r.status_code == 200:
            r_s = r.content.decode()
            r_j = json.loads(r_s)

            ts_1 = (pd.DataFrame(r_j['ts_confidence']))
            ts_2 = (pd.DataFrame(r_j['ts_objects']))
            ts_3 = (pd.DataFrame(r_j['ts_parked']))

            fig, axes = plt.subplots(3, 1, figsize=(5, 6))

            fig.suptitle("Timeseries Reports")

            axes[0].set_title("Average Confidence")
            axes[1].set_title("Number of tracked objects")
            axes[2].set_title("Number of parked vehicles")

            sns.lineplot(data=ts_1.reset_index(), ax=axes[0], x=ts_1.index, y="avg_confidence", hue="class_name", palette="Set1")
            sns.lineplot(data=ts_2.reset_index(), ax=axes[1], x=ts_2.index, y="num_tracked", hue="class_name", palette="Set1")
            sns.lineplot(data=ts_3.reset_index(), ax=axes[2], x=ts_3.index, y="num_parked_vehicles", palette="Set1")

            axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[2].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            axes[0].yaxis.grid()
            axes[1].yaxis.grid()
            axes[2].yaxis.grid()

            fig.tight_layout()

            st.pyplot(fig)

        else:
            st.error(f"Could not fetch summary... {r.status_code}")            


def get_zones_to_draw(camera):
    try:
        response = requests.get(
            f"{API_BASE}/analytics/zones/poly", params={"camera_id": camera["id"]}
        )
        response.raise_for_status()
        rows = response.json()
        return list(rows)

    except Exception as e:
        st.error(f"Could not fetch zones to draw: {e}")


@st.fragment(run_every=10)
def draw_table(camera):
    try:
        response = requests.get(
            f"{API_BASE}/analytics/zones", params={"camera_id": camera["id"]}
        )
        response.raise_for_status()
        rows = response.json()
        zone = {str(i["zone"]): i["occupancy"] for i in rows}
        st.table(zone)

    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")


def place_a_video(camera):
    uri = camera["uri"]
    cap = cv2.VideoCapture(uri)
    placeholder = st.empty()
    if not cap.isOpened():
        st.error(f"Could not open stream: {uri}")
        return (None,placeholder,[])

    
    zones = get_zones_to_draw(camera)
    return (cap, placeholder, zones)


def draw_zones(canvas: np.ndarray, completed: list[Zone]) -> np.ndarray:
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


def continue_video(cap, placeholder, zones=[]):
    zones = [
        Zone(
            id=int(z["id"]),
            name=str(z["name"]),
            polygon=np.array(z["polygon"], dtype=np.int32),
            camera_id=["camera_id"],
        )
        for z in zones
    ]

    while True and not cap is None:
        ret, frame = cap.read()
        if st.session_state.show_zones:
            frame = draw_zones(frame, zones)
        if not ret:
            break
        placeholder.image(frame, channels="BGR")
    if not cap is None:
        cap.release()


@st.fragment(run_every=10)
def number_of_cars(camera):
    try:
        response = requests.get(
            f"{API_BASE}/analytics/cameras/recent", params={"camera_id": camera["id"]}
        )
        response.raise_for_status()
        data = response.json()
        seen = set()
        unique = []

        for x in data:
            if x["event_type"]=="departure":
                seen.add(x)
            if x["tracker_id"] in seen:
                continue
            unique.append(x)
            seen.add(x["tracker_id"])
        

        st.write(
            f"#### Number of parked cars seen by {camera['name']}: {len(unique)}"
        )

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
    if "confirmed_camera" not in st.session_state:
            st.session_state.confirmed_camera = None
    if "show_zones" not in st.session_state:
        st.session_state.show_zones = False
    if "show_tracking_map" not in st.session_state:
        st.session_state.show_tracking_map = False
    camera_form()
    camera_button()
    camera_selected()


body()
