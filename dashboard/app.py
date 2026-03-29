import streamlit as st
import requests
import os
import cv2
import base64
from dotenv import load_dotenv
import pandas as pd 
import seaborn as sns
import json

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

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
        st.text("## Tracking Plots and Heatmaps")

        request_tracking_plots(list(filter(lambda x: x["name"]==option, rows))[0]['id'])

        # create button to create main metrics report (KPI, time series blablabla)
        st.text("## Metrics Report and Time Series")
        cols = st.columns(2)
        with cols[0]:
            ti = st.datetime_input("Time Start")
        with cols[1]:
            tf = st.datetime_input("Time End")

        request_report(list(filter(lambda x: x["name"]==option, rows))[0]['id'],
                       ti, tf)
        
        request_timeseries(list(filter(lambda x: x["name"]==option, rows))[0]['id'],
                       ti, tf)

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

