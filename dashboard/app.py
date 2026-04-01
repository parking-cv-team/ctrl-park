import streamlit as st
import streamlit.components.v1 as components
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
import time
from multiprocessing import Process
from dashboard.calbrate_camera import run_zone_creator
import matplotlib.dates as mdates
import subprocess
from dashboard.camera_merger import merge_cameras


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
#           3D viewer

load_dotenv()

RTSP_URL = "rtsp://localhost:8554/live.stream"
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ctrl+Park",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg0:       #0d1117;
    --bg1:       #161b27;
    --bg2:       #1e2535;
    --bg3:       #28304a;
    --line:      #2e3a52;
    --txt0:      #e6edf3;
    --txt1:      #8b949e;
    --txt2:      #484f58;
    --blue:      #58a6ff;
    --blue-bg:   #1c2d40;
    --green:     #3fb950;
    --green-bg:  #0d2818;
    --red:       #f85149;
    --red-bg:    #2d1217;
    --amber:     #e3b341;
    --amber-bg:  #2d1f06;
    --r:         6px;
    --r-lg:      10px;
    --font:      'IBM Plex Sans', system-ui, sans-serif;
    --mono:      'IBM Plex Mono', monospace;
}

/* ── Global ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: var(--bg0) !important;
    color: var(--txt0) !important;
    font-family: var(--font) !important;
    -webkit-font-smoothing: antialiased;
}

.block-container {
    max-width: 1060px !important;
    padding: 2rem 2rem 4rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg1) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.25rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: var(--txt0) !important;
    font-family: var(--font) !important;
}

/* ── Typography ── */
h1 {
    font-family: var(--font) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--txt0) !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 0.2rem !important;
    padding: 0 !important;
}
h2 {
    font-family: var(--font) !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--txt0) !important;
    margin: 1.75rem 0 0.4rem !important;
    padding: 0 !important;
}
h3 {
    font-family: var(--font) !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: var(--txt0) !important;
    margin: 1.25rem 0 0.35rem !important;
    padding: 0 !important;
}
h4 {
    font-family: var(--font) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--txt1) !important;
    margin: 1rem 0 0.3rem !important;
    padding: 0 !important;
}
hr {
    border: none !important;
    border-top: 1px solid var(--line) !important;
    margin: 1.5rem 0 !important;
}

/* ── Labels ── */
label,
[data-testid="stWidgetLabel"] p {
    font-family: var(--font) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--txt1) !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input {
    background-color: var(--bg2) !important;
    color: var(--txt0) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--txt2) !important; }
[data-testid="stTextInput"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.2) !important;
    outline: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: var(--bg2) !important;
    color: var(--txt0) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.2) !important;
}
[data-baseweb="popover"] ul,
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: var(--bg2) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
}
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"] {
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
    color: var(--txt0) !important;
    background-color: transparent !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] [role="option"]:hover {
    background-color: var(--bg3) !important;
}

/* ── Datetime input ── */
[data-testid="stDatetimeInput"] input {
    background-color: var(--bg2) !important;
    color: var(--txt0) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
}
[data-testid="stDatetimeInput"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.2) !important;
}

/* ── Buttons ── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    border-radius: var(--r) !important;
    padding: 0.5rem 1.1rem !important;
    min-height: 2.25rem !important;
    transition: background 0.15s, border-color 0.15s !important;
}
/* Default */
.stButton > button {
    background-color: var(--bg2) !important;
    color: var(--txt0) !important;
    border: 1px solid var(--line) !important;
}
.stButton > button:hover {
    background-color: var(--bg3) !important;
    border-color: var(--txt1) !important;
    color: var(--txt0) !important;
}
/* Primary — solid blue, dark text, always legible */
.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button {
    background-color: var(--blue) !important;
    color: #0d1117 !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #79b8ff !important;
    color: #0d1117 !important;
}
.stButton > button:active { transform: translateY(1px) !important; }

/* ── Form container ── */
[data-testid="stForm"] {
    background-color: var(--bg1) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r-lg) !important;
    padding: 1.25rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--line) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    color: var(--txt2) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: -1px !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--txt1) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--txt0) !important;
    font-weight: 500 !important;
    border-bottom-color: var(--blue) !important;
}
[data-testid="stTabs"] [role="tabpanel"] {
    padding-top: 1.25rem !important;
    background: transparent !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background-color: var(--bg1) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r-lg) !important;
    padding: 1rem 1.1rem !important;
    transition: border-color 0.15s !important;
}
[data-testid="stMetric"]:hover { border-color: var(--blue) !important; }
[data-testid="stMetricLabel"] p {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: var(--txt2) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 500 !important;
    color: var(--txt0) !important;
    letter-spacing: -0.02em !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] * {
    font-family: var(--font) !important;
    font-size: 0.83rem !important;
    background-color: var(--bg1) !important;
    color: var(--txt0) !important;
}

/* ── st.table ── */
[data-testid="stTable"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-family: var(--font) !important;
    font-size: 0.85rem !important;
    border: 1px solid var(--line) !important;
    overflow: hidden !important;
}
[data-testid="stTable"] th {
    background-color: var(--bg2) !important;
    color: var(--txt2) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    padding: 0.6rem 0.9rem !important;
    border-bottom: 1px solid var(--line) !important;
    text-align: left !important;
}
[data-testid="stTable"] td {
    padding: 0.6rem 0.9rem !important;
    color: var(--txt0) !important;
    background-color: var(--bg1) !important;
    border-bottom: 1px solid var(--line) !important;
}
[data-testid="stTable"] tr:last-child td { border-bottom: none !important; }
[data-testid="stTable"] tr:hover td { background-color: var(--bg2) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    font-family: var(--font) !important;
    font-size: 0.85rem !important;
    border-radius: var(--r) !important;
    border-width: 1px !important;
    border-style: solid !important;
    padding: 0.7rem 0.9rem !important;
}
[data-testid="stAlert"] p { color: inherit !important; font-size: 0.85rem !important; }
.stSuccess > div { background-color: var(--green-bg) !important; border: 1px solid var(--green) !important; border-radius: var(--r) !important; color: var(--green) !important; }
.stError   > div { background-color: var(--red-bg)   !important; border: 1px solid var(--red)   !important; border-radius: var(--r) !important; color: var(--red)   !important; }
.stInfo    > div { background-color: var(--blue-bg)  !important; border: 1px solid var(--blue)  !important; border-radius: var(--r) !important; color: var(--blue)  !important; }
.stWarning > div { background-color: var(--amber-bg) !important; border: 1px solid var(--amber) !important; border-radius: var(--r) !important; color: var(--amber) !important; }
.stSuccess > div p { color: var(--green) !important; }
.stError   > div p { color: var(--red)   !important; }
.stInfo    > div p { color: var(--blue)  !important; }
.stWarning > div p { color: var(--amber) !important; }

/* ── Column layout ── */
[data-testid="stHorizontalBlock"] {
    gap: 0.75rem !important;
    align-items: flex-start !important;
}
[data-testid="stColumn"] { padding: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--txt2); }

/* ── Sidebar section divider ── */
.sb-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--txt2);
    margin: 1.5rem 0 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sb-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line);
}

/* ── Live badge ── */
.cp-live {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 5px;
    padding: 0.35rem 0.75rem;
    font-family: var(--font);
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--green);
    margin-bottom: 1rem;
}
.cp-live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 2px rgba(63,185,80,0.25);
    flex-shrink: 0;
}

/* ── Section divider ── */
.cp-divlabel {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 0.75rem;
    font-family: var(--font);
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--txt2);
}
.cp-divlabel::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line);
}
</style>
"""

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#161b27",
    "axes.facecolor":   "#1e2535",
    "axes.edgecolor":   "#2e3a52",
    "axes.labelcolor":  "#8b949e",
    "axes.labelsize":   9,
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "text.color":       "#e6edf3",
    "grid.color":       "#2e3a52",
    "grid.linewidth":   0.8,
    "legend.facecolor": "#1e2535",
    "legend.edgecolor": "#2e3a52",
    "legend.fontsize":  8,
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       130,
    "axes.titlesize":   10,
    "axes.titlecolor":  "#e6edf3",
})


def camera_button():

    # Initialize session state first (before any API calls)
    if "confirmed_camera" not in st.session_state:
        st.session_state.confirmed_camera = None
    if "show_zones" not in st.session_state:
        st.session_state.show_zones = False
    if "show_tracking_map" not in st.session_state:
        st.session_state.show_tracking_map = False

    try:
        response = requests.get(f"{API_BASE}/analytics/cameras")
        response.raise_for_status()
        rows = response.json()

        st.markdown("## Camera Choice")
        st.markdown("In this part you will be able to choose a camera and get its relative analytics")
        option = st.selectbox(
            label="Choose a camera to see related data", options=tuple([i["name"] for i in rows])
        )

        if st.button("Confirm Selection"):
            st.session_state.confirmed_camera = list(
                filter(lambda x: x["name"] == option, rows)
            )[0]
            st.session_state.show_zones = False
            st.session_state.show_tracking_map = False
            st.session_state.show_kpis_live = False

    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")


def camera_selected():
    camera = st.session_state.confirmed_camera

    if st.session_state.confirmed_camera is not None:

        st.success(f"You confirmed: {st.session_state.confirmed_camera['name']}")

        tab_overview, tab_kpis, tab_tracking, tab_reports, tab_3d = st.tabs(
            [
                "Overview", "KPIs", "Tracking", "Reports", "3D Simulation"
            ]
        )
        with tab_overview:

            number_of_cars(camera)

            cap, placeholder, zones = place_a_video(camera)

            if play_vid := st.button("Play Video (with zones)"):
                subprocess.Popen(
                    ['python', 'dashboard/video_player.py', camera['uri'], str(camera['id'])]
                )

            # tabella occupazioni
            with st.empty():
                draw_table(camera)

            # veicoli fuori
            with st.empty():
                veicoli_fuori(camera)
        # bottoni per visualizzare le zone, visualizzare le heatmaps, e visualizzare le metriche (KPIs) in tempo reale
        with tab_kpis:
            request_kpis_live(camera['id'])
        
        with tab_reports:
            st.markdown("### Metrics Report and Time Series")
            cols = st.columns(2)
            with cols[0]:
                # setto il tempo iniziale a 24 ore prima
                default_dt = datetime.now() - timedelta(days=1)
                ti = st.datetime_input("Time Start", value=default_dt, key='ti', step = 60)
            with cols[1]:
                tf = st.datetime_input("Time End", key = 'tf', step=60)

            with st.empty():
                request_report(camera["id"], ti, tf)

            with st.empty():
                request_timeseries(camera["id"], ti, tf)

        with tab_tracking:
            request_tracking_plots(camera)

        with tab_3d:
            # 3d map logic
            mapped_zones = get_mapped_zones()
            singlecamera = False
            if not mapped_zones:
                st.warning("No mapped zones found. Attempting single camera mode")
                mapped_zones = get_mapped_zones(camera['id'], True)
                singlecamera = True
            if mapped_zones:
                display_3d_viewer(camera['id'], mapped_zones, single_camera=singlecamera)


    else:
        st.info("Please select an option and click the button.")


# helper function to display metric safely in case of exceptions
def safe_metric(widget, label, value_fn):
    try:
        widget.metric(label, value_fn())
    except Exception as e:
        widget.metric(label, "N/A")

@st.fragment(run_every=5)  
def request_kpis_live(camera_id):
    st.markdown("## Live KPIs")
    placeholder = st.empty()

    with placeholder.container():
        try:
            r = requests.get(
                f"{API_BASE}/analytics/metrics_report/kpi",
                params={
                    "camera_id": camera_id,
                    "t_start": (datetime.now() - timedelta(days=100)).isoformat(),
                    "t_end": datetime.now().isoformat(),
                },
                timeout=5,
            )
            r.raise_for_status()
        except Exception as e:
            st.error(f"Could not retrieve KPI metrics: {e}")
            return

        r_j = r.json()

        st.write("### Total Tracked")
        total_car, total_ped, total = st.columns(3)

        st.write("### Average Confidence")
        conf_car, conf_ped, conf_avg = st.columns(3)

        st.write("### Occupancies (total and normalized)")
        occ_max, occ_avg = st.columns(2)
        total_zones, occ_max_norm, occ_avg_norm = st.columns(3)

        st.write("### Average Time Tracked")
        avg_time_car, avg_time_ped = st.columns(2)

        st.write("### Amount of tracked objects and departures")
        n_tracked_cars, n_tracked_peds, n_departures = st.columns(3)

        df_total_tracked = pd.DataFrame(r_j["total_tracked_by_class"])
        df_avg_conf = pd.DataFrame(r_j["avg_confidence_by_class"])
        df_zones = pd.DataFrame(r_j["total_zones"])
        df_max_occs = pd.DataFrame(r_j["max_occupations"])
        df_avg_occs = pd.DataFrame(r_j["avg_occupations"])
        df_time = pd.DataFrame(r_j["avg_track_time"])
        df_ndep = pd.DataFrame(r_j["n_departures"])
        df_ntracked = pd.DataFrame(r_j["n_tracked_det"])

        safe_metric(total_car, "Cars", lambda: df_total_tracked.iloc[0, 1])  # pyright: ignore[reportArgumentType]
        safe_metric(total_ped, "Pedestrians", lambda: df_total_tracked.iloc[1, 1])  # pyright: ignore[reportArgumentType]
        safe_metric(total, "Both", lambda: df_total_tracked.iloc[0, 1] + df_total_tracked.iloc[1, 1])  # type: ignore

        safe_metric(conf_car, "Cars", lambda: f"{df_avg_conf.iloc[0, 1]:.3f}")
        safe_metric(conf_ped, "Pedestrians", lambda: f"{df_avg_conf.iloc[1, 1]:.3f}")
        safe_metric(conf_avg, "Average", lambda: f"{(df_avg_conf.iloc[0, 1] + df_avg_conf.iloc[1, 1]) * 0.5:.3f}")  # type: ignore

        safe_metric(occ_max, "Maximum Occupancies", lambda: df_max_occs.iloc[0, 0])  # type: ignore
        safe_metric(occ_avg, "Average Occupancies", lambda: df_avg_occs.iloc[0, 0])  # type: ignore

        safe_metric(total_zones, "Total zones", lambda: df_zones.iloc[0, 0])  # type: ignore
        safe_metric(occ_max_norm, "Maximum Occupancies (norm.)", lambda: f"{df_max_occs.iloc[0, 0] / df_zones.iloc[0, 0]:.3f}")  # pyright: ignore[reportOperatorIssue]
        safe_metric(occ_avg_norm, "Average Occupancies (norm.)", lambda: f"{df_avg_occs.iloc[0, 0] / df_zones.iloc[0, 0]:.3f}")  # type: ignore

        safe_metric(avg_time_car, "Cars", lambda: df_time.iloc[0, 1])  # type: ignore
        safe_metric(avg_time_ped, "Pedestrians", lambda: df_time.iloc[1, 1])  # type: ignore

        safe_metric(n_tracked_cars, "Cars", lambda: df_ntracked.iloc[0, 1])  # type: ignore
        safe_metric(n_tracked_peds, "Pedestrians", lambda: df_ntracked.iloc[1, 1])  # type: ignore
        safe_metric(n_departures, "Departures", lambda: df_ndep.iloc[0, 1])  # type: ignore

    
def veicoli_fuori(camera):
    try:
        response = requests.get(
            f"{API_BASE}/analytics/cameras/recent/outside_zones",
            params={"camera_id": camera["id"]},
        )
        response.raise_for_status()
        data = response.json()
        seen = set()
        unique = []

        for x in data:
            if x["event_type"] == "departure":
                seen.add(x)
            if x["tracker_id"] in seen:
                continue
            unique.append(x)
            seen.add(x["tracker_id"])

        s = "Found the following objects outside zones:"
        for x in unique:
            s += (
                f"  \n⋙ A **{x['class']}** at *x:{x['cx']} y:{x['cy']}* having tracking id *{x['tracker_id']}* at timestamp: *{x['time']}*"
            )

        st.markdown(s)

    except Exception as e:
        st.error(f"Could not get cars outside parking zones: {e}")
    pass


# logic to make API request and display the plot
def request_tracking_plots(camera):
    # if st.button("Show tracking map"):
    frame = get_first_frame(camera['uri'])
    payload = {"camera_id": camera['id']}
    if frame is not None:
        _, buf = cv2.imencode(".png", frame)
        payload["frame"] = base64.b64encode(buf).decode("utf-8")  # type: ignore
    r = requests.post(f"{API_BASE}/analytics/trajectory_analysis", json=payload)

    if r.status_code == 200:
        st.image(r.content, caption="Tracking Scatterplot", width="stretch")  # type: ignore
    else:
        st.error(f"Failed to Fetch Plot: {r.status_code}")


# get first frame of RTSP stream
def get_first_frame(camera_uri):
    # this should be the logic to obtain the first frame of a RTSP stream, to check if it actually works
    # if it doesn't, replace with a working logic and uhhhhhhh
    cap = cv2.VideoCapture(camera_uri)
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
        r = requests.get(
            f"{API_BASE}/analytics/metrics_report/kpi",
            params={"camera_id": camera_id, "t_start": t_i, "t_end": t_f},
        )

        if r.status_code == 200:
            r_s = r.content.decode()
            r_j = json.loads(r_s)

            st.dataframe(pd.DataFrame(r_j["total_tracked_by_class"]))
            st.dataframe(pd.DataFrame(r_j["avg_confidence_by_class"]))
            st.dataframe(pd.DataFrame(r_j["total_zones"]))
            st.dataframe(pd.DataFrame(r_j["max_occupations"]))
            st.dataframe(pd.DataFrame(r_j["avg_occupations"]))
            st.dataframe(pd.DataFrame(r_j["avg_track_time"]))
            st.dataframe(pd.DataFrame(r_j["n_departures"]))
            st.dataframe(pd.DataFrame(r_j["n_tracked_det"]))

        else:
            st.error(f"Could not fetch summary... {r.status_code}")


def request_timeseries(camera_id, t_i, t_f):
    # TODO: see if i can map the timestamps better, integers make no sense...

    if st.button("Get Timeseries Report"):
        # step 1: KPIs
        r = requests.get(
            f"{API_BASE}/analytics/metrics_report/timeseries",
            params={"camera_id": camera_id, "t_start": t_i, "t_end": t_f},
        )

        if r.status_code == 200:
            r_s = r.content.decode()
            r_j = json.loads(r_s)

            ts_1 = pd.DataFrame(r_j["ts_confidence"])
            ts_2 = pd.DataFrame(r_j["ts_objects"])
            ts_3 = pd.DataFrame(r_j["ts_parked"])

            ts_1['t'] = pd.to_datetime(ts_1['t'])
            ts_2['t'] = pd.to_datetime(ts_2['t'])
            ts_3['t'] = pd.to_datetime(ts_3['t'])
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 7))

            fig.suptitle("Timeseries Reports")

            axes[0].set_title("Average Confidence")
            axes[1].set_title("Number of tracked objects")
            axes[2].set_title("Number of parked vehicles")

            sns.lineplot(
                data=ts_1,
                ax=axes[0],
                x="t",
                y="avg_confidence",
                hue="class_name",
                palette="Set1",
            )

            sns.lineplot(
                data=ts_2,
                ax=axes[1],
                x="t",
                y="num_tracked",
                hue="class_name",
                palette="Set1",
            )

            sns.lineplot(
                data=ts_3,
                ax=axes[2],
                x="t",
                y="num_parked_vehicles",
                palette="Set1",
            )

            for ax in axes:
                ax.yaxis.grid()

            locator = mdates.AutoDateLocator(minticks=4, maxticks=16)
            formatter = mdates.ConciseDateFormatter(locator)

            axes[2].xaxis.set_major_locator(locator)
            axes[2].xaxis.set_major_formatter(formatter)

            axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[2].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            fig.autofmt_xdate(rotation=30, ha="right")
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
        
        emoji_mapping = {"not occupied": "| 🟢 | **Free**", "occupied": "| 🔴 | **Occupied**"}

        zone = {k: emoji_mapping.get(v,v) for k,v in zone.items()}

        st.table(zone)

    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")


def place_a_video(camera):
    uri = camera["uri"]
    cap = cv2.VideoCapture(uri)
    placeholder = st.empty()
    if not cap.isOpened():
        st.error(f"Could not open stream: {uri}")
        return (None, placeholder, [])

    zones = get_zones_to_draw(camera)
    return (cap, placeholder, zones)


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
            if x["event_type"] == "departure":
                seen.add(x['tracker_id'])
            if x["tracker_id"] in seen:
                continue
            unique.append(x)
            seen.add(x["tracker_id"])

        st.write(f"#### Number of parked cars seen by {camera['name']}: {len(unique)}")

    except Exception as e:
        st.error(f"Could not fetch the number of cars: {e}")
    pass


def camera_form():
    st.write("## Add Camera Source")
    st.markdown("In this part you will register your camera source in the database and followingly let the pipeline process run. The source can be either a path to a .mp4 file or a RTSP stream link.")
    if "camera_name" not in st.session_state:
        st.session_state.camera_name = None
    with st.form("camera-form"):
        name = st.text_input("Camera name")
        uri = st.text_input("Camera URI")
        
        submitted = st.form_submit_button("Register")
        if submitted:
            try:
                
                st.session_state.camera_name = name
                pippo = Process(target=run_zone_creator, args=(st.session_state.camera_name,uri), daemon=True)
                pippo.start()
                pippo.join()
                st.success("The camera is being registered")
                
            except Exception as e:
                st.error(f"Error: {e}")
    


def display_3d_viewer(camera_id, zones, single_camera):
    """Load and display the 3D viewer with zone data from database."""
    st.write("## 3D Parking Lot Viewer")
    st.markdown("In this part you will be able to interact with a 3D simulation of the parking lots")

    # Convert zones to format expected by viewer (polygon -> points)
    zones_data = []
    for zone in zones:
        zones_data.append(
            {
                "id": zone.get("id", -1),
                "points": zone.get("polygon_global_metric", []),
            }
        )

    # Load HTML template
    html_code = open("dashboard/viewer.html", "r").read()

    # Inject zones data and API base URL into HTML as JavaScript variables
    # Not the safe or elegant way, but it works for a demo. For prod, this is how you get random js injections
    zones_json = json.dumps(zones_data)
    inject_script = f"""
    <script>
    window.zonesDataFromPython = {zones_json};
    window.API_BASE = "{API_BASE}";
    window.SINGLE_CAMERA_MODE = {'true' if single_camera else 'false'};
    window.CAMERA_ID = {camera_id};
    </script>
    """
    html_code = html_code.replace("<body>", inject_script + "<body>")

    components.html(html_code, height=700)


def get_mapped_zones(camera_id=0, single_camera=False):
    url = f"{API_BASE}/mapped_zones/poly"
    if single_camera:
        url += f"?single_camera=true&camera_id={camera_id}"

    r = requests.get(url)

    if r.status_code == 200:
        zones = r.json()
        return zones
    else:
        st.error(f"Could not fetch mapped zones: {r.status_code}")
        return None


def merge():
    st.markdown("## Merge Parking lots")
    st.markdown("Press the button below to merge the designed parking lots so far")
    if st.button(label="Merge...", type="primary"):
        pippi = Process(target=merge_cameras, daemon=True)
        pippi.start()
        pippi.join()
        st.session_state.has_been_merged = True


def been_merged():
    try:
        pass
        r= requests.get(f"{API_BASE}/merged")
        return r.json()
    except:
        pass
    return False

def ping(e):
    try:
        requests.get(f"{API_BASE}/ping",params={"e":e})
    except:
        pass

def start_pipeline():
    try:
        r = requests.post(f"{API_BASE}/start/pipeline")
        r = r.json()
        st.write(f"{r["status"]} with {r["streams"]} streams")
        st.rerun()
    except Exception as e:
        ping(e)
        st.error("something has gone wrong starting the pipeline")
        return
    st.session_state.pipeline_started = True


def has_pipeline_started():
    
    try:
        r = requests.get(f"{API_BASE}/analytics/recent")
        r = r.json()
        ping(len(r) != 0)
        return len(r) != 0
    except Exception as e:
        ping(e)
        pass
    
    return False

def body():
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────────
    if "confirmed_camera" not in st.session_state:
        st.session_state.confirmed_camera = None
    if "show_zones" not in st.session_state:
        st.session_state.show_zones = False
    if "show_tracking_map" not in st.session_state:
        st.session_state.show_tracking_map = False
    if "has_been_merged" not in st.session_state:
        st.session_state.has_been_merged = been_merged()
    if "pipeline_started" not in st.session_state:
        st.session_state.pipeline_started = has_pipeline_started()

    # ── Sidebar — setup steps ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Ctrl+Park")
        st.markdown(
            "<p style='color:#484f58;font-size:0.8rem;margin-top:-0.5rem;margin-bottom:1rem'>"
            "Parking monitoring system</p>",
            unsafe_allow_html=True,
        )

        if not st.session_state.pipeline_started:
            st.markdown('<div class="sb-label">Step 1 — Register camera</div>', unsafe_allow_html=True)
            camera_form()

            st.markdown('<div class="sb-label">Step 2 — Merge parking lots</div>', unsafe_allow_html=True)
            merge()

            if st.session_state.has_been_merged:
                st.markdown('<div class="sb-label">Step 3 — Start pipeline</div>', unsafe_allow_html=True)
                if st.button(label="Start pipeline", type="primary"):
                    start_pipeline()
        else:
            st.success("Pipeline is running")

    # ── Main area ─────────────────────────────────────────────────────────────
    st.markdown("## Ctrl+Park Dashboard")
    st.markdown("---")
    camera_button()
    camera_selected()


body()
