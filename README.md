# ctrl-park

### Install Dependencies
pip install -r requirements.txt



### Install MySQL and create database
Install MySQL here: https://www.mysql.com/it/downloads/

### Run the Backend API
**N.B.** Running the backend with the following command is also what applies database migrations

python -m uvicorn backend.main:app --reload

### Run the Dashboard
python -m streamlit run dashboard/app.py

### Test the Pipeline
python -m processing.run

### Manual Testing
Add Camera:
curl -X POST "http://localhost:8000/camera" -H "Content-Type: application/json" -d '{"name":"cam1","uri":"rtsp://example.com/stream"}'

Get Analytics: 
curl "http://localhost:8000/analytics/recent"

To video feed on the dashboard (probably only works on linux for now):


    0. have docker

    1. setup the docker:
        docker run --rm -it -v $PWD/rtsp-simple-server.yml:/rtsp-simple-server.yml -p 8554:8554 aler9/rtsp-simple-server:v1.3.0

Or, equivalently on Windows:
        docker run --rm -it -v %CD%/rtsp-simple-server.yml:/rtsp-simple-server.yml -p 8554:8554 aler9/rtsp-simple-server:v1.3.0

    2. setup ffmpeg that feeds the video (video/testfile.mp4 in this case) to the docker:
        ffmpeg -re -stream_loop -1 -i video/testfile.mp4 -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live.stream

    3. profit


### .env template:
DB_USER=root
DB_PASSWORD=root
DB_HOST=localhost
DB_PORT=3306
DB_NAME=ctrl_park
API_BASE_URL=http://localhost:8000
CAMERA_URI=video/testfile.mp4
USE_LOGGING=True
QUEUE_SIZE_LOG_PATH=queue_size.log
MAX_QUEUE_SIZE=200
QUEUE_RESTART_THRESHOLD=100
SLEEP_TIME=1
TARGET_FPS=3
LOST_TRACK_BUFFER=30
MIN_DETECTION_SAVE_INTERVAL=1.0
BBOX_MOVEMENT_THRESHOLD=10
OCCUPANCY_TRANSITION_BUFFER_MINUTES=2
OCCUPANT_ABSENCE_THRESHOLD_SECONDS=120
OLD_DETECTION_CUTOFF_SECONDS=10
