# ctrl-park

### Install Dependencies
pip install -r requirements.txt

**N.B.** Make sure you have installed Tkinter on your Python installation!

### Install MySQL and create database
Install MySQL here: https://www.mysql.com/it/downloads/

mysql -u root -p -e "CREATE DATABASE ctrl_park;"

### Run the Backend API
**N.B.** Running the backend with the following command is also what applies database migrations

python -m uvicorn backend.main:app --reload

### Run the Dashboard

python -m streamlit run dashboard/app.py

### Test the Pipeline
python -m processing.run

### Draw new zones
python -m processing.draw_zones [option] --uri <\your uri>

    Only one option at a time may be selected, selecting more than one does not garantee the expected behaviour.

    -a --add        to add to the existing zones, specification of the uri is obligatory
    -r --remove     to remove one or more given zones from a given setup, specification of the uri is obligatory
    -v --visualize  to see the zones overlayed on the uri

### Manual Testing
Add Camera:
curl -X POST "http://localhost:8000/camera" -H "Content-Type: application/json" -d '{"name":"cam1","uri":"rtsp://example.com/stream"}'

Get Analytics: 
curl "http://localhost:8000/analytics/recent"

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
