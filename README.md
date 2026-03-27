# ctrl-park

### Install Dependencies
pip install -r requirements.txt

### Install MySQL and create database
Install MySQL here: https://www.mysql.com/it/downloads/

mysql -u root -p -e "CREATE DATABASE ctrl_park;"

### Run the Backend API
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

### .env template:
DB_USER=root
DB_PASSWORD=root
DB_HOST=localhost
DB_PORT=3306
DB_NAME=ctrl_park
API_BASE_URL=http://localhost:8000
CAMERA_URI=video/testfile.mp4


