# Real-Time Object Detection & Logging Platform

A modular webcam detection dashboard built with Python 3.12, OpenCV, Ultralytics YOLO, Streamlit, and MySQL. It annotates live frames, filters by selected COCO classes and confidence, and stores qualifying events in MySQL. Repeated detections at nearly the same location are suppressed for five seconds.

## Project structure

```text
Real-Time-Object-Detection/
├── main.py
├── detector.py
├── database.py
├── ui.py
├── config.py
├── database_schema.sql
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── models/
```

## Requirements

- Python 3.12
- A webcam accessible to OpenCV
- MySQL Server

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create the database and table using a MySQL account with permission to create databases:

   ```bash
   mysql -u root -p < database_schema.sql
   ```

3. Copy `.env.example` to `.env` and set your local MySQL host, username, password, and database. Never commit `.env`. `CAMERA_INDEX` defaults to `0`.

4. Launch the dashboard:

   ```bash
   streamlit run main.py
   ```

The first launch may download Ultralytics' small `yolov8n.pt` model. To use a local copy, put it at `models/yolov8n.pt`; model files are excluded from Git.

## Using the dashboard

- Use the Camera toggle to start or stop capture.
- Choose a minimum confidence (default 70%) and the classes to detect/log. Select All and Clear All are available.
- The live image shows the selected detections and confidence labels. Metrics show the current frame's qualifying detections and this session's successful inserts.
- Recent Logs displays the newest 50 database rows. Press Refresh logs to fetch them again.
- The sidebar shows MySQL connectivity. Missing tables, invalid credentials, and unavailable servers are reported with setup guidance.

Only selected objects meeting the configured confidence threshold are considered. Duplicate suppression compares same-class bounding boxes and only writes an overlapping detection again after its five-second cooldown. The cooldown is in-memory and resets when the app process restarts.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DB_HOST` | `localhost` | MySQL hostname |
| `DB_PORT` | `3306` | MySQL port |
| `DB_USER` | `root` | MySQL username |
| `DB_PASSWORD` | empty | MySQL password |
| `DB_NAME` | `vision_platform` | Database name |
| `CAMERA_INDEX` | `0` | OpenCV camera index |
| `YOLO_MODEL` | `models/yolov8n.pt` | Optional model path; fallback is `yolov8n.pt` |

## Troubleshooting

- **Camera unavailable:** check OS camera permissions, close other camera apps, and try a different `CAMERA_INDEX`.
- **MySQL credentials rejected:** update `.env` and verify the account can access `vision_platform`.
- **Database/table missing:** run `database_schema.sql` with a database administrator account.
- **YOLO model unavailable:** install requirements and check that the model is present or that Ultralytics can retrieve it.
- **Package import error:** activate the project virtual environment and run `pip install -r requirements.txt`.
