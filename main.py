"""Application entry point: streamlit run main.py"""

from __future__ import annotations

import logging
from contextlib import suppress

import cv2
import streamlit as st

from config import CONFIG_ERRORS, settings
from database import DetectionDatabase
from detector import DuplicateSuppressor, detect_frame, load_model
from ui import dashboard_header, metrics, render_logs, sidebar_controls

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Real-Time Object Detection", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Loading lightweight YOLO model…")
def get_model():
    return load_model(settings.model_path)


@st.cache_resource
def get_database() -> DetectionDatabase:
    return DetectionDatabase()


@st.cache_resource
def get_suppressor() -> DuplicateSuppressor:
    return DuplicateSuppressor(settings.log_cooldown_seconds, settings.log_iou_threshold)


def release_camera() -> None:
    capture = st.session_state.pop("capture", None)
    if capture is not None:
        with suppress(Exception):
            capture.release()


dashboard_header()
if CONFIG_ERRORS:
    st.error("Invalid configuration: " + " ".join(CONFIG_ERRORS))
db = get_database()
db_ok, db_message = db.health_check()

try:
    model = get_model()
    model_classes = list(model.names.values())
except Exception as exc:
    model = None
    model_classes = ["person", "cell phone", "laptop", "bottle", "car"]
    st.error(f"YOLO model unavailable: {exc}")

if "selected_objects" not in st.session_state:
    common = {"person", "cell phone", "laptop", "bottle", "car"}
    st.session_state.selected_objects = [name for name in model_classes if name in common]
running, confidence, selected = sidebar_controls(model_classes, db_ok, db_message)

if "logged_count" not in st.session_state:
    st.session_state.logged_count = 0
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []

if running and model is not None:
    if "capture" not in st.session_state or not st.session_state.capture.isOpened():
        capture = cv2.VideoCapture(settings.camera_index)
        if not capture.isOpened():
            capture.release()
            st.error("Webcam unavailable. Check that it is connected and not in use by another app.")
            st.session_state.camera_running = False
        else:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            st.session_state.capture = capture
else:
    release_camera()

if running and model is not None and st.session_state.get("camera_running", False):
    @st.fragment(run_every=0.08)
    def live_view() -> None:
        capture = st.session_state.get("capture")
        if capture is None or not capture.isOpened():
            st.error("Camera stopped. Toggle Camera off and on to reconnect.")
            return
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            st.warning("Camera returned an empty frame. Check the webcam connection.")
            return
        try:
            annotated, detections = detect_frame(model, frame, selected, confidence)
            st.session_state.last_detections = detections
            if db_ok:
                for detection in detections:
                    if get_suppressor().should_log(detection):
                        try:
                            db.insert_detection(detection)
                            st.session_state.logged_count += 1
                        except Exception as exc:
                            st.session_state.db_runtime_error = db._friendly_error(exc)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        except Exception as exc:
            st.error(f"Frame processing failed: {exc}")

    live_view()
elif running and model is None:
    st.info("Install the packages and make the YOLO model available to start detection.")
else:
    st.info("Turn on Camera in the sidebar to begin live detection.")

runtime_error = st.session_state.get("db_runtime_error")
if runtime_error:
    st.warning(f"Database logging paused for failed writes: {runtime_error}")
metrics(len(st.session_state.last_detections), st.session_state.logged_count, db_ok)
left, right = st.columns([3, 1])
with right:
    if st.button("Refresh logs", use_container_width=True):
        st.rerun()
try:
    render_logs(db.recent_logs(50), runtime_error)
except Exception as exc:
    render_logs([], db._friendly_error(exc))
