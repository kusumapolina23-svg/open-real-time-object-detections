"""Streamlit dashboard components."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def dashboard_header() -> None:
    st.title("Real-Time Object Detection & Logging Platform")
    st.caption("Live camera detection powered by YOLO, with qualifying events stored in MySQL.")


def sidebar_controls(classes: list[str], db_ok: bool, db_message: str) -> tuple[bool, float, set[str]]:
    with st.sidebar:
        st.header("Controls")
        running = st.toggle("Camera", key="camera_running", help="Start or stop webcam capture")
        threshold_pct = st.slider("Confidence Threshold", 0, 100, 70, 1)
        st.subheader("Objects to Detect")
        col1, col2 = st.columns(2)
        if col1.button("Select All", use_container_width=True):
            st.session_state.selected_objects = classes
        if col2.button("Clear All", use_container_width=True):
            st.session_state.selected_objects = []
        chosen = st.multiselect(
            "Choose classes", classes, default=st.session_state.get("selected_objects", classes),
            key="selected_objects", label_visibility="collapsed",
        )
        st.subheader("Database")
        if db_ok:
            st.success(db_message, icon="✅")
        else:
            st.error(db_message, icon="⚠️")
        st.caption("Credentials are read from the local .env file.")
    return running, threshold_pct / 100.0, set(chosen)


def metrics(detected: int, logged: int, db_ok: bool) -> None:
    first, second, third = st.columns(3)
    first.metric("Objects in current frame", detected)
    second.metric("Events logged this session", logged)
    third.metric("Database", "Connected" if db_ok else "Unavailable")


def render_logs(rows: list[dict[str, Any]], error: str | None = None) -> None:
    st.subheader("Recent Logs")
    if error:
        st.warning(error)
        return
    if not rows:
        st.info("No detection events have been logged yet.")
        return
    frame = pd.DataFrame(rows).rename(columns={
        "timestamp": "Timestamp", "object_class": "Object", "confidence": "Confidence",
        "bbox_x": "X", "bbox_y": "Y", "bbox_w": "Width", "bbox_h": "Height",
    })
    frame["Confidence"] = (frame["Confidence"] * 100).map(lambda value: f"{value:.1f}%")
    st.dataframe(frame[["Timestamp", "Object", "Confidence", "X", "Y", "Width", "Height"]],
                 use_container_width=True, hide_index=True)
