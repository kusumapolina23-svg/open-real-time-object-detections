"""YOLO inference, frame annotation, and duplicate event suppression."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from config import Settings, settings


def load_model(model_path: Path | str = settings.model_path) -> YOLO:
    """Load a local model, falling back to Ultralytics' downloadable yolov8n.pt."""
    path = Path(model_path)
    try:
        return YOLO(str(path if path.exists() else "yolov8n.pt"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load YOLO model at {path}. Ensure ultralytics is installed and "
            "the model file exists or can be downloaded."
        ) from exc


def detect_frame(
    model: YOLO, frame: np.ndarray, selected_classes: set[str], confidence: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return an annotated BGR frame and qualifying detections."""
    results = model.predict(frame, conf=confidence, verbose=False)
    detections: list[dict[str, Any]] = []
    annotated = frame.copy()
    if not results or results[0].boxes is None:
        return annotated, detections

    result = results[0]
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(names[class_id])
        score = float(box.conf.item())
        if class_name not in selected_classes or score < confidence:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        item = {
            "class_name": class_name, "confidence": score, "x": x1, "y": y1,
            "width": max(0, x2 - x1), "height": max(0, y2 - y1),
        }
        detections.append(item)
        color = (42, 190, 120)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(0, y1 - th - 10)
        cv2.rectangle(annotated, (x1, top), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated, detections


class DuplicateSuppressor:
    """Suppress repeat inserts for the same class and overlapping location."""

    def __init__(self, cooldown_seconds: float = settings.log_cooldown_seconds,
                 iou_threshold: float = settings.log_iou_threshold) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.iou_threshold = iou_threshold
        self.last_logged: dict[str, list[tuple[float, tuple[int, int, int, int]]]] = {}

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1, ix2, iy2 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    def should_log(self, detection: dict[str, Any], now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        name = detection["class_name"]
        rect = (detection["x"], detection["y"], detection["width"], detection["height"])
        recent = [(stamp, old) for stamp, old in self.last_logged.get(name, [])
                  if now - stamp < self.cooldown_seconds]
        self.last_logged[name] = recent
        if any(self._iou(rect, old) >= self.iou_threshold for _, old in recent):
            return False
        recent.append((now, rect))
        self.last_logged[name] = recent
        return True
