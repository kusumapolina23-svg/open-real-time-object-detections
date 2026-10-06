"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
CONFIG_ERRORS: list[str] = []


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError:
        CONFIG_ERRORS.append(f"{key} must be an integer; received {raw!r}.")
        return default


@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_user: str = os.getenv("DB_USER", "root")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_name: str = os.getenv("DB_NAME", "vision_platform")
    db_port: int = _env_int("DB_PORT", 3306)
    model_path: Path = Path(os.getenv("YOLO_MODEL", str(PROJECT_DIR / "models" / "yolov8n.pt")))
    camera_index: int = _env_int("CAMERA_INDEX", 0)
    default_confidence: float = 0.70
    log_cooldown_seconds: float = 5.0
    log_iou_threshold: float = 0.30

    def validate(self) -> None:
        if CONFIG_ERRORS:
            raise ValueError(" ".join(CONFIG_ERRORS))
        if not self.db_host.strip() or not self.db_user.strip() or not self.db_name.strip():
            raise ValueError("DB_HOST, DB_USER, and DB_NAME must not be empty.")
        if not 1 <= self.db_port <= 65535:
            raise ValueError("DB_PORT must be between 1 and 65535.")
        if self.camera_index < 0:
            raise ValueError("CAMERA_INDEX must be zero or greater.")


settings = Settings()
