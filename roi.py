"""Forehead ROI localization using MediaPipe's Tasks API (FaceLandmarker).

Newer mediapipe wheels dropped the old `mp.solutions.face_mesh` API in
favor of `mediapipe.tasks` -- this uses the current, supported one. We
track a skin patch that moves rigidly with the head (forehead, between the
eyebrows and the hairline) and avoid eyes/mouth, which introduce motion
that isn't related to blood volume pulse.
"""

import time
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class ForeheadROI:
    """Wraps a per-instance FaceLandmarker (not shared across connections)."""

    def __init__(self) -> None:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(_ensure_model()))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def detect_box(self, frame_rgb: np.ndarray):
        """Returns (x1, y1, x2, y2) for the forehead patch, or None if no face is found.

        `frame_rgb` is used only for landmark detection -- pass a brightened
        copy here if the raw feed is dim, since detection tolerates that but
        the actual pulse *signal* (see `extract`) should always be sampled
        from the unmodified frame to avoid distorting the subtle color
        changes rPPG depends on.
        """
        h, w, _ = frame_rgb.shape

        ts_ms = int(time.time() * 1000)
        self._last_ts_ms = max(
            self._last_ts_ms + 1, ts_ms
        )  # must be strictly increasing

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, self._last_ts_ms)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        xs = np.array([p.x for p in landmarks])
        ys = np.array([p.y for p in landmarks])
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        face_w, face_h = x_max - x_min, y_max - y_min

        # Forehead band: centered horizontally, just below the hairline / above the brows.
        fx1 = x_min + 0.32 * face_w
        fx2 = x_min + 0.68 * face_w
        fy1 = y_min + 0.06 * face_h
        fy2 = y_min + 0.22 * face_h

        x1, x2 = int(fx1 * w), int(fx2 * w)
        y1, y2 = int(fy1 * h), int(fy2 * h)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def extract(self, frame_rgb: np.ndarray):
        """Returns (mean_rgb, box) for the forehead patch, sampled from `frame_rgb`."""
        box = self.detect_box(frame_rgb)
        if box is None:
            return None, None
        return mean_rgb_in_box(frame_rgb, box), box


def mean_rgb_in_box(
    frame_rgb: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = frame_rgb[y1:y2, x1:x2]
    return patch.reshape(-1, 3).mean(axis=0)
