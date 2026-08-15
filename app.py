"""Contactless Vitals: a CPU-only webcam heart-rate (rPPG) demo.

MediaPipe locates a forehead region and the POS algorithm in ``rppg.py``
turns its subtle RGB fluctuations into a BPM estimate. The custom Gradio
dashboard uses FastRTC's video-only WebRTC component, so the browser does
not request microphone access. A hidden fixed-window state provides the
additional input required to initialize FastRTC's video handler.
"""

import logging
import time
import traceback

import cv2
import gradio as gr
import numpy as np
from fastrtc import (
    AdditionalOutputs,
    StreamHandler,
    WebRTC,
    get_cloudflare_turn_credentials_async,
)
from gradio.utils import get_space

logging.basicConfig(level=logging.INFO)

from roi import ForeheadROI, mean_rgb_in_box
from rppg import PulseEstimator

DEFAULT_WINDOW_SECONDS = 8.0
TOO_DARK_THRESHOLD = (
    25.0  # mean 0-255 pixel value below this: no algorithm can find a face
)
DIM_THRESHOLD = (
    70.0  # below this, boost brightness before detection (not for the pulse signal)
)


def _measurement_markup(
    value: str | None = None, status: str = "Waiting for camera"
) -> str:
    """Render the measurement in a stable layout independent of Markdown defaults."""
    reading = ""
    reading_class = "reading"
    if value is not None:
        reading_class += " has-value"
        reading = (
            f'<div class="reading-value">{value}</div>'
            '<div class="reading-unit">BPM</div>'
        )
    return (
        f'<div class="{reading_class}" aria-live="polite">'
        f'{reading}<p class="reading-status">{status}</p></div>'
    )


class VitalsHandler(StreamHandler):
    def __init__(self) -> None:
        super().__init__()
        # Created lazily on first frame rather than here: loading the
        # FaceLandmarker (TFLite + XNNPACK init) takes real wall-clock time,
        # and copy() runs synchronously inside fastrtc's connection setup.
        self.roi: ForeheadROI | None = None
        self.pulse = PulseEstimator(window_seconds=DEFAULT_WINDOW_SECONDS)
        self._window_seconds = DEFAULT_WINDOW_SECONDS
        self._last_ui_status: str | None = None
        self._last_ui_update = 0.0

    def copy(self) -> "VitalsHandler":
        return VitalsHandler()

    def shutdown(self) -> None:
        # Each connection gets its own MediaPipe FaceLandmarker -- without
        # releasing it here, every reconnect leaked one.
        if self.roi is not None:
            self.roi.close()

    def receive(self, frame) -> None:
        return None  # audio unused: modality="video" never opens an audio track

    def emit(self):
        return None  # audio unused

    def __call__(self, frame_bgr: np.ndarray, window_seconds: float):
        try:
            if window_seconds != self._window_seconds:
                self._window_seconds = window_seconds
                self.pulse = PulseEstimator(window_seconds=window_seconds)
            frame, status = self._process(frame_bgr)
            # Additional outputs travel through a separate Gradio event. Sending
            # one at camera frame rate can overwhelm that event and leave the
            # dashboard stuck on its initial text, so cap UI updates at 4 Hz.
            now = time.monotonic()
            status_changed = status != self._last_ui_status
            if (
                self._last_ui_status is None
                or (status_changed and now - self._last_ui_update >= 0.25)
                or now - self._last_ui_update >= 1.0
            ):
                self._last_ui_status = status
                self._last_ui_update = now
                return frame, AdditionalOutputs(status)
            return frame
        except Exception:  # noqa: BLE001
            # fastrtc's recv() loop swallows exceptions from bare callables
            # at DEBUG level; print unconditionally and keep the stream alive
            # while the Measurement card reports the error.
            print("VitalsHandler.__call__ crashed:")
            traceback.print_exc()
            out = frame_bgr.copy()
            return out, AdditionalOutputs(
                _measurement_markup(status="Error — see server terminal")
            )

    def _process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, str]:
        out = frame_bgr.copy()
        brightness = float(frame_bgr.mean())

        if brightness < TOO_DARK_THRESHOLD:
            return out, _measurement_markup(status="Too dark — add more light")

        if self.roi is None:
            self.roi = ForeheadROI()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Detect on a brightened copy if dim, but always sample the pulse
        # color from the *original* frame -- boosting would distort the
        # subtle RGB ratios the POS algorithm depends on.
        detect_rgb = (
            _gamma_boost(frame_rgb) if brightness < DIM_THRESHOLD else frame_rgb
        )
        box = self.roi.detect_box(detect_rgb)

        if box is None:
            return out, _measurement_markup(status="Looking for a face…")

        mean_rgb = mean_rgb_in_box(frame_rgb, box)
        self.pulse.add_sample(time.monotonic(), mean_rgb)
        x1, y1, x2, y2 = box
        # Accent-blue guide, matching the interface without obscuring the feed.
        cv2.rectangle(out, (x1, y1), (x2, y2), (235, 99, 37), 2)

        bpm = self.pulse.estimate()
        if bpm is None:
            elapsed, target = self.pulse.measurement_progress()
            status = _measurement_markup(
                status=f"Measuring… {elapsed:.0f}/{target:.0f} seconds"
            )
        else:
            status = _measurement_markup(f"{bpm:.0f}", "Live reading")
        return out, status


_GAMMA_LUT = ((np.arange(256) / 255.0) ** (1.0 / 2.0) * 255).astype(np.uint8)


def _gamma_boost(frame_rgb: np.ndarray) -> np.ndarray:
    """Brightens shadows for face *detection* only (gamma=2.0)."""
    return cv2.LUT(frame_rgb, _GAMMA_LUT)


def _update_measurement_status(_current: str, status: str) -> str:
    """Forward FastRTC's latest status to the dashboard reading card."""
    return status


def _measurement_started():
    """Show immediate feedback while the first processed frame arrives."""
    return (
        _measurement_markup(status="Hold still, measuring…"),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def _measurement_stopped():
    """Reset the reading panel when the user stops the camera stream."""
    return (
        _measurement_markup(),
        gr.update(visible=True),
        gr.update(visible=False),
    )


START_MEASUREMENT_JS = """
() => {
    const root = document.querySelector('#vitals-video');
    if (!root) return;

    const start = () => {
        const button = root.querySelector('.button-wrap button');
        if (button) {
            button.click();
            return true;
        }
        return false;
    };

    if (start()) return;
    const permissionButton = root.querySelector(
        '[title="grant webcam access"] button'
    );
    if (!permissionButton) return;
    permissionButton.click();

    let attempts = 0;
    const waitForCamera = window.setInterval(() => {
        attempts += 1;
        if (start() || attempts >= 300) window.clearInterval(waitForCamera);
    }, 100);
}
"""


STOP_MEASUREMENT_JS = """
() => {
    const root = document.querySelector('#vitals-video');
    const button = root?.querySelector('.button-wrap button');
    if (button) button.click();

    // FastRTC reopens a local preview after its native stop action. Fully
    // release those tracks and remount the component so it returns to the
    // initial Enable camera screen instead.
    window.setTimeout(() => {
        root?.querySelectorAll('video').forEach((video) => {
            const stream = video.srcObject;
            if (stream instanceof MediaStream) {
                stream.getTracks().forEach((track) => track.stop());
                video.srcObject = null;
            }
        });
        window.location.reload();
    }, 350);
}
"""


APP_CSS = """
:root {
    --app-bg: #F8FAFC;
    --surface: #FFFFFF;
    --surface-soft: #F1F5F9;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #64748B;
    --border: #E2E8F0;
    --border-strong: #CBD5E1;
    --accent: #2563EB;
    --accent-hover: #1D4ED8;
    --accent-soft: #EFF6FF;
    --video-bg: #111827;
}

html, body, body.dark { background: var(--app-bg) !important; }

.gradio-container {
    max-width: 1160px !important;
    margin: 0 auto !important;
    padding: 28px 24px 40px !important;
    color: var(--text-primary) !important;
    background: var(--app-bg) !important;
    color-scheme: light;
    --body-background-fill: var(--app-bg);
    --body-text-color: var(--text-primary);
    --block-background-fill: var(--surface);
    --block-border-color: var(--border);
    --block-label-text-color: var(--text-primary);
    --button-secondary-background-fill: var(--surface);
    --button-secondary-text-color: var(--text-primary);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container .block { box-shadow: none !important; }

/* ---------- header ---------- */
.app-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}
.app-header h1 {
    margin: 0;
    color: var(--text-primary);
    font-size: 27px;
    font-weight: 600;
    letter-spacing: -0.015em;
}
.app-subtitle { margin: 4px 0 0; color: var(--text-secondary); font-size: 13px; font-weight: 500; }
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: 0 0 auto;
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); }
.app-description { margin: 12px 0 20px; color: var(--text-secondary); font-size: 14px; line-height: 1.5; }

/* ---------- panels ---------- */
.dashboard-row { gap: 16px !important; align-items: flex-start !important; }
.camera-column, .side-column {
    align-self: flex-start !important;
    gap: 0 !important;
}
.panel {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    background: var(--surface) !important;
    box-shadow: none !important;
}
.panel > .styler {
    width: 100% !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.camera-panel { padding: 16px !important; }
.camera-panel .block { overflow-x: hidden !important; }
.panel-heading { padding: 0 0 12px !important; }
.panel-heading h2 { margin: 0; color: var(--text-primary); font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
.panel-heading p { margin: 3px 0 0; color: var(--text-muted); font-size: 13px; }

/* ---------- camera ---------- */
#vitals-video {
    position: relative !important;
    overflow: hidden !important;
    min-height: 410px !important;
    aspect-ratio: 16 / 10 !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
    background: var(--video-bg) !important;
}
#vitals-video .video-container {
    width: 100% !important;
    height: 100% !important;
    min-height: 410px !important;
    background: var(--video-bg) !important;
}
#vitals-video .video-container > .wrap {
    position: relative !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 410px !important;
    background: var(--video-bg) !important;
}
#vitals-video video,
#vitals-video .full-screen video {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    object-fit: contain !important;
    object-position: center center !important;
    background: var(--video-bg) !important;
}
#vitals-video [title="grant webcam access"] {
    position: absolute !important;
    inset: 0 !important;
    z-index: 3 !important;
    height: 100% !important;
    background-color: var(--video-bg) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='36' height='28' viewBox='0 0 36 28' fill='none'%3E%3Crect x='1' y='1' width='26' height='26' rx='4' stroke='%2398A2B3' stroke-width='2'/%3E%3Cpath d='M27 9.5 35 5v18l-8-4.5v-9Z' stroke='%2398A2B3' stroke-width='2' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center calc(50% - 66px) !important;
    pointer-events: auto !important;
}
#vitals-video [title="grant webcam access"]::after {
    position: absolute;
    left: 24px;
    right: 24px;
    z-index: 2;
    text-align: center;
    pointer-events: none;
}
#vitals-video [title="grant webcam access"]::after {
    content: "Camera access is required to estimate your heart rate.";
    top: calc(50% + 34px);
    color: #94A3B8;
    font-size: 13px;
    font-weight: 400;
}
#vitals-video [title="grant webcam access"] > button {
    position: absolute !important;
    top: calc(50% - 18px) !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 4 !important;
    width: auto !important;
    height: 38px !important;
    min-height: 38px !important;
    padding: 0 15px !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    color: #f2f4f7 !important;
    background: var(--accent) !important;
    box-shadow: none !important;
    cursor: pointer !important;
}
#vitals-video [title="grant webcam access"] > button:hover {
    border-color: var(--accent-hover) !important;
    background: var(--accent-hover) !important;
}
#vitals-video [title="grant webcam access"] > button .wrap {
    display: flex !important;
    min-height: 0 !important;
    height: auto !important;
    padding: 0 !important;
    color: transparent !important;
    background: transparent !important;
    font-size: 0 !important;
}
#vitals-video [title="grant webcam access"] > button .wrap::after {
    content: "Enable camera";
    color: #f2f4f7;
    font-size: 13px;
    font-weight: 600;
}
#vitals-video [title="grant webcam access"] > button .icon-wrap { display: none !important; }
#vitals-video .button-wrap {
    visibility: hidden !important;
    pointer-events: none !important;
}

.camera-hints {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    box-sizing: border-box;
    width: 100%;
    max-width: 100%;
    margin: 12px 0 0;
    padding: 12px 14px;
    border-top: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface-soft);
}
.camera-hints .hint {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-secondary);
    font-size: 13px;
}
.camera-hints .hint svg { flex: 0 0 auto; stroke: var(--accent); }

/* ---------- measurement ---------- */
.status-panel.panel {
    position: relative !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    justify-content: flex-start !important;
    gap: 0 !important;
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    padding: 22px !important;
    overflow: hidden !important;
}
.status-panel > .styler,
.status-panel > .styler > .block,
.status-panel .form {
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.status-panel > .styler {
    width: 100% !important;
    padding: 0 !important;
    gap: 0 !important;
    overflow: visible !important;
}
.status-panel > .styler > .block {
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.status-panel .panel-heading {
    padding: 0 0 16px !important;
    border: 0 !important;
}
#bpm-display {
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 0 16px !important;
    overflow: visible !important;
    border: 0 !important;
    background: transparent !important;
}
#bpm-display > div,
#bpm-display .prose {
    margin: 0 !important;
    padding: 0 !important;
}
.reading {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;
    color: var(--text-primary);
}
.reading-value {
    min-height: 0;
    margin: 0 0 4px;
    color: var(--text-primary);
    font-size: 60px;
    font-weight: 600;
    letter-spacing: -0.035em;
    line-height: 1;
}
.reading-unit {
    margin: 0;
    color: var(--text-muted);
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.reading-status {
    margin: 14px 0 0 !important;
    color: var(--text-secondary) !important;
    font-size: 15px;
    line-height: 1.45;
}
.reading:not(.has-value) .reading-status { margin-top: 0 !important; }

.measurement-actions {
    width: 100% !important;
    gap: 8px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
#start-measurement, #stop-measurement {
    position: relative !important;
    z-index: 5 !important;
    flex: 1 1 0 !important;
    width: 100% !important;
    margin: 0 !important;
    pointer-events: auto !important;
}
#start-measurement:not(button), #stop-measurement:not(button) {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}
.measurement-actions button {
    width: 100% !important;
    min-height: 44px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
button#start-measurement,
#start-measurement button {
    border: 1px solid var(--accent) !important;
    color: #ffffff !important;
    background: var(--accent) !important;
}
button#start-measurement:hover,
#start-measurement button:hover {
    border-color: var(--accent-hover) !important;
    background: var(--accent-hover) !important;
}
button#stop-measurement,
#stop-measurement button {
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    background: var(--surface) !important;
}
button#stop-measurement:hover,
#stop-measurement button:hover {
    border-color: var(--accent-hover) !important;
    color: var(--accent-hover) !important;
    background: var(--accent-soft) !important;
}
.measurement-actions button:focus-visible,
#vitals-video [title="grant webcam access"] > button:focus-visible {
    outline: 3px solid var(--accent-soft) !important;
    outline-offset: 2px !important;
}

.privacy-note {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    margin: 14px 0 0;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    color: var(--text-muted) !important;
    font-size: 12px;
    line-height: 1.5;
}
.privacy-note svg { flex: 0 0 auto; margin-top: 1px; stroke: var(--accent); }

/* ---------- instructions ---------- */
.instructions { margin-top: 28px; padding-top: 22px; border-top: 1px solid var(--border); }
.instructions h2 { margin: 0 0 16px; color: var(--text-primary); font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }
.steps-row { display: flex; }
.step { flex: 1; padding: 0 20px; border-left: 1px solid var(--border); }
.step:first-child { padding-left: 0; border-left: none; }
.step-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.step-num { color: var(--accent); font-size: 13px; font-weight: 600; }
.step h3 { margin: 0; color: var(--text-primary); font-size: 14px; font-weight: 600; }
.step p { margin: 0; color: var(--text-muted); font-size: 13px; line-height: 1.5; }

/* ---------- footer ---------- */
.app-footer {
    display: flex;
    gap: 48px;
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
}
.app-footer div { flex: 1; }
.app-footer h4 { margin: 0 0 4px; color: var(--text-secondary) !important; font-size: 12px; font-weight: 600; }
.app-footer p { margin: 0; color: var(--text-muted) !important; font-size: 12px; line-height: 1.5; }

@media (max-width: 900px) {
    #vitals-video,
    #vitals-video .video-container,
    #vitals-video .video-container > .wrap { min-height: 340px !important; }
}
@media (max-width: 760px) {
    .gradio-container { padding: 20px 16px 32px !important; }
    .app-header { flex-direction: column; align-items: flex-start; gap: 6px; }
    .dashboard-row > .camera-column,
    .dashboard-row > .side-column { min-width: 0 !important; }
    .steps-row { flex-direction: column; gap: 16px; }
    .step { padding: 0 0 0 12px; border-left: 2px solid var(--border); }
    .step:first-child { padding-left: 12px; border-left: 2px solid var(--border); }
    .app-footer { flex-direction: column; gap: 16px; }
    #vitals-video,
    #vitals-video .video-container,
    #vitals-video .video-container > .wrap { min-height: 260px !important; }
}
"""


def _build_ui() -> tuple[gr.Blocks, WebRTC]:
    """Build the custom dashboard while keeping FastRTC's stream behavior."""
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="gray",
    )

    with gr.Blocks(css=APP_CSS, theme=theme, title="Contactless Vitals") as demo:
        gr.HTML(
            """
            <header class="app-header">
                <div>
                    <h1>Contactless Vitals</h1>
                    <p class="app-subtitle">Camera-based heart rate estimation</p>
                </div>
                <div class="status-indicator"><span class="status-dot"></span>Local processing</div>
            </header>
            <p class="app-description">Estimate your pulse from subtle color changes in the forehead region.</p>
            """
        )

        with gr.Row(elem_classes=["dashboard-row"], equal_height=False):
            with (
                gr.Column(scale=7, min_width=460, elem_classes=["camera-column"]),
                gr.Group(elem_classes=["panel", "camera-panel"]),
            ):
                gr.HTML(
                    """
                        <div class="panel-heading">
                            <h2>Camera</h2>
                            <p>Position your face within the frame.</p>
                        </div>
                        """
                )
                camera = WebRTC(
                    label="Live heart-rate camera",
                    show_label=False,
                    container=False,
                    modality="video",
                    mode="send-receive",
                    elem_id="vitals-video",
                    full_screen=False,
                    button_labels={
                        "start": "Start measurement",
                        "stop": "Stop measurement",
                        "waiting": "Starting…",
                    },
                    rtc_configuration=(
                        get_cloudflare_turn_credentials_async if get_space() else None
                    ),
                    icon_button_color="#2563eb",
                    pulse_color="#2563eb",
                )
                gr.HTML(
                    """
                        <div class="camera-hints" aria-label="Measurement tips">
                            <span class="hint">
                                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="5" stroke-width="1.4"/></svg>
                                Good lighting
                            </span>
                            <span class="hint">
                                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="5" stroke-width="1.4"/></svg>
                                Forehead visible
                            </span>
                            <span class="hint">
                                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="5" stroke-width="1.4"/></svg>
                                Hold still
                            </span>
                        </div>
                        """
                )

            with gr.Column(scale=3, min_width=280, elem_classes=["side-column"]):
                with gr.Group(elem_classes=["panel", "status-panel"]):
                    gr.HTML('<div class="panel-heading"><h2>Heart rate</h2></div>')
                    bpm_display = gr.HTML(
                        _measurement_markup(),
                        elem_id="bpm-display",
                        container=False,
                        padding=False,
                    )
                    with gr.Row(elem_classes=["measurement-actions"]):
                        start_button = gr.Button(
                            "Start measurement",
                            variant="primary",
                            elem_id="start-measurement",
                        )
                        stop_button = gr.Button(
                            "Stop",
                            variant="secondary",
                            visible=False,
                            elem_id="stop-measurement",
                        )
                    gr.HTML(
                        """
                        <p class="privacy-note">
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                                <rect x="2.5" y="5.2" width="7" height="5.3" rx="1" stroke-width="1.1"/>
                                <path d="M4 5.2V3.6a2 2 0 0 1 4 0v1.6" stroke-width="1.1"/>
                            </svg>
                            Video is processed locally and is not recorded.
                        </p>
                        """
                    )

                # A real additional input is required for FastRTC's video
                # handshake. Keep the fixed 8-second window as hidden state.
                window_seconds_state = gr.State(DEFAULT_WINDOW_SECONDS)

        gr.HTML(
            """
            <section class="instructions" aria-labelledby="instructions-title">
                <h2 id="instructions-title">How to get a reliable reading</h2>
                <div class="steps-row">
                    <article class="step">
                        <div class="step-head"><span class="step-num">1</span><h3>Use even lighting</h3></div>
                        <p>Face a window or soft light.</p>
                    </article>
                    <article class="step">
                        <div class="step-head"><span class="step-num">2</span><h3>Start the camera</h3></div>
                        <p>Allow camera access when prompted.</p>
                    </article>
                    <article class="step">
                        <div class="step-head"><span class="step-num">3</span><h3>Keep your forehead visible</h3></div>
                        <p>Center your face in the frame.</p>
                    </article>
                    <article class="step">
                        <div class="step-head"><span class="step-num">4</span><h3>Hold still</h3></div>
                        <p>Remain still for several seconds while the signal stabilizes.</p>
                    </article>
                </div>
            </section>
            <footer class="app-footer">
                <div>
                    <h4>How it works</h4>
                    <p>MediaPipe tracks the forehead region while the POS algorithm analyzes pulse-related RGB changes.</p>
                </div>
                <div>
                    <h4>Research demo</h4>
                    <p>Not a medical device. Not intended for diagnosis, treatment, or emergency monitoring.</p>
                </div>
            </footer>
            """
        )

        camera.stream(
            fn=VitalsHandler(),
            # FastRTC's bare video handler needs an initial input payload
            # before it starts processing.
            inputs=[camera, window_seconds_state],
            outputs=[camera],
            time_limit=60 if get_space() else None,
            concurrency_limit=3 if get_space() else None,
        )
        camera.on_additional_outputs(
            _update_measurement_status,
            inputs=[bpm_display],
            outputs=[bpm_display],
            concurrency_limit=3 if get_space() else "default",
            show_progress="hidden",
        )
        start_button.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=START_MEASUREMENT_JS,
            queue=False,
        )
        stop_button.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=STOP_MEASUREMENT_JS,
            queue=False,
        )
        camera.start_recording(
            fn=_measurement_started,
            outputs=[bpm_display, start_button, stop_button],
            queue=False,
            show_progress="hidden",
        )
        camera.stop_recording(
            fn=_measurement_stopped,
            outputs=[bpm_display, start_button, stop_button],
            queue=False,
            show_progress="hidden",
        )

    # on_additional_outputs is a long-running generator; enabling Gradio's
    # queue keeps status and BPM updates streaming for the whole measurement.
    demo.queue()
    return demo, camera


demo, stream = _build_ui()

if __name__ == "__main__":
    demo.launch()
