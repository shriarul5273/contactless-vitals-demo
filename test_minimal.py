"""Minimal fastrtc smoke test: no mediapipe, no rPPG, just a live video flip.

Run this standalone (`python test_minimal.py`) to check whether WebRTC video
streaming works at all in this browser/environment, independent of anything
in app.py. If clicking Record here does nothing either, the problem is
environmental (browser WebRTC support, firewall/VPN blocking UDP, aiortc/av
install, etc.), not a bug in the rPPG app's code.
"""

import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler, Stream


class FlipHandler(AsyncAudioVideoStreamHandler):
    def __init__(self) -> None:
        super().__init__("mono", output_sample_rate=24000, input_sample_rate=48000)
        self._last_frame: np.ndarray | None = None

    def copy(self) -> "FlipHandler":
        return FlipHandler()

    async def receive(self, frame) -> None:
        return None

    async def emit(self):
        return None

    async def video_receive(self, frame: np.ndarray) -> None:
        self._last_frame = np.flip(
            frame, axis=0
        )  # upside down = visible proof it's live

    async def video_emit(self):
        if self._last_frame is not None:
            return self._last_frame
        return np.zeros((480, 640, 3), dtype=np.uint8)


stream = Stream(
    handler=FlipHandler(),
    modality="audio-video",
    mode="send-receive",
    ui_args={"title": "Minimal WebRTC Test", "full_screen": False},
)

if __name__ == "__main__":
    stream.ui.launch()
