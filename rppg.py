"""Remote photoplethysmography (rPPG): estimate heart rate from a rolling
buffer of mean forehead-skin RGB samples, using the POS algorithm.

Reference: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
"Algorithmic Principles of Remote PPG." IEEE Transactions on Biomedical
Engineering, 64(7), 1479-1491.

This is deliberately classical signal processing rather than a trained
model: POS separates the pulse signal from motion/lighting artifacts by
projecting temporally-normalized RGB onto a plane orthogonal to the
expected skin-tone vector, no GPU or training data required.
"""

from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt

LOW_HZ = 0.7  # 42 BPM
HIGH_HZ = 4.0  # 240 BPM
RESAMPLE_HZ = 20.0


class PulseEstimator:
    """Buffers timestamped RGB samples and estimates BPM with the POS algorithm."""

    def __init__(self, window_seconds: float = 8.0) -> None:
        self.window_seconds = window_seconds
        self._min_samples = int(window_seconds * RESAMPLE_HZ * 0.5)
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()
        self.bpm: float | None = None
        self._bpm_smoothed: float | None = None

    def add_sample(self, t: float, mean_rgb: np.ndarray) -> None:
        self._t.append(t)
        self._rgb.append(mean_rgb)
        cutoff = t - self.window_seconds
        while self._t and self._t[0] < cutoff:
            self._t.popleft()
            self._rgb.popleft()

    def ready(self) -> bool:
        return (
            len(self._t) >= 2
            and (self._t[-1] - self._t[0]) >= self.window_seconds * 0.75
        )

    def measurement_progress(self) -> tuple[float, float]:
        """Return collected and required signal durations in seconds."""
        target = self.window_seconds * 0.75
        elapsed = 0.0 if len(self._t) < 2 else self._t[-1] - self._t[0]
        return min(elapsed, target), target

    def estimate(self) -> float | None:
        if not self.ready():
            return None

        t = np.array(self._t)
        rgb = np.array(self._rgb)  # (N, 3), R, G, B order

        # Webcam frame arrival isn't perfectly periodic; resample onto a
        # uniform grid before filtering.
        uniform_t = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
        if len(uniform_t) < self._min_samples:
            return None
        r = np.interp(uniform_t, t, rgb[:, 0])
        g = np.interp(uniform_t, t, rgb[:, 1])
        b = np.interp(uniform_t, t, rgb[:, 2])

        pulse = _pos_algorithm(r, g, b)
        pulse = _bandpass(pulse, RESAMPLE_HZ, LOW_HZ, HIGH_HZ)

        freqs = np.fft.rfftfreq(len(pulse), d=1.0 / RESAMPLE_HZ)
        power = np.abs(np.fft.rfft(pulse * np.hanning(len(pulse)))) ** 2
        band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        if not np.any(band):
            return None
        peak_freq = freqs[band][np.argmax(power[band])]
        bpm = float(peak_freq * 60.0)

        # Exponential smoothing so the on-screen number doesn't jitter frame to frame.
        self._bpm_smoothed = (
            bpm if self._bpm_smoothed is None else 0.8 * self._bpm_smoothed + 0.2 * bpm
        )
        self.bpm = self._bpm_smoothed
        return self.bpm


def _pos_algorithm(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps = 1e-8
    rn = r / (np.mean(r) + eps)
    gn = g / (np.mean(g) + eps)
    bn = b / (np.mean(b) + eps)

    x = gn - bn
    y = -2.0 * rn + gn + bn

    alpha = (np.std(x) + eps) / (np.std(y) + eps)
    return x + alpha * y


def _bandpass(
    signal: np.ndarray, fs: float, low: float, high: float, order: int = 3
) -> np.ndarray:
    nyq = fs / 2.0
    b_coef, a_coef = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b_coef, a_coef, signal)
