---
title: Contactless Vitals
emoji: 💓
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
---

# Contactless Vitals — Webcam Heart Rate (rPPG)

Estimates your heart rate from a live webcam feed, no wearable required.
Runs entirely on **CPU** — no GPU / ZeroGPU needed.

## How it works

1. **Face tracking** — MediaPipe Face Mesh locates 468 facial landmarks per
   frame and derives a forehead ROI (a skin patch that moves rigidly with
   the head, away from eyes/mouth motion). See `roi.py`.
2. **Pulse extraction** — the mean RGB of that ROI is buffered over a
   rolling ~8s window and processed with the **POS algorithm**
   (Wang et al., 2017): temporally-normalized RGB is projected onto a
   plane orthogonal to skin tone, which cancels most lighting/motion
   artifacts and leaves the blood-volume-pulse signal. A bandpass filter
   (0.7–4 Hz) and FFT peak-pick turn that into a BPM estimate. See
   `rppg.py` — this is classical signal processing, not a trained model.
3. **Live video** — streamed over WebRTC via
   [`fastrtc`](https://github.com/gradio-app/fastrtc) for low latency,
   wrapped in a Gradio UI.

## Why CPU-only

The core pipeline (face landmarks + NumPy/SciPy signal processing) doesn't
need a GPU, and ZeroGPU's daily quota for anonymous visitors (2 minutes)
would make the demo unreliable for someone clicking in cold from a link.
Keeping it CPU-only means it works instantly for anyone, no login required.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the printed local URL and allow webcam access.

## Get a good reading

1. Sit facing a window or a steady lamp. Avoid strong backlight and
   flickering screens.
2. Select **Start measurement** and allow camera access when your browser asks.
3. Center your face and keep your forehead visible. A green box marks the
   skin region being sampled.
4. Hold still and breathe normally. A reading usually appears after 6–12
   seconds.

If the app cannot find your face, improve the front lighting and face the
camera directly.

## Limitations

- Needs steady, reasonably even lighting and a mostly-still face — this is
  inherent to rPPG, not a bug.
- Not a medical device; this is a portfolio/demo project, not for
  diagnostic use.
- Optimized for one active session at a time on free CPU hardware; several
  concurrent visitors will see it slow down.
