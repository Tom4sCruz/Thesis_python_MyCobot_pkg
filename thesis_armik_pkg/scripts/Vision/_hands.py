"""
HAND tracking for reach_cube.py --on-hand: MediaPipe Hands on the colour
frame, the RealSense depth stream for the distance.

MediaPipe only gives the 21 hand landmarks in PIXELS (its own z is relative
to the wrist, not a range), so a landmark is just a ray from the camera. The
depth stream (camera.point_at, _vision.py) gives the range along it, and the
camera -> robot transform (the BASE marker) puts the point in the robot frame.

MediaPipe is imported on first use, so nothing else needs it installed. The
legacy `mp.solutions.hands` (model built in) is used when the installed
MediaPipe has it; otherwise the Tasks HandLandmarker, which needs the model
file (HAND_MODEL_PATH, download URL below).

Units: cm, robot frame (as _vision.py).
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

import _vision as V


HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
DETECT_SCALE = 0.5               # MediaPipe runs on a frame shrunk by this (CPU on the Jetson)
MIN_DETECT_CONF = 0.6
MIN_TRACK_CONF = 0.5

PALM_IDS = (0, 5, 9, 13, 17)     # wrist + the 4 knuckles: the palm centre
# Landmark pairs to draw (MediaPipe's HAND_CONNECTIONS)
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


class HandTracker:
    """detect(frame) -> (21, 2) pixel landmarks of the most confident hand, or None."""

    def __init__(self, model_path=HAND_MODEL_PATH):
        try:
            import mediapipe as mp
        except ImportError:
            raise SystemExit("mediapipe is not installed (pip install mediapipe; wheels need "
                             "Python >= 3.9 on the Jetson's aarch64) -- needed for --on-hand")
        self._mp = mp
        self._t0 = time.monotonic()
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            self._legacy = solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=0,
                min_detection_confidence=MIN_DETECT_CONF, min_tracking_confidence=MIN_TRACK_CONF)
            self._tasks = None
        else:
            if not os.path.exists(model_path):
                raise SystemExit(f"this MediaPipe needs the hand model file: download\n  "
                                 f"{HAND_MODEL_URL}\nto {model_path} (or pass --hand-model)")
            vision = mp.tasks.vision
            opts = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=MIN_DETECT_CONF,
                min_tracking_confidence=MIN_TRACK_CONF)
            self._tasks = vision.HandLandmarker.create_from_options(opts)
            self._legacy = None
            self._last_ts = -1

    def detect(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            res = self._legacy.process(rgb)
            hands = res.multi_hand_landmarks
            if not hands:
                return None
            lms = hands[0].landmark
        else:
            ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
            self._last_ts = ts
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self._tasks.detect_for_video(img, ts)
            if not res.hand_landmarks:
                return None
            lms = res.hand_landmarks[0]
        return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=float)

    def close(self):
        if self._legacy is not None:
            self._legacy.close()
        elif self._tasks is not None:
            self._tasks.close()


def palm_point(landmarks, point_fn, T_robot_cam):
    """Palm centre in the robot frame (cm): the median of the depth points at
    the wrist and the knuckles (PALM_IDS). None if fewer than 2 have depth."""
    pts = [point_fn(*landmarks[i]) for i in PALM_IDS]
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return None
    p_cam = np.median(np.array(pts), axis=0)
    return T_robot_cam[:3, :3] @ p_cam + T_robot_cam[:3, 3]


def draw_hand(img, landmarks, color=(0, 200, 255)):
    for a, b in CONNECTIONS:
        cv2.line(img, V._pt(landmarks[a]), V._pt(landmarks[b]), color, 2, cv2.LINE_AA)
    for p in landmarks:
        cv2.circle(img, V._pt(p), 3, color, -1, cv2.LINE_AA)
