#!/usr/bin/env python3
"""
VISION, NO ROBOT: capture the marker layout -> scene.json (for sim_pick_rviz.py)
================================================================================

Point a camera (the laptop webcam by default) at the BASE (10), TARGET (11)
and cube (0-5) markers. Once all three have been seen and held steady for
STEADY_S seconds, their positions are saved to scene.json and the script
exits -- the camera is free again. sim_pick_rviz.py then replays the pick in
RViz from that file, no robot and no camera needed.

Robot frame for the visualisation: the robot stands exactly on the CENTRE of
the BASE marker (VIZ_BASE_POSE: x = y = 0, the table height and "robot +X"
arrow yaw of _vision.BASE_MARKER_POSE). base_calib.json is NOT applied -- it
belongs to the real camera and arm.

Intrinsics: a laptop webcam has no factory calibration, so without --calib
the camera model is a rough guess (distances maybe 10-20 % off). Fine for a
visualisation; for better numbers run calibrate_camera.py --source v4l2 and
pass its .npz with --calib. The RealSense's camera_calib.npz is never loaded
automatically -- it belongs to a different camera.

    python3 scripts/Vision/capture_scene.py                     # webcam 0 -> scene.json
    python3 scripts/Vision/capture_scene.py --camera 2 --scene lab1.json
    python3 scripts/Vision/capture_scene.py --source realsense  # works with the RealSense too

Keys: S save now (if everything is visible), Q / Esc quit without saving.
Runs on the host (needs a GUI OpenCV build: pip install opencv-python).
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import argparse
import json
import math
import time

import cv2
import numpy as np

import _vision as V
from reach_cube import MAX_TILTED_REACH_CM, Scene


# =============================================================================
# CONSTANTS
# =============================================================================

STEADY_S = 3.0                   # everything visible and still this long -> save
STEADY_TOL_CM = 0.5              # "still": cube centre / target within this of the run's start
SCENE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scene.json")
VIZ_BASE_POSE = (0.0, 0.0, V.BASE_MARKER_POSE[2], V.BASE_MARKER_POSE[3])
CAMERA_SOURCE = "v4l2"           # laptop webcam by default
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720

WINDOW = "capture_scene  (hold still 3 s -> saved;  S save now, Q quit)"


# =============================================================================
# STEADINESS
# =============================================================================

class SteadyTracker:
    """Times how long the cube and target have stayed put.

    A run starts at the first sample; any later sample further than tol from
    that first one restarts the run. steady_s() is the run's length, value()
    the per-coordinate median over it."""

    def __init__(self, tol_cm=STEADY_TOL_CM):
        self.tol = tol_cm
        self.reset()

    def reset(self):
        self.samples = []                 # (t, cube xyz, cube yaw, target xy)

    def add(self, t, cube_center, cube_yaw, target_xy):
        c, txy = np.asarray(cube_center, float), np.asarray(target_xy, float)
        if self.samples:
            _, c0, _, t0 = self.samples[0]
            if np.linalg.norm(c - c0) > self.tol or np.linalg.norm(txy - t0) > self.tol:
                self.samples = []
        self.samples.append((t, c, float(cube_yaw), txy))

    def steady_s(self, now):
        return now - self.samples[0][0] if self.samples else 0.0

    def value(self):
        cs = np.array([s[1] for s in self.samples])
        a = np.radians(4.0 * np.array([s[2] for s in self.samples]))
        yaw = V.wrap90(math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 4.0)
        return np.median(cs, axis=0), yaw, np.median(np.array([s[3] for s in self.samples]), axis=0)


def scene_dict(cube_center, cube_yaw, target_xy, table_z, faces, camera, intrinsics, args):
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera,
        "intrinsics": intrinsics,
        "base_pose": list(VIZ_BASE_POSE),
        "table_z": float(table_z),
        "cube": {"center": [round(float(v), 3) for v in cube_center],
                 "yaw": round(float(cube_yaw), 2), "faces": [int(f) for f in faces]},
        "target": {"xy": [round(float(v), 3) for v in target_xy]},
        "marker_cm": {"cube": args.cube_marker_cm, "big": args.big_marker_cm},
    }


def reach_report(scene):
    for name, xy in (("cube", scene["cube"]["center"][:2]), ("target", scene["target"]["xy"])):
        r = math.hypot(*xy)
        ok = "OK" if r <= MAX_TILTED_REACH_CM else f"OUT OF REACH (> ~{MAX_TILTED_REACH_CM:.0f} cm)"
        print(f"  {name:6s} ({xy[0]:6.1f}, {xy[1]:6.1f}) cm  -> {r:5.1f} cm from the base axis  {ok}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Capture BASE/TARGET/cube marker positions -> scene.json")
    V.add_camera_args(ap)
    ap.set_defaults(source=CAMERA_SOURCE, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    ap.add_argument("--calib", default=None, help="camera .npz for THIS camera (default: rough guess)")
    ap.add_argument("--scene", default=SCENE_PATH, help="output JSON")
    ap.add_argument("--cube-marker-cm", type=float, default=V.CUBE_MARKER_CM)
    ap.add_argument("--big-marker-cm", type=float, default=V.BIG_MARKER_CM)
    args = ap.parse_args()
    args.scene = V.check_out_path(args.scene, "--scene")
    args.cube_z = "table"                    # Scene option: cube rests on the table

    cam = V.open_camera(args.source, args.camera, args.width, args.height)
    scene = None
    tracker = SteadyTracker()
    saved = None
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Camera frame grab failed", file=_sys.stderr)
                return 1
            if scene is None:
                h, w = frame.shape[:2]
                K, dist, how = V.camera_model(cam, w, h, args.calib)
                print(f"camera {cam.name}: {w}x{h}, {how}")
                scene = Scene(K, dist, None, args, base_pose=VIZ_BASE_POSE)
            now = time.monotonic()
            scene.update(frame, frozen=False)

            # --- overlay ---------------------------------------------------
            m = scene.markers
            if m:
                ids = np.array(list(m), dtype=np.int32).reshape(-1, 1)
                cv2.aruco.drawDetectedMarkers(
                    frame, [c.reshape(1, 4, 2).astype(np.float32) for c in m.values()], ids)
            T = scene.T_robot_cam
            seen = {"BASE": T is not None and V.BASE_ID in m,
                    "TARGET": V.TARGET_ID in m and scene.target_xy is not None,
                    "cube": scene.cube is not None}
            if T is not None:
                V.draw_robot_axes(frame, T, K, dist)
                if scene.target_xy is not None:
                    V.draw_point(frame, [*scene.target_xy, scene.table_z], T, K, dist,
                                 (255, 0, 255), "target")
                if scene.cube is not None:
                    V.draw_cube_box(frame, scene.cube["center"], scene.cube["yaw"],
                                    V.CUBE_SIDE_CM, T, K, dist, (0, 220, 0))

            if all(seen.values()):
                tracker.add(now, scene.cube["center"], scene.cube["yaw"], scene.target_xy)
            else:
                tracker.reset()
            held = tracker.steady_s(now)
            status = "   ".join(f"{k}: {'ok' if v else '--'}" for k, v in seen.items())
            V.put_text(frame, status, (10, 25),
                       (0, 220, 0) if all(seen.values()) else (0, 140, 255))
            if saved is None:
                V.put_text(frame, f"steady {held:3.1f} / {STEADY_S:.0f} s", (10, 50))
                bar = int((frame.shape[1] - 20) * min(held / STEADY_S, 1.0))
                cv2.rectangle(frame, (10, 60), (10 + bar, 70), (0, 220, 0), -1)
            else:
                V.put_text(frame, f"SAVED -> {args.scene}", (10, 50), (0, 220, 0))
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Quit without saving." if saved is None else "Done.")
                break
            if saved is not None:
                if now - saved > 1.0:
                    break
                continue
            if held >= STEADY_S or (key == ord("s") and tracker.samples):
                center, yaw, target_xy = tracker.value()
                data = scene_dict(center, yaw, target_xy, scene.table_z, scene.cube["faces"],
                                  cam.name, args.calib or "guess", args)
                with open(args.scene, "w") as fh:
                    json.dump(data, fh, indent=2)
                print(f"Saved {args.scene}  (steady {held:.1f} s)")
                reach_report(data)
                saved = now
            elif key == ord("s"):
                print("S: need BASE, TARGET and the cube all visible")
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
