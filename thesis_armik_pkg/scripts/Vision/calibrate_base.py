#!/usr/bin/env python3
"""
VISION: teach-pairs refinement of the camera -> robot transform -> base_calib.json
=================================================================================

reach_cube.py turns camera positions into robot positions through the BASE
marker, whose pose in the robot frame is _vision.BASE_MARKER_POSE -- a ruler
measurement, good to maybe a centimetre and a few degrees. This script
measures what is left over with the arm itself: you place the cube, the
camera says where it is, you hand-guide the gripper onto it (the way it should
grasp it), and armik's FK says where the gripper tip really is. From 3-5 such
pairs it fits a correction (yaw about z, dx, dy, dz) and saves it to
base_calib.json, which reach_cube.py loads automatically.

dz also absorbs the grasp height (where on the cube the tip sits when it
grabs), so with a calibration reach_cube.GRASP_Z_OFFSET_CM stays 0.

    python3 scripts/Vision/calibrate_base.py --port /dev/ttyTHS1   # take_arm.sh first
    python3 scripts/Vision/calibrate_base.py --mock                # flow only

PROCEDURE (the camera and the base marker must not move afterwards)
    1. The servos are released at start -- SUPPORT THE ARM, it goes limp.
    2. Put the cube somewhere in the workspace, keep the arm out of the way
       until the cube box turns green, press C (cube position captured).
    3. Guide the gripper down around the cube, centred and gripper-down as for
       a grasp, hold it still, press T (tip captured -> one pair).
    4. Repeat 2-3 at 3-5 spots spread over the area you'll use (near / far,
       left / right). U drops the last pair.
    5. F fits, prints the per-pair residual and saves. Residuals of a few mm
       are good; a pair far off the rest is a bad capture -- U it and redo.
    Q / Esc quits (servos are re-engaged on the way out).

Uses the same camera flags as reach_cube.py (--source, --calib, ...).
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

from armik import Arm, ArmConnection, config
import _vision as V
from reach_cube import Scene


WINDOW = "calibrate_base  (C cube, T tip, U undo, F fit+save, Q quit)"
MIN_PAIRS = 3


# =============================================================================
# FIT
# =============================================================================

def fit_correction(cube_pts, tip_pts):
    """Rigid correction about z mapping camera-derived cube positions (nominal
    frame) onto FK tip positions: tip ~= Rz(yaw) @ cube + (dx, dy, dz).
    Least squares in xy (2D Kabsch), dz = mean z difference."""
    P = np.asarray(cube_pts, dtype=float)
    Q = np.asarray(tip_pts, dtype=float)
    pc, qc = P[:, :2].mean(axis=0), Q[:, :2].mean(axis=0)
    A, B = P[:, :2] - pc, Q[:, :2] - qc
    H = A.T @ B
    yaw = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]) if len(P) > 1 else 0.0
    R = V.rot_z(math.degrees(yaw))[:2, :2]
    t = qc - R @ pc
    dz = float(np.mean(Q[:, 2] - P[:, 2]))
    calib = {"yaw_deg": math.degrees(yaw), "dx_cm": float(t[0]), "dy_cm": float(t[1]),
             "dz_cm": dz}
    fitted = (V.correction_T(calib) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    return calib, np.linalg.norm(fitted - Q, axis=1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Fit the camera->robot correction from hand-guided grasps.")
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--mock", action="store_true")
    V.add_camera_args(ap)
    ap.add_argument("--calib", default=None, help="camera .npz; default: RealSense factory intrinsics")
    ap.add_argument("--out", default=V.BASE_CALIB_PATH)
    ap.add_argument("--cube-marker-cm", type=float, default=V.CUBE_MARKER_CM)
    ap.add_argument("--big-marker-cm", type=float, default=V.BIG_MARKER_CM)
    args = ap.parse_args()
    args.vision_z = False                     # Scene option: cube z from the table plane

    cam = V.open_camera(args.source, args.camera, args.width, args.height)
    arm = Arm(connection=ArmConnection(args.port, mock=args.mock))
    print("Releasing the servos -- SUPPORT THE ARM.")
    time.sleep(1.0)
    arm.conn.release_all_servos()

    scene = None
    pairs = []                                 # (cube xyz, tip xyz), nominal frame
    pending = None                             # captured cube waiting for its tip
    msg = "place the cube, arm clear, press C"
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
                scene = Scene(K, dist, None, args)      # None = no correction: fit the nominal frame
            scene.update(frame, frozen=False)
            stable = scene.stable_cube()

            T = scene.T_robot_cam
            if T is not None:
                V.draw_robot_axes(frame, T, K, dist)
                if scene.cube is not None:
                    V.draw_cube_box(frame, scene.cube["center"], scene.cube["yaw"], V.CUBE_SIDE_CM,
                                    T, K, dist, (0, 220, 0) if stable else (0, 200, 255))
                if pending is not None:
                    V.draw_point(frame, pending, T, K, dist, (255, 200, 0), "captured")
            V.put_text(frame, f"pairs: {len(pairs)}   {'BASE not seen' if T is None else ''}", (10, 25))
            V.put_text(frame, msg, (10, 50), (0, 220, 255))
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                if T is None or stable is None:
                    msg = "C: need the BASE marker and a steady cube"
                else:
                    pending = stable[0].copy()
                    msg = (f"cube at ({pending[0]:.1f}, {pending[1]:.1f}, {pending[2]:.1f}) -- "
                           "guide the gripper onto it, press T")
            elif key == ord("t"):
                if pending is None:
                    msg = "T: capture the cube with C first"
                else:
                    tip = np.array(arm.get_coords()[:3], dtype=float)
                    pairs.append((pending, tip))
                    d = np.linalg.norm(tip - pending)
                    print(f"pair {len(pairs)}: cube {np.round(pending, 2)}  tip {np.round(tip, 2)}  "
                          f"(raw gap {d:.1f} cm)")
                    pending = None
                    msg = f"pair {len(pairs)} kept -- next spot (C), or F to fit"
            elif key == ord("u") and (pairs or pending is not None):
                if pending is not None:
                    pending = None
                else:
                    pairs.pop()
                msg = f"undone, {len(pairs)} pairs"
            elif key == ord("f"):
                if len(pairs) < MIN_PAIRS:
                    msg = f"F: need at least {MIN_PAIRS} pairs, have {len(pairs)}"
                    continue
                calib, res = fit_correction([p for p, _ in pairs], [q for _, q in pairs])
                print(f"\ncorrection: yaw {calib['yaw_deg']:+.2f} deg, "
                      f"dx {calib['dx_cm']:+.2f}  dy {calib['dy_cm']:+.2f}  dz {calib['dz_cm']:+.2f} cm")
                for i, r in enumerate(res):
                    print(f"  pair {i + 1}: residual {10 * r:.1f} mm")
                rms = float(np.sqrt(np.mean(res ** 2)))
                print(f"  RMS {10 * rms:.1f} mm  ({'PASS' if rms < 0.5 else 'check the worst pair'})")
                calib["rms_cm"] = rms
                calib["pairs"] = [{"cube": p.tolist(), "tip": q.tolist()} for p, q in pairs]
                calib["base_marker_pose"] = list(V.BASE_MARKER_POSE)
                with open(args.out, "w") as fh:
                    json.dump(calib, fh, indent=2)
                print(f"Saved {args.out}")
                msg = f"saved ({10 * rms:.1f} mm RMS) -- Q to quit"
    finally:
        print("Re-engaging the servos.")
        arm.conn.focus_all_servos()
        cam.release()
        cv2.destroyAllWindows()
        arm.close()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
