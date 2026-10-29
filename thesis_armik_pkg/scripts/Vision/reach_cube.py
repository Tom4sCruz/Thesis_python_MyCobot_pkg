#!/usr/bin/env python3
"""
VISION, STAGE 2: see the cube, press SPACE, pick it and place it on the target
==============================================================================

A RealSense camera watches the robot's BASE marker (ID 10), the TARGET
marker (ID 11) and the cube (IDs 0-5 on its faces). The base marker gives
the camera -> robot transform, so cube and target positions come out directly
in armik's robot frame (cm). Nothing moves until you press SPACE.

    python3 scripts/Vision/reach_cube.py --mock                    # real camera, simulated arm
    python3 scripts/Vision/reach_cube.py --port /dev/ttyTHS1       # real arm (take_arm.sh first)
    python3 scripts/Vision/reach_cube.py --port /dev/ttyTHS1 --no-place   # grab + lift only

KEYS
    SPACE   idle    -> start (or resume) the pick-and-place
            moving  -> STOP at once (the stream halts within one setpoint)
    G       toggle "holding a cube" by hand (e.g. it slipped out)
    O       open the gripper (and clear "holding")
    H       go HOME (SPACE stops this too)
    B       forget the locked base/target poses and re-acquire them
    Q / Esc stop and quit

STATES
    A run is a list of steps; "holding" records whether the gripper has a cube:

        ready -> rise -> above cube -> descend -> GRASP (holding = True)
              -> lift -> above target -> lower -> RELEASE (holding = False)
              -> retreat -> home

    SPACE mid-run stops the arm where it is. The next SPACE plans a FRESH run
    from the arm's current pose: not holding -> detect the cube again and
    reach for it; holding -> skip straight to carrying it to the target.
    "holding" turns True as soon as the close command is sent (the gripper
    finishes closing even if you stop the arm). The gripper's own reading
    can't tell "closed on a cube" from "closed on nothing" at the cube-width
    setting, so if the cube falls out press G.

PERCEPTION
    * base and target poses are a running average of the last few frames
      while the arm is idle, and FROZEN while it moves (the arm hides the
      base marker) -- don't move the camera during a run;
    * the cube is fused from every visible face (_vision.cube_from_faces);
      it is assumed to rest on the table: each face is slid along its camera
      ray onto the height it must have (see cube_from_faces -- a 3 cm
      marker's bearing is accurate, its range is not) unless --vision-z;
    * SPACE only starts when the cube estimate has been steady for
      STABLE_FRAMES frames. The cube/target positions are snapshotted then
      and never updated mid-run;
    * while idle the arm's tip (armik FK) is drawn as a magenta dot: if it
      sits on the real gripper, the camera->robot transform is right. It is
      not polled during a move (serial traffic would make the stream stutter).

STOPPING
    The arm is driven through AbortableConnection, an ArmConnection whose
    send_angles() raises once SPACE is pressed. Arm._execute() already turns
    that into a firmware stop() and a failed (ok=False) execution, so armik/
    itself is untouched -- the compiled profile tapes stay valid.

Every waypoint is IK-checked before the first command; an unreachable cube or
target is refused with the reason and the arm does not move.

Units: cm and degrees (armik's public API).
"""

from __future__ import annotations

import os as _os, sys as _sys
# scripts/Vision/ is two levels below the package root -> three dirname() calls
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import argparse
import math
import threading
import time
from collections import deque

import cv2
import numpy as np

from armik import Arm, ArmConnection, ArmError, config, ik
from armik.kinematics import check_joint_limits, check_workspace_bounds
import _vision as V


# =============================================================================
# CONSTANTS
# =============================================================================

# -- arm ------------------------------------------------------------------------
HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
HOME_MOVE_S = 2.5                # minimum duration of a joint-space move (home / ready)
JOINT_MOVE_DPS = 35.0            # ... longer if the biggest joint travel needs it
READY_CM = (15.0, 0.0, 8.0)      # gripper-down pose reached in JOINT space first when the
                                 # gripper is not already pointing down (e.g. from HOME)

# Gripper straight down (rx, ry, rz) -- same as the profiles' PICK_ORIENTATION_DEG.
PICK_ORIENTATION_DEG = (180.0, 0.0, -45.0)
ORIENT_DOWN_TOL_DEG = 10.0       # rx/ry within this of PICK_ORIENTATION = "already pointing down"
# Cube yaw (robot frame, mod 90) that the gripper at PICK_ORIENTATION_DEG's rz
# grabs squarely. With the profiles' TOOL_RPY the jaws close along a robot axis
# at rz = -45, i.e. a cube squared up with the robot axes (yaw 0). Tune if the
# jaws land on the cube's corners.
GRIP_SQUARE_YAW_DEG = 0.0
YAW_SIGN = 1.0                   # flip to -1 if the gripper turns the wrong way to meet the cube
ALIGN_YAW = True                 # False = always grab at PICK_ORIENTATION_DEG's rz

# -- heights (cm) ---------------------------------------------------------------
GRASP_Z_OFFSET_CM = 0.0          # grasp tip z = cube centre z + this. 0 after calibrate_base.py
                                 # (its fit already puts the tip where the hand-guided grasp was)
HOVER_ABOVE_CM = 6.0             # approach / lift height above the grasp height
RELEASE_CLEARANCE_CM = 0.5       # cube bottom this far above the table when it's let go

# -- speeds -----------------------------------------------------------------------
TRAVEL_SPEED_CM_S = 6.0
APPROACH_SPEED_CM_S = 3.0        # descend onto the cube / lower onto the target

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 120.0            # as the profiles
GRIP_CLOSED_DEG = 65.0           # closed ON a 3.5 cm cube
GRIP_WAIT_S = 1.2                # time given to the gripper to open / close

# -- perception -----------------------------------------------------------------
BASE_LOCK_FRAMES = 15            # base / target pose = average of the last N idle sightings
STABLE_FRAMES = 10               # cube estimate must be steady this many frames ...
STABLE_TOL_CM = 0.5              # ... within this radius, before SPACE is accepted
TIP_POLL_S = 0.5                 # idle FK-tip overlay refresh period
VISION_Z_WARN_CM = 3.0           # warn if the seen cube z is this far from the table plane

WINDOW = "reach_cube  (SPACE go/stop, G holding, O open, H home, B re-lock, Q quit)"


# =============================================================================
# CONNECTION THAT CAN BE STOPPED MID-STREAM
# =============================================================================

class AbortableConnection(ArmConnection):
    """ArmConnection whose send_angles() raises once `abort` is set.

    Arm._execute() streams by calling send_angles() once per setpoint and
    catches ArmError by stopping the arm (conn.stop()) and returning
    ok=False -- so setting `abort` halts any streamed move (send_coords,
    move_joints) within one control period."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abort = threading.Event()

    def send_angles(self, angles_deg, deg_per_s):
        if self.abort.is_set():
            raise ArmError("stopped (SPACE)")
        return super().send_angles(angles_deg, deg_per_s)


# =============================================================================
# PERCEPTION
# =============================================================================

class Scene:
    """Per-frame marker detection -> base lock, target and cube estimates."""

    def __init__(self, K, dist, base_calib, args):
        self.K, self.dist = K, dist
        self.calib = base_calib
        self.args = args
        self.detector = V.make_detector()
        self.base_hist = deque(maxlen=BASE_LOCK_FRAMES)      # T_cam_base
        self.target_hist = deque(maxlen=BASE_LOCK_FRAMES)    # target (x, y) robot cm
        self.cube_hist = deque(maxlen=STABLE_FRAMES)         # (center, yaw)
        self.T_robot_cam = None
        self.target_xy = None
        self.cube = None                                     # latest fused estimate
        self.markers = {}

    def relock(self):
        self.base_hist.clear()
        self.target_hist.clear()
        self.T_robot_cam = None
        self.target_xy = None

    @property
    def table_z(self):
        return V.table_z(self.calib)

    def update(self, frame, frozen):
        """frozen=True while the arm moves: base/target locks are not updated."""
        self.markers = V.detect(self.detector, frame)
        m = self.markers
        if not frozen and V.BASE_ID in m:
            self.base_hist.append(V.marker_pose(m[V.BASE_ID], self.args.big_marker_cm,
                                                self.K, self.dist))
            self.T_robot_cam = V.robot_from_camera(V.mean_pose(list(self.base_hist)), self.calib)
        if self.T_robot_cam is None:
            self.cube = None
            return

        if not frozen and V.TARGET_ID in m:
            cands = V.marker_pose_candidates(m[V.TARGET_ID], self.args.big_marker_cm,
                                             self.K, self.dist)
            # lying flat: take the candidate whose normal is most vertical
            T = max((self.T_robot_cam @ T for T, _ in cands), key=lambda T: abs(T[2, 2]))
            self.target_hist.append(T[:2, 3].copy())
            self.target_xy = np.median(np.array(self.target_hist), axis=0)

        self.cube = V.cube_from_faces(m, self.K, self.dist, self.T_robot_cam,
                                      side=V.CUBE_SIDE_CM, marker_len=self.args.cube_marker_cm,
                                      table_z=None if self.args.vision_z else self.table_z)
        if self.cube is not None:
            self.cube_hist.append((self.cube["center"].copy(), self.cube["yaw"]))
        else:
            self.cube_hist.clear()

    def stable_cube(self):
        """(center, yaw) median over STABLE_FRAMES if steady, else None."""
        if len(self.cube_hist) < STABLE_FRAMES:
            return None
        centers = np.array([c for c, _ in self.cube_hist])
        med = np.median(centers, axis=0)
        if np.max(np.linalg.norm(centers - med, axis=1)) > STABLE_TOL_CM:
            return None
        a = np.radians(4.0 * np.array([y for _, y in self.cube_hist]))
        yaw = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 4.0
        return med, V.wrap90(yaw)


# =============================================================================
# MOTION
# =============================================================================

def grip_rz(cube_yaw):
    """Gripper rz that meets the cube square-on, within +-45 deg of the default."""
    rz0 = PICK_ORIENTATION_DEG[2]
    if not ALIGN_YAW:
        return rz0
    return rz0 + YAW_SIGN * V.wrap90(cube_yaw - GRIP_SQUARE_YAW_DEG)


def _angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _pointing_down(tip):
    """Is the gripper already (nearly) straight down? tip = get_coords()."""
    rx, ry, _ = PICK_ORIENTATION_DEG
    return (abs(_angle_diff(tip[3], rx)) <= ORIENT_DOWN_TOL_DEG
            and abs(_angle_diff(tip[4], ry)) <= ORIENT_DOWN_TOL_DEG)


def joint_move_duration(q_from, q_to):
    travel = float(np.max(np.abs(np.asarray(q_to) - np.asarray(q_from))))
    return max(HOME_MOVE_S, travel / JOINT_MOVE_DPS)


def _ik_target(x, y, z, rx, ry, rz):
    """armik's internal desired vector (mm / deg), as Arm._plan builds it."""
    z_mm = z * 10.0 + (config.JOINT_1_HEIGHT_CM * 10.0 if config.Z_RELATIVE_TO_JOINT1 else 0.0)
    return np.array([x * 10.0, y * 10.0, z_mm, rx, ry, rz], dtype=float)


def ik_check(points, q_seed):
    """IK-solve each (name, x, y, z, rz) gripper-down point in order, seeding each from
    the previous solution. Returns (None, last_q) if all are fine, else (reason, None)."""
    rx, ry, _ = PICK_ORIENTATION_DEG
    mask = np.ones(6, dtype=bool)
    q = np.asarray(q_seed, dtype=float)
    for name, x, y, z, rz in points:
        why = check_workspace_bounds([x * 10.0, y * 10.0, z * 10.0])
        if why:
            return f"{name} ({x:.1f}, {y:.1f}, {z:.1f}) cm: {why}", None
        sol = ik.try_solve(_ik_target(x, y, z, rx, ry, rz), mask, q)
        if sol is None:
            return f"{name} ({x:.1f}, {y:.1f}, {z:.1f}) cm: no IK solution", None
        problems = check_joint_limits(sol)
        if problems:
            return f"{name}: " + "; ".join(problems), None
        q = sol
    return None, q


class Runner:
    """Runs one pick-and-place in a worker thread; SPACE sets conn.abort."""

    def __init__(self, arm: Arm, conn: AbortableConnection, no_place: bool):
        self.arm, self.conn = arm, conn
        self.no_place = no_place
        self.holding = False
        self.step = ""
        self.status = "IDLE"           # IDLE / RUNNING / STOPPED / DONE / HOLDING / FAILED
        self.message = "press SPACE to pick the cube"
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        self.conn.abort.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    # -- building a run -------------------------------------------------------

    def start_pick(self, cube_center, cube_yaw, target_xy, table_z):
        """Plan the run from the CURRENT pose and start it. Returns an error string
        (nothing moves) or None."""
        q0 = np.array(self.arm.get_angles(), dtype=float)
        tip = self.arm.get_coords()
        down = _pointing_down(tip)

        place_z = table_z + V.CUBE_SIDE_CM / 2.0 + RELEASE_CLEARANCE_CM + GRASP_Z_OFFSET_CM
        carry_z = place_z + HOVER_ABOVE_CM
        if not self.holding:
            rz = grip_rz(cube_yaw)
            grasp = (float(cube_center[0]), float(cube_center[1]),
                     float(cube_center[2]) + GRASP_Z_OFFSET_CM)
            hover_z = grasp[2] + HOVER_ABOVE_CM
            carry_z = max(carry_z, hover_z)
        else:
            # resuming with the cube in hand: keep the yaw it was grabbed with
            rz = tip[5] if down else PICK_ORIENTATION_DEG[2]

        steps, checks = [], []
        need_ready = not down
        if need_ready:
            checks.append(("ready", *READY_CM, rz))
            steps.append(("ready", lambda: self._ready(rz)))

        if not self.holding:
            # from wherever the arm is: up, across, down onto the cube
            steps.append(("rise", lambda: self._rise(hover_z, rz)))
            steps.append(("open gripper", lambda: self._grip(GRIP_OPEN_DEG)))
            steps.append(("above cube", lambda: self._move(grasp[0], grasp[1], hover_z, rz)))
            steps.append(("descend", lambda: self._move(*grasp, rz, APPROACH_SPEED_CM_S)))
            steps.append(("grasp", self._grasp))
            steps.append(("lift", lambda: self._move(grasp[0], grasp[1], carry_z, rz)))
            checks += [("above cube", grasp[0], grasp[1], hover_z, rz),
                       ("grasp", *grasp, rz),
                       ("lift", grasp[0], grasp[1], carry_z, rz)]
        else:
            steps.append(("rise", lambda: self._rise(carry_z, rz)))

        if not self.no_place:
            if target_xy is None:
                return "TARGET marker not seen yet"
            tx, ty = float(target_xy[0]), float(target_xy[1])
            steps += [("above target", lambda: self._move(tx, ty, carry_z, rz)),
                      ("lower", lambda: self._move(tx, ty, place_z, rz, APPROACH_SPEED_CM_S)),
                      ("release", self._release),
                      ("retreat", lambda: self._move(tx, ty, carry_z, rz)),
                      ("home", self._home)]
            checks += [("above target", tx, ty, carry_z, rz),
                       ("place", tx, ty, place_z, rz)]

        seed = q0
        if need_ready:
            # check the ready pose from the current angles, the rest from there
            why, seed = ik_check(checks[:1], q0)
            if why:
                return why
            checks = checks[1:]
        why, _ = ik_check(checks, seed)
        if why:
            return why

        self.conn.abort.clear()
        self.status, self.step = "RUNNING", ""
        self.message = (f"cube ({grasp[0]:.1f}, {grasp[1]:.1f}, {grasp[2]:.1f}) cm  rz {rz:.0f}"
                        if not self.holding else "carrying the held cube")
        print(f"[run] {'carry' if self.holding else 'pick'}: {self.message}; "
              f"steps: {', '.join(n for n, _ in steps)}")
        self._thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
        self._thread.start()
        return None

    def start_home(self):
        self.conn.abort.clear()
        self.status, self.message = "RUNNING", "going home"
        self._thread = threading.Thread(target=self._run, args=([("home", self._home)],),
                                        daemon=True)
        self._thread.start()

    # -- worker ---------------------------------------------------------------

    def _run(self, steps):
        for name, fn in steps:
            self.step = name
            print(f"[run]   {name}")
            ok = fn()
            if self.conn.abort.is_set():
                self.status = "STOPPED"
                self.message = (f"stopped during '{name}' -- SPACE: "
                                + ("carry the held cube on" if self.holding else "reach again"))
                print(f"[run] STOPPED during {name} (holding={self.holding})")
                return
            if not ok:
                self.status = "FAILED"
                self.message = f"'{name}' failed: {self.arm.last_error}"
                print(f"[run] FAILED during {name}: {self.arm.last_error}")
                return
        self.step = ""
        if self.holding:
            self.status, self.message = "HOLDING", "holding the cube -- SPACE: carry it to the target"
        else:
            self.status, self.message = "DONE", "done -- SPACE for another pick"
        print(f"[run] {self.status}")

    def _move(self, x, y, z, rz, speed=TRAVEL_SPEED_CM_S):
        rx, ry, _ = PICK_ORIENTATION_DEG
        if self.arm.send_coords(x, y, z, rx, ry, rz, speed=speed):
            return True
        return "already at the requested coordinates" in (self.arm.last_error or "")

    def _rise(self, z, rz):
        """Straight up to height z (skipped if already at least that high)."""
        tip = self.arm.get_coords()
        if tip[2] >= z - 0.3:
            return True
        return self._move(tip[0], tip[1], z, rz)

    def _ready(self, rz):
        """Joint-space move to a gripper-down pose (no Cartesian line from HOME)."""
        rx, ry, _ = PICK_ORIENTATION_DEG
        q0 = np.array(self.arm.get_angles(), dtype=float)
        q = ik.try_solve(_ik_target(*READY_CM, rx, ry, rz), np.ones(6, dtype=bool), q0)
        if q is None:
            self.arm.last_error = "no IK solution for READY_CM"
            return False
        return bool(self.arm.move_joints(q, joint_move_duration(q0, q)))

    def _home(self):
        q0 = self.arm.get_angles()
        return bool(self.arm.move_joints(HOME, joint_move_duration(q0, HOME)))

    def _wait(self, seconds):
        """Abortable sleep: True if it ran out normally."""
        return not self.conn.abort.wait(seconds)

    def _grip(self, deg):
        return bool(self.arm.send_gripper(deg)) and self._wait(GRIP_WAIT_S)

    def _grasp(self):
        if not self.arm.send_gripper(GRIP_CLOSED_DEG):
            return False
        self.holding = True               # the gripper closes even if the arm is stopped now
        return self._wait(GRIP_WAIT_S)

    def _release(self):
        if not self.arm.send_gripper(GRIP_OPEN_DEG):
            return False
        self.holding = False
        return self._wait(GRIP_WAIT_S)


# =============================================================================
# OVERLAY
# =============================================================================

STATUS_COLORS = {"IDLE": (255, 255, 255), "RUNNING": (0, 220, 255), "STOPPED": (0, 140, 255),
                 "DONE": (0, 220, 0), "HOLDING": (255, 200, 0), "FAILED": (0, 0, 255)}


def draw_overlay(frame, scene: Scene, runner: Runner, tip, fps, stable):
    K, dist = scene.K, scene.dist
    if scene.markers:
        ids = np.array(list(scene.markers), dtype=np.int32).reshape(-1, 1)
        corners = [c.reshape(1, 4, 2).astype(np.float32) for c in scene.markers.values()]
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    y = 25
    def line(text, color=(255, 255, 255)):
        nonlocal y
        V.put_text(frame, text, (10, y), color)
        y += 24

    T = scene.T_robot_cam
    if T is None:
        line(f"BASE marker (id {V.BASE_ID}) not seen -- no robot frame yet", (0, 0, 255))
    else:
        V.draw_robot_axes(frame, T, K, dist)
        if scene.target_xy is not None:
            p = [scene.target_xy[0], scene.target_xy[1], scene.table_z]
            V.draw_point(frame, p, T, K, dist, (255, 0, 255), "target")
        if scene.cube is not None:
            c = scene.cube
            V.draw_cube_box(frame, c["center"], c["yaw"], V.CUBE_SIDE_CM, T, K, dist,
                            (0, 220, 0) if stable else (0, 200, 255))
            x, yy, z = c["center"]
            line(f"cube ({x:6.1f}, {yy:6.1f}, {z:5.1f}) cm  yaw {c['yaw']:5.1f}  "
                 f"faces {c['faces']}  {'steady' if stable else 'settling'}")
            dz = c["raw_z"] - (scene.table_z + V.CUBE_SIDE_CM / 2.0)
            if abs(dz) > VISION_Z_WARN_CM and not scene.args.vision_z:
                line(f"  seen z is {dz:+.1f} cm off the table plane (held? stacked? bad base pose?)",
                     (0, 140, 255))
        else:
            line("cube not seen", (0, 140, 255))
        if scene.target_xy is not None:
            line(f"target ({scene.target_xy[0]:6.1f}, {scene.target_xy[1]:6.1f}) cm")
        else:
            line(f"TARGET marker (id {V.TARGET_ID}) not seen", (0, 140, 255))
        if tip is not None:
            V.draw_point(frame, tip[:3], T, K, dist, (255, 0, 200), "tip (FK)", radius=5)

    col = STATUS_COLORS.get(runner.status, (255, 255, 255))
    step = f" [{runner.step}]" if runner.running and runner.step else ""
    line(f"{runner.status}{step}   holding: {'YES' if runner.holding else 'no'}", col)
    line(runner.message, col)
    h = frame.shape[0]
    V.put_text(frame, f"{fps:4.1f} fps", (10, h - 12))


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Detect the cube with ArUco, SPACE to pick and place it.")
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--mock", action="store_true", help="simulated arm (the camera is still real)")
    V.add_camera_args(ap)
    ap.add_argument("--calib", default=None,
                    help="camera .npz (calibrate_camera.py); default: RealSense factory intrinsics")
    ap.add_argument("--base-calib", default=V.BASE_CALIB_PATH,
                    help="teach-pairs correction from calibrate_base.py (skipped if missing)")
    ap.add_argument("--cube-marker-cm", type=float, default=V.CUBE_MARKER_CM,
                    help="measured black square of a cube-face marker")
    ap.add_argument("--big-marker-cm", type=float, default=V.BIG_MARKER_CM,
                    help="measured black square of the BASE / TARGET markers")
    ap.add_argument("--vision-z", action="store_true",
                    help="use the cube z seen by the camera instead of the table plane")
    ap.add_argument("--no-place", action="store_true",
                    help="stop after grasping and lifting (first tests)")
    args = ap.parse_args()

    calib = V.load_base_calib(args.base_calib)
    print(f"base correction: {calib}"
          + ("" if _os.path.exists(args.base_calib or "") else "  (none -- run calibrate_base.py)"))

    cam = V.open_camera(args.source, args.camera, args.width, args.height)
    conn = AbortableConnection(args.port, mock=args.mock)
    arm = Arm(connection=conn)
    arm.set_single_joint(0)                   # streamed moves only: those are what SPACE can stop
    runner = Runner(arm, conn, args.no_place)

    scene = None
    tip, t_tip = None, 0.0
    fps, t_prev = 0.0, time.monotonic()
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
                scene = Scene(K, dist, calib, args)

            moving = runner.running
            scene.update(frame, frozen=moving)
            stable = scene.stable_cube()

            now = time.monotonic()
            if moving:
                tip = None
            elif now - t_tip > TIP_POLL_S:
                try:
                    tip = arm.get_coords()
                except Exception as exc:            # a missed serial read: just skip the dot
                    print(f"tip read failed: {exc}")
                    tip = None
                t_tip = now

            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            draw_overlay(frame, scene, runner, tip, fps, stable is not None)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if runner.running:
                    print("[key] SPACE -> stop")
                    runner.stop()
                elif scene.T_robot_cam is None:
                    runner.message = "can't start: BASE marker not seen"
                elif not runner.holding and stable is None:
                    runner.message = "can't start: cube not seen / not steady yet"
                else:
                    center, yaw = stable if stable is not None else (None, 0.0)
                    err = runner.start_pick(center, yaw, scene.target_xy, scene.table_z)
                    if err:
                        runner.status, runner.message = "FAILED", f"refused: {err}"
                        print(f"[run] refused: {err}")
            elif key == ord("g") and not runner.running:
                runner.holding = not runner.holding
                runner.message = f"holding set to {runner.holding} by hand"
            elif key == ord("o") and not runner.running:
                arm.send_gripper(GRIP_OPEN_DEG)
                runner.holding = False
                runner.message = "gripper opened"
            elif key == ord("h") and not runner.running:
                runner.start_home()
            elif key == ord("b") and not runner.running:
                scene.relock()
                runner.message = "re-acquiring base / target"
    finally:
        runner.stop()
        runner.join(timeout=3.0)
        cam.release()
        cv2.destroyAllWindows()
        arm.close()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
