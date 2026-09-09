#!/usr/bin/env python3
"""
MOVEMENT PROFILE: Low-human / High-robot  (v2 -- reliable grab)
=============================================================

Same idea as LowH-HighR.py -- move cubes from one side of the frame to the
other like a classic industrial robot -- but with a grab that actually works:

  * the HORIZONTAL transit between cubes turns only J1 (the base), with J6
    counter-rotated by the same amount so the GRIPPER ALWAYS FACES THE SAME
    ABSOLUTE DIRECTION -- J2..J5 never move during the transit;
  * the VERTICAL grab / place is a COORDINATED multi-joint move that keeps the
    end-effector pointing straight down the whole way, so the gripper descends
    onto the cube instead of swiping sideways into it;
  * per cube, both the reach and the carry run the SAME three steps:
        coordinated lift to CARRY_HEIGHT_CM  ->  J1+J6 swing to the next
        azimuth  ->  coordinated descent onto the cube / drop point.
    Lifting first means a swing never drags the gripper along the table
    after a release.
  * cubes grabbed in a fixed order (PICK_ORDER, left -> right), same trajectory
    every run, nothing randomised.

    python3 scripts/Profiles/LowH-HighR_v2.py --mock --yes      # no hardware
    python3 scripts/Profiles/LowH-HighR_v2.py --port /dev/ttyTHS1

The J1+J6 swing uses arm.move_joints (streamed, so the two co-rotate and the
heading stays fixed throughout). Each coordinated lift/descent temporarily drops
to normal (blended, multi-joint) mode via arm.set_single_joint(0). The
disabled-by-default nudge / trigger-box paths keep the original staccato
behaviour.

Everything you tune is a CONSTANT below. The cube coordinates and
PICK_ORIENTATION_DEG are PLACEHOLDERS -- measure them on your arm first.
"""

from __future__ import annotations

import os as _os, sys as _sys
# scripts/Profiles/ is two levels below the package root -> three dirname() calls
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import argparse
import threading
import time

import numpy as np

from armik import Arm, ArmError, config, pose_coords

# ===========================================================================
# CONSTANTS
# ===========================================================================

# -- run / connection -----------------------------------------------------------
HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
SETTLE_S = 0.3
PREFLIGHT = True

# -- cubes (MEASURE AND REPLACE) ----------------------------------------------
# (x, y, z) CM, at the GRIPPER TIP, base frame, z from the table.
CUBES_INITIAL_POINTS = [          # a row on the pick side, LISTED LEFT -> RIGHT
    (15.0, 10.0, 0.0),
    (15.0, 13.0, 0.0),
    (15.0, 16.0, 0.0),
    (15.0, 19.0, 0.0),
]
CUBES_TARGET_POINTS = [           # the drop row; cube picked k-th goes to slot k
    (15.0, -10.0, 0.0),
    (15.0, -13.0, 0.0),
    (15.0, -16.0, 0.0),
    (15.0, -19.0, 0.0),
]

# Gripper orientation (rx, ry, rz DEG) held for EVERY move. CALIBRATION: jog to
# gripper-straight-down, read arm.get_coords()[3:] (current TOOL frame).
PICK_ORIENTATION_DEG = (180.0, 0.0, -45.0)

APPROACH_HEIGHT_CM = 6.0          # hover height above a cube (preflight checks only)

# -- robotic motion --------------------------------------------------------------
JOINT_SPEED_DPS = 60.0           # fast, CONSTANT deg/s for the J1 swing + homing
                                # (clamped per joint to config.MAX_JOINT_SPEED_DPS)
SEG_PLAN_S = 4.0              # planning-only per-waypoint duration for send_path (nudge path)
DELAY_BETWEEN_JOINTS_S = 0.10  # -> config.SINGLE_JOINT_DELAY            (armik default 0.15)
DELAY_BETWEEN_POINTS_S = 0.30  # -> config.SINGLE_JOINT_DELAY_BETWEEN_POINTS (default 2.0!)

# -- v2 coordinated grab/place -------------------------------------------------
CARRY_HEIGHT_CM = 12.0        # height the end-effector is lifted/held at during the J1 transit
DESCENT_SPEED_CM_S = 4.0      # cm/s for the coordinated lift and grab/place descent

# grab order -- deterministic. Arrange CUBES_INITIAL_POINTS left->right, or set
# explicit indices here.
PICK_ORDER = list(range(len(CUBES_INITIAL_POINTS)))

# -- scripted nudge ("defective cube") ------------------------------------------
NUDGE_CYCLE = -1                # EVEN (reach) cycle index whose cube is nudged; -1 = off
NUDGE_OFFSET_CM = (3.0, 0.0, 0.0)   # where the nudged cube ends up
NUDGE_POINT_TILT_DEG = 25.0    # pitch the gripper this far off straight-down, toward the cube
NUDGE_AIM_SPEED_DPS = 40.0     # deg/s for the "point at it" move
NUDGE_LOOK_S = 1.5            # hold the "regarding it" pose before moving on

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 110.0          # 0 = closed .. config.MAX_GRIPPER_DEG = full open
GRIP_CLOSED_DEG = 25.0         # tune to the cube width
GRIP_SPEED = 90  #config.GRIPPER_DEFAULT_SPEED
GRIP_SETTLE_S = 0.35          # quiet time after a gripper command: it must LAND and the jaws
                             # start moving. Tunable down to GRIP_MIN_GAP_S, not below.
GRIP_MIN_GAP_S = 1.0         # hard floor -- pymycobot drops a gripper command with no quiet gap
REACH_TOL_CM = 3.0            # has_reached_* tolerance, per axis

# -- gripper trigger boxes (optional, experimental in single-joint mode) --------
# [[(cx,cy,cz), (l,w,h), cycle_n], ...] -- on cycle cycle_n the gripper fires the
# moment the tip enters this box (that cycle runs on a background thread, polled
# live). Empty -> gripper fires at the cycle end. Single-joint timing is
# unpredictable so the fire point is position-based, not time-based.
TRIGGER_BOXES = []


# ===========================================================================
# GEOMETRY HELPERS
# ===========================================================================

def _approach(p):
    return (p[0], p[1], p[2] + APPROACH_HEIGHT_CM)


def get_path(origin_point, target_point):
    """Straight waypoints for one move: traverse over the target, then descend
    onto it. `origin_point` is unused. Kept for the nudge path + the RViz marker."""
    return [tuple(map(float, _approach(target_point))), tuple(map(float, target_point))]


def get_durations(pts):
    """Planning-only per-waypoint durations."""
    return [SEG_PLAN_S] * len(pts)


# ===========================================================================
# STATE CHECKS
# ===========================================================================

def current_pos(arm):
    return tuple(float(v) for v in arm.get_coords()[:3])


def _within(xyz, centre, half_extents):
    d = np.abs(np.asarray(xyz, float) - np.asarray(centre, float))
    return bool(np.all(d <= np.asarray(half_extents, float)))


def is_in_trigger_box(end_effector_coords, cycle_n):
    for entry in TRIGGER_BOXES:
        centre, dims, cyc = entry
        if cyc == cycle_n and _within(end_effector_coords, centre,
                                      np.asarray(dims, float) / 2.0):
            return True
    return False


def _has_trigger(cycle_n):
    return any(cyc == cycle_n for _, _, cyc in TRIGGER_BOXES)


def has_reached_cube(end_effector_coords, cube_xyz):
    return _within(end_effector_coords, cube_xyz, (REACH_TOL_CM,) * 3)


def has_reached_target(end_effector_coords, target_xyz):
    return _within(end_effector_coords, target_xyz, (REACH_TOL_CM,) * 3)


# ===========================================================================
# MOTION
# ===========================================================================

def go_home(arm):
    """Robotic homing -- drive J1..J6 to their HOME values ONE AT A TIME."""
    for j in range(1, config.DOF + 1):
        tgt = float(HOME[j - 1])
        cur = float(arm.get_angles()[j - 1])
        if abs(tgt - cur) <= config.SINGLE_JOINT_TOL_DEG:
            continue
        try:
            arm._drive_joint(j, tgt, JOINT_SPEED_DPS, "home")
        except ArmError as exc:
            print(f"  homing: J{j} would not move -- {exc}")
            return
        time.sleep(config.SINGLE_JOINT_DELAY)


def _coord_move(arm, x, y, z, label, speed=None):
    """One COORDINATED (multi-joint, blended) Cartesian move, end-effector held
    at PICK_ORIENTATION_DEG. Temporarily leaves single-joint mode so the move
    goes through armik's streamed executor. Treats 'already at' as success."""
    rx, ry, rz = PICK_ORIENTATION_DEG
    arm.set_single_joint(0)
    try:
        ok = arm.send_coords(x=float(x), y=float(y), z=float(z),
                             rx=rx, ry=ry, rz=rz,
                             speed=speed or DESCENT_SPEED_CM_S)
    finally:
        arm.set_single_joint(1)
    if not ok:
        if "already at" in (arm.last_error or "").lower():
            return True
        print(f"  {label}: send_coords REFUSED -- {arm.last_error}")
        return False
    pl = arm.last_plan
    if pl is not None:
        print(f"  {label}: {pl.path_length_cm:.1f} cm, {pl.duration_s:.2f} s")
    return True


def _swing(arm, target_xyz, label):
    """Aim the arm at target_xyz's azimuth by rotating J1 and J6 only -- J1 to
    the IK azimuth, J6 to the IK wrist angle that holds the gripper's world
    heading (PICK_ORIENTATION_DEG is world-fixed, so the IK J6 counter-rotates
    with J1). J2..J5 stay put; the coordinated descent fixes them. move_joints
    runs the streamed executor, so J1 and J6 turn together and the heading stays
    fixed throughout, not just at the ends."""
    rx, ry, rz = PICK_ORIENTATION_DEG
    tx, ty = float(target_xyz[0]), float(target_xyz[1])
    pl = arm.plan_coords(x=tx, y=ty, z=CARRY_HEIGHT_CM, rx=rx, ry=ry, rz=rz,
                         speed=config.DEFAULT_SPEED_CM_S)
    if not pl.ok:
        if "already at" in (pl.error or "").lower():
            return True
        print(f"  {label}: J1 plan REFUSED -- {pl.error}")
        return False
    q_cur = [float(v) for v in arm.get_angles()]
    q_ik = [float(v) for v in pl.segment_q[-1]]
    dj1 = q_ik[0] - q_cur[0]
    dj6 = q_ik[5] - q_cur[5]
    tol = config.SINGLE_JOINT_TOL_DEG
    if abs(dj1) <= tol and abs(dj6) <= tol:
        return True
    q_target = list(q_cur)
    q_target[0] = q_ik[0]
    q_target[5] = q_ik[5]
    dur = max(abs(dj1), abs(dj6)) / JOINT_SPEED_DPS
    print(f"  {label}: J1 {q_cur[0]:.1f}->{q_ik[0]:.1f}, J6 {q_cur[5]:.1f}->{q_ik[5]:.1f} deg")
    if not arm.move_joints(q_target, duration=max(dur, 0.1)):
        print(f"  {label}: move_joints REFUSED -- {arm.last_error}")
        return False
    return True


def _lift_swing_descend(arm, target_xyz, label):
    """v2 reach / carry: coordinated lift to CARRY_HEIGHT_CM -> J1 swing (with
    J6 compensation) to target_xyz's azimuth -> coordinated descent onto
    target_xyz, end-effector held straight down throughout."""
    here = current_pos(arm)
    if not _coord_move(arm, here[0], here[1], CARRY_HEIGHT_CM, f"{label}: lift"):
        return False
    if not _swing(arm, target_xyz, f"{label}: swing"):
        return False
    return _coord_move(arm, target_xyz[0], target_xyz[1], target_xyz[2],
                       f"{label}: descend")


def _fire_gripper(arm, deg):
    """Send the gripper command twice with a tiny gap -- pymycobot drops a
    gripper packet that is not followed by a short quiet window."""
    ok = arm.send_gripper(deg, speed=GRIP_SPEED)
    time.sleep(0.06)
    arm.send_gripper(deg, speed=GRIP_SPEED)
    return bool(ok)


def _grip(arm, deg, label):
    print(f"  gripper -> {deg:.0f} deg ({label})")
    if not _fire_gripper(arm, deg):
        print(f"  send_gripper REFUSED -- {arm.last_error}")
        return False
    if GRIP_SETTLE_S < GRIP_MIN_GAP_S:
        print(f"  (GRIP_SETTLE_S {GRIP_SETTLE_S}s < floor {GRIP_MIN_GAP_S}s -- using the floor)")
    time.sleep(max(max(GRIP_SETTLE_S, GRIP_MIN_GAP_S) - 0.06, 0.0))
    return True


def _staccato_kw(arm, pts):
    """send_path kwargs for a joint-by-joint move through `pts`, with a live
    lift waypoint prepended. Used only by the nudge / trigger paths in v2."""
    sent = [_approach(current_pos(arm))] + [tuple(map(float, p)) for p in pts]
    rx, ry, rz = PICK_ORIENTATION_DEG
    return sent, dict(
        x=[p[0] for p in sent], y=[p[1] for p in sent], z=[p[2] for p in sent],
        rx=rx, ry=ry, rz=rz, speed=JOINT_SPEED_DPS, durations=[SEG_PLAN_S] * len(sent),
    )


def _send_staccato(arm, pts, label):
    """Blocking one-joint-at-a-time move (armik J1..J6). Used by run_nudge."""
    sent, kw = _staccato_kw(arm, pts)
    r = arm.send_path(**kw)
    if not r:
        print(f"  {label}: send_path REFUSED -- {arm.last_error}")
        return False
    ex = arm.last_execution
    print(f"  {label}: {len(sent)} waypoints, {ex.duration_s:.1f}s, {ex.setpoints} joint moves")
    return True


class _PathThread(threading.Thread):
    def __init__(self, arm, kw):
        super().__init__(daemon=True)
        self.arm, self.kw = arm, kw
        self.result, self.exc = None, None

    def run(self):
        try:
            self.result = self.arm.send_path(**self.kw)
        except BaseException as exc:                       # noqa: BLE001
            self.exc, self.result = exc, 0


def _send_staccato_with_trigger(arm, pts, cycle_n, grip_deg, label):
    """Run the move on a background thread; fire the gripper the first time the
    tip is inside an active trigger box for this cycle (position-polled).
    Still uses armik's J1..J6 order."""
    _sent, kw = _staccato_kw(arm, pts)
    th = _PathThread(arm, kw)
    th.start()
    fired = False
    while th.is_alive():
        if not fired and th.exc is None:
            try:
                if is_in_trigger_box(current_pos(arm), cycle_n):
                    print(f"  {label}: trigger box entered -> gripper {grip_deg:.0f}")
                    _fire_gripper(arm, grip_deg)
                    fired = True
            except Exception:
                pass
        time.sleep(0.05)
    th.join()
    if th.exc is not None:
        raise th.exc
    if not th.result:
        print(f"  {label}: send_path (threaded) REFUSED -- {arm.last_error}")
        return False
    if not fired:
        print(f"  {label}: box not entered -- firing gripper {grip_deg:.0f} now")
        _fire_gripper(arm, grip_deg)
    time.sleep(max(GRIP_SETTLE_S, GRIP_MIN_GAP_S))
    return True


def run_nudge(arm, seg, ci, segments):
    """Scripted 'defective cube': reach, the cube moves, point the gripper at
    its new spot, hold, then IGNORE it (its carry is skipped)."""
    rx0, ry0, rz0 = PICK_ORIENTATION_DEG
    print(f"  NUDGE cycle {ci}: reaching for cube #{seg['k']+1} ...")
    if not _send_staccato(arm, [_approach(seg["target"])], "  nudge reach (hover)"):
        return False

    new_cube = tuple(float(c + o) for c, o in zip(seg["target"], NUDGE_OFFSET_CM))
    aim = _approach(new_cube)
    print(f"  cube moved to {tuple(round(v, 1) for v in new_cube)} -- "
          f"pointing the gripper at it, then ignoring it")
    r = arm.send_path(x=[float(aim[0])], y=[float(aim[1])], z=[float(aim[2])],
                      rx=rx0, ry=ry0 + NUDGE_POINT_TILT_DEG, rz=rz0,
                      speed=NUDGE_AIM_SPEED_DPS, durations=[SEG_PLAN_S])
    if not r:
        print(f"  aim REFUSED -- {arm.last_error}")
        return False
    time.sleep(NUDGE_LOOK_S)

    nxt = ci + 1
    if nxt < len(segments) and segments[nxt]["kind"] == "carry":
        segments[nxt]["kind"] = "skip"
    return True


# ===========================================================================
# PREFLIGHT
# ===========================================================================

def preflight(arm, segments):
    rx, ry, rz = PICK_ORIENTATION_DEG
    print("\n--- preflight: planning every cube point (no motion) ---")
    checks = []
    for i, (s, t) in enumerate(zip(CUBES_INITIAL_POINTS, CUBES_TARGET_POINTS)):
        checks += [(f"init{i+1}", s), (f"init{i+1}^", _approach(s)),
                   (f"tgt{i+1}", t), (f"tgt{i+1}^", _approach(t))]
    if 0 <= NUDGE_CYCLE < len(segments):
        nc = tuple(c + o for c, o in zip(segments[NUDGE_CYCLE]["target"], NUDGE_OFFSET_CM))
        checks.append(("nudge^", _approach(nc)))
    bad = 0
    for name, (x, y, z) in checks:
        pl = arm.plan_coords(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                             speed=config.DEFAULT_SPEED_CM_S)
        err = (pl.error or "").lower()
        if pl.ok:
            print(f"  OK  {name:10s} ({x:5.1f},{y:6.1f},{z:4.1f})  "
                  f"peak {pl.peak_joint_dps:.0f} deg/s")
        elif "already at" in err:
            print(f"  OK  {name:10s} ({x:5.1f},{y:6.1f},{z:4.1f})  (already there)")
        else:
            print(f"  BAD {name:10s} ({x:5.1f},{y:6.1f},{z:4.1f})  {pl.error}")
            bad += 1
    print(f"--- preflight: {len(checks) - bad}/{len(checks)} reachable ---")
    return bad == 0


# ===========================================================================
# MAIN
# ===========================================================================

def _build_segments(order, home_tip):
    segs = []
    prev_target = home_tip
    for k in order:
        segs.append({"kind": "reach", "origin": prev_target,
                     "target": CUBES_INITIAL_POINTS[k], "k": int(k)})
        segs.append({"kind": "carry", "origin": CUBES_INITIAL_POINTS[k],
                     "target": CUBES_TARGET_POINTS[k], "k": int(k)})
        prev_target = CUBES_TARGET_POINTS[k]
    return segs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--rviz", action="store_true",
                    help="mock only: stream the simulated pose + precomputed path to RViz2 "
                         "(needs rclpy; run inside the Study-docker container)")
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")
    args = ap.parse_args()

    if args.rviz and not args.mock:
        print("--rviz is mock-only; ignoring it (add --mock to visualize).")
        args.rviz = False

    if len(CUBES_INITIAL_POINTS) != len(CUBES_TARGET_POINTS):
        print("CUBES_INITIAL_POINTS and CUBES_TARGET_POINTS must be the same length.")
        return 1
    if NUDGE_CYCLE >= 0 and NUDGE_CYCLE % 2 != 0:
        print(f"NUDGE_CYCLE must be an EVEN (reach) cycle, got {NUDGE_CYCLE}.")
        return 1

    if not args.mock and not args.yes:
        print("This will move the robot arm and actuate the gripper. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    config.SINGLE_JOINT_DELAY = DELAY_BETWEEN_JOINTS_S
    config.SINGLE_JOINT_DELAY_BETWEEN_POINTS = DELAY_BETWEEN_POINTS_S

    order = list(PICK_ORDER)
    p_home = pose_coords(HOME)                       # mm/deg (Z_RELATIVE_TO_JOINT1 assumed False)
    home_tip = (p_home[0] / 10.0, p_home[1] / 10.0, p_home[2] / 10.0)

    print(f"pick order (cube indices): {order}")
    segments = _build_segments(order, home_tip)
    if 0 <= NUDGE_CYCLE < len(segments):
        nk = segments[NUDGE_CYCLE]["k"]
        print(f"NUDGE on cycle {NUDGE_CYCLE}: cube #{nk + 1} at {CUBES_INITIAL_POINTS[nk]} "
              f"-- nudge THAT cube as the arm nears it; its carry is skipped")

    paths = [get_path(s["origin"], s["target"]) for s in segments]

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)

    bridge = None
    if args.rviz:
        from _rviz_bridge import RvizBridge
        # polyline the EE actually traces: from HOME, then per segment
        # lift straight up to CARRY_HEIGHT_CM -> traverse at that height ->
        # descend onto the target (mirrors _lift_swing_descend)
        flat_path = [tuple(map(float, home_tip))]
        for s in segments:
            ox, oy, _oz = s["origin"]
            tx, ty, tz = s["target"]
            flat_path += [(float(ox), float(oy), float(CARRY_HEIGHT_CM)),
                          (float(tx), float(ty), float(CARRY_HEIGHT_CM)),
                          (float(tx), float(ty), float(tz))]
        cube_pts = [tuple(map(float, c)) for c in CUBES_INITIAL_POINTS]
        try:
            bridge = RvizBridge(arm.get_angles, path_xyz_cm=flat_path,
                                cube_points=cube_pts, tip_source=arm.get_coords,
                                gripper_source=arm.get_gripper_value)
            bridge.start()
            print("RViz bridge up: publishing /joint_states + /visualization_marker")
        except RuntimeError as exc:
            print(exc)
            bridge = None

    try:
        if not arm.conn.is_power_on():
            print("powering on...")
            arm.conn.power_on()
            time.sleep(1.5)

        arm.set_single_joint(1)
        print("homing (one joint at a time)...")
        go_home(arm)
        time.sleep(SETTLE_S)
        print(f"start pose (tip, cm/deg): {[round(v, 2) for v in arm.get_coords()]}")

        if PREFLIGHT and not preflight(arm, segments):
            print("\npreflight failed -- fix the cube coordinates or PICK_ORIENTATION_DEG. "
                  "Nothing moved.")
            go_home(arm)
            return 1

        if not _grip(arm, GRIP_OPEN_DEG, "open before first pick"):
            return 1

        for ci, seg in enumerate(segments):
            kind = seg["kind"]

            if kind == "skip":
                print(f"\n=== cycle {ci}  cube #{seg['k'] + 1} -- DEFECTIVE, carry skipped ===")
                continue

            grip_deg = GRIP_CLOSED_DEG if kind == "reach" else GRIP_OPEN_DEG
            label = "reach & grasp" if kind == "reach" else "carry & place"
            print(f"\n=== cycle {ci}/{len(segments) - 1}  {label}  cube #{seg['k'] + 1}  "
                  f"-> {tuple(round(v, 1) for v in seg['target'])} ===")

            if kind == "reach" and ci == NUDGE_CYCLE:
                if not run_nudge(arm, seg, ci, segments):
                    print("\naborting run."); go_home(arm); return 1
                continue

            pts = paths[ci]
            if _has_trigger(ci):
                if not _send_staccato_with_trigger(arm, pts, ci, grip_deg, label):
                    print("\naborting run."); go_home(arm); return 1
                continue

            if not _lift_swing_descend(arm, seg["target"], label):
                print("\naborting run."); go_home(arm); return 1

            cur = current_pos(arm)
            reached = (has_reached_cube(cur, seg["target"]) if kind == "reach"
                       else has_reached_target(cur, seg["target"]))
            if reached:
                if not _grip(arm, grip_deg,
                             "close on cube" if kind == "reach" else "release cube"):
                    return 1
            else:
                print(f"  !! tip at {tuple(round(v, 2) for v in cur)}, expected "
                      f"{tuple(round(v, 1) for v in seg['target'])} +/- {REACH_TOL_CM} cm "
                      f"-- gripper NOT fired")

        print("\ndone. homing...")
        go_home(arm)
        return 0

    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping the arm.")
        arm.stop()
        return 1
    finally:
        if bridge is not None:
            bridge.stop()
        try:
            arm.set_single_joint(0)
        except Exception:
            pass
        arm.close()


if __name__ == "__main__":
    _sys.exit(main())
