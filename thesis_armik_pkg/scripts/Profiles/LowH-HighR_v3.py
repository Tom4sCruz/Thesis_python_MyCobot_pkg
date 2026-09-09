#!/usr/bin/env python3
"""
MOVEMENT PROFILE: Low-human / High-robot  (v3 -- J2-only raise)
==============================================================

Move cubes from one side of the frame to the other, the way a classic
industrial robot would:

  * ONE JOINT AT A TIME, with a hard stop and a short pause at each. Each cube
    is a fixed SWING / DESCEND / LIFT choreography (no hover-then-lower double
    move): J1 swings the arm to the cube's azimuth; J6..J2 descend onto it and
    grab; J2 ALONE lifts the shoulder; J1 swings to the target azimuth; J6..J2
    descend and drop; J2 alone lifts; J1 swings to the next cube. Unlike
    LowH-HighR.py (which raises with J2..J4), the raise here is a plain J2
    rotation, so between cubes the gripper stays folded in its grab pose and
    points forward-down, not straight down -- the NEXT descent's J6..J2 step
    restores the straight-down pose at the target. The between-cube transit
    turns J1 ONLY. End-of-run homing is J2..J4, then J1, then the wrist. The
    (disabled by default) nudge / trigger paths still use armik's fixed J1..J6
    executor;
  * a FAST CONSTANT joint speed (JOINT_SPEED_DPS), sharp corners, no blending;
  * exactly the SAME trajectory every run -- nothing is randomised (unless the
    JERK dials are raised: a deliberate tremor + uneven pace for a smooth-vs-
    jerky comparison; JERK = 0 -> identical to a clean run);
  * cubes grabbed in a fixed order (PICK_ORDER, left -> right) and placed into
    CUBES_TARGET_POINTS[k] in that order -- they end in a row;
  * NO overshoot -- each joint is driven until it is within
    config.SINGLE_JOINT_TOL_DEG of target, then stopped;
  * one cube is NUDGED mid-run (scripted): the arm reaches for it, the cube
    moves, the arm POINTS ITS GRIPPER at the new spot, holds a beat, then
    IGNORES it -- that cube's carry is skipped and it is left behind
    ("defective"). Fully scripted; the arm has no sensors.

    python3 scripts/Profiles/LowH-HighR.py --mock --yes      # no hardware
    python3 scripts/Profiles/LowH-HighR.py --port /dev/ttyTHS1

Structure mirrors HighH-LowR.py (constants -> helpers -> precompute -> execute).
armik's single-joint executor prints a line per servo poll -- the console is
chatty; that is expected.

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

from armik import Arm, ArmError, config, jerk, kinematics, pose_coords

# ===========================================================================
# CONSTANTS
# ===========================================================================

# -- run / connection -----------------------------------------------------------
HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
SETTLE_S = 0.3
PREFLIGHT = True

# -- cubes (MEASURE AND REPLACE) ----------------------------------------------
# (x, y, z) CM, at the GRIPPER TIP, base frame, z from the table.

Z_CUBE_COORD = -4.0

CUBES_INITIAL_POINTS = [          # a row on the pick side, LISTED LEFT -> RIGHT
    (15.0, 10.0, Z_CUBE_COORD),
    (15.0, 13.0, Z_CUBE_COORD),
    (15.0, 16.0, Z_CUBE_COORD),
    (15.0, 19.0, Z_CUBE_COORD),

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

APPROACH_HEIGHT_CM = 6.0          # hover height above a cube before descending

# -- robotic motion --------------------------------------------------------------
JOINT_SPEED_DPS = 60.0           # fast, CONSTANT deg/s for every single-joint move and
                                # for homing (clamped per joint to config.MAX_JOINT_SPEED_DPS)
SEG_PLAN_S = 4.0               # generous per-waypoint duration handed to send_path ONLY so
                              # its blended pre-check passes; single-joint execution ignores it
DELAY_BETWEEN_JOINTS_S = 0.10  # -> config.SINGLE_JOINT_DELAY            (armik default 0.15)
DELAY_BETWEEN_POINTS_S = 0.30  # -> config.SINGLE_JOINT_DELAY_BETWEEN_POINTS (default 2.0!)
# --rviz ONLY (mock, no hardware): pace the single-joint motion so RViz can show
# each joint rotating fully to a stop before the next one starts. Physical runs
# (no --rviz) keep the fast DELAY_* values above. Keep RVIZ_JOINT_MOVE_S larger
# than _rviz_bridge.VIZ_SEQ_MOVE_S; change both together to re-speed the viz.
RVIZ_JOINT_MOVE_S = 1.0
RVIZ_BETWEEN_POINTS_S = 0.5

# grab order -- deterministic. Arrange CUBES_INITIAL_POINTS left->right, or set
# explicit indices here.
PICK_ORDER = list(range(len(CUBES_INITIAL_POINTS)))

# -- phase joint order (1-based ids) ---------------------------------------------
# The normal cycle is an explicit swing / descend / lift choreography (see the
# MOTION section). Each phase steps ONLY its listed joints, one servo at a time.
DESCEND_JOINT_ORDER = (6, 5, 4, 3, 2)   # grab / drop: wrist J6..J3 first, shoulder
                                        # J2 LAST -> a clean vertical drop. J1 is
                                        # already aligned by the preceding swing.
LIFT_JOINT_ORDER    = (2,)              # v3: raise turns J2 ONLY (LowH-HighR.py uses
                                        # (2, 3, 4)) -- a plain shoulder rotation; J3..J6
                                        # stay in the grab pose until the next descent
SWING_JOINT         = 1                 # transit between cubes turns J1 ONLY -- the
                                        # gripper heading rides along and is corrected
                                        # by the next descent's J6..J2 step

# _send_ordered() (kept only for the disabled-by-default nudge path) still uses
# armik's fixed J1..J6 executor order, picking one of these per waypoint.
LOWER_JOINT_ORDER = (1, 6, 5, 4, 3, 2)   # descending onto a cube/target
RAISE_JOINT_ORDER = (1, 2, 3, 4, 5, 6)   # lifting away / traversing

# -- deliberate jitter (jerk) --------------------------------------------------
# All four default to 0 / None -> armik.jerk.JerkInjector is INERT and the motion
# is byte-for-byte identical to a clean run. Raise JERK for a visible tremor +
# uneven pace; set JERK_SEED to an int to replay a run exactly.
JERK = 0.0                       # -> arm.jerk             (0 smooth; ~1-3 subtle; ~5-10 violent)
JERK_RANDOM_TWITCH = 0.0         # -> arm.random_twitch    (flinch probability per joint step [0,1])
JERK_TWITCH_INTENSITY_DEG = 0.0  # -> arm.twitch_intensity (peak flinch amplitude, deg)
JERK_SEED = None                 # -> arm.jerk_seed        (None = fresh each run; int = repeatable)

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
GRIP_MIN_GAP_S = 0.2         # hard floor -- pymycobot drops a gripper command with no quiet gap
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
    onto it. `origin_point` is unused -- _send_staccato() prepends a live lift
    waypoint, so nothing depends on the precomputed start."""
    return [tuple(map(float, _approach(target_point))), tuple(map(float, target_point))]


def get_durations(pts):
    """Planning-only per-waypoint durations (single-joint execution ignores them)."""
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
    """Robotic homing -- drive J1..J6 to their HOME values ONE AT A TIME.
    Used at start-of-run and on an abort (order does not matter for safety); the
    normal end-of-run homing follows the J2..J4 / J1 / wrist rule in main()."""
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


def _make_run_injector():
    """One JerkInjector for the whole hand-rolled run (mirrors Arm._make_jerk).
    A single rng stream -> a given JERK_SEED reproduces the run exactly, and the
    AR(1) tremor state carries across phases like a real tremor. INERT (no rng
    draw, zero offsets, unit speed factor) unless JERK > 0 or both
    JERK_RANDOM_TWITCH and JERK_TWITCH_INTENSITY_DEG are non-zero."""
    rng = np.random.default_rng(JERK_SEED)
    return jerk.JerkInjector(JERK, JERK_RANDOM_TWITCH, JERK_TWITCH_INTENSITY_DEG,
                             config.DOF, rng)


def _plan_pose_q(arm, x, y, z, label):
    """IK-plan a straight-down pose (PICK_ORIENTATION_DEG) at (x, y, z) cm and
    return the goal joint vector (list[6], deg), or None. No motion. Uses
    plan_path with a generous SEG_PLAN_S per-waypoint duration purely so the
    blended-execution pre-check passes -- we only read segment_q, we never stream
    this plan (the actual motion is one servo at a time in _step_joints).
    'already at' -> current angles."""
    rx, ry, rz = PICK_ORIENTATION_DEG
    pl = arm.plan_path(x=[float(x)], y=[float(y)], z=[float(z)],
                       rx=rx, ry=ry, rz=rz, durations=[SEG_PLAN_S])
    if pl.ok and pl.segment_q is not None:
        return [float(v) for v in pl.segment_q[-1]]
    if "already at" in (pl.error or "").lower():
        return [float(v) for v in arm.get_angles()]
    print(f"  {label}: plan REFUSED -- {pl.error}")
    return None


def _step_joints(arm, order, q_goal, inj, is_last, label):
    """Drive the joints in `order` (1-based) to their `q_goal` values, one servo
    at a time: skip any already within SINGLE_JOINT_TOL_DEG; else
    arm._drive_joint (send at JOINT_SPEED_DPS, block until arrived, re-send on a
    stall), then SINGLE_JOINT_DELAY. A trailing
    SINGLE_JOINT_DELAY_BETWEEN_POINTS unless `is_last`. Re-reads arm.get_angles()
    on entry, so drift from a J1-only swing never accumulates. Mirrors
    Arm._execute_single_joint's inner loop, including the deliberate-jitter path
    (angle + speed perturbed for every step EXCEPT when is_last, and only when
    the injector is active). Returns bool."""
    soft = config.joint_limits_array()
    cur = np.array(arm.get_angles(), dtype=float)
    moved = []
    for j in order:
        a_goal = float(q_goal[j - 1])
        if abs(a_goal - float(cur[j - 1])) <= config.SINGLE_JOINT_TOL_DEG:
            continue
        speed = JOINT_SPEED_DPS
        cmd = a_goal
        if inj.active and not is_last:
            move_time = abs(a_goal - float(cur[j - 1])) / max(speed, 1e-6)
            jittered = float(np.clip(a_goal + float(inj.offsets(move_time)[j - 1]),
                                     soft[j - 1, 0], soft[j - 1, 1]))
            cand = cur.copy()
            cand[j - 1] = jittered
            if kinematics.check_workspace_bounds(
                    kinematics.forward_kinematics(cand)[:3, 3]) is None:
                cmd = jittered
            speed = min(max(speed * inj.speed_factor(), 1.0),
                        config.MAX_JOINT_SPEED_DPS[j - 1])
        cand = cur.copy()
        cand[j - 1] = cmd
        be = kinematics.check_workspace_bounds(
            kinematics.forward_kinematics(cand)[:3, 3])
        if be is not None:
            print(f"  {label}: J{j} refused -- {be}")
            return False
        try:
            arm._drive_joint(j, cmd, speed, label)
        except ArmError as exc:
            print(f"  {label}: {exc}")
            return False
        cur[j - 1] = cmd
        moved.append(j)
        time.sleep(config.SINGLE_JOINT_DELAY)
    if not is_last:
        time.sleep(config.SINGLE_JOINT_DELAY_BETWEEN_POINTS)
    print(f"  {label}: [{'/'.join(f'J{j}' for j in moved) or '-'}]")
    return True


def _descend(arm, point, inj, label, is_last=False):
    """SWING already done -- step J6..J2 down onto `point` (grab / drop)."""
    q = _plan_pose_q(arm, point[0], point[1], point[2], label)
    if q is None:
        return False
    return _step_joints(arm, DESCEND_JOINT_ORDER, q, inj, is_last, label)


def _lift(arm, point, inj, label, is_last=False):
    """Step J2 only up toward its value in the straight-down 'above' pose over
    `point` (v3). J3..J6 keep their post-grab angles, so this is a plain shoulder
    lift, not a return to a straight-down pose."""
    ax, ay, az = _approach(point)
    q = _plan_pose_q(arm, ax, ay, az, label)
    if q is None:
        return False
    return _step_joints(arm, LIFT_JOINT_ORDER, q, inj, is_last, label)


def _swing_j1(arm, point, inj, label, is_last=False):
    """Turn J1 ONLY to `point`'s azimuth, arm held at the 'above' height. The
    gripper heading rides along; the next _descend's J6..J2 step corrects it."""
    ax, ay, az = _approach(point)
    q = _plan_pose_q(arm, ax, ay, az, label)
    if q is None:
        return False
    return _step_joints(arm, (SWING_JOINT,), q, inj, is_last, label)


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
    lift waypoint prepended."""
    sent = [_approach(current_pos(arm))] + [tuple(map(float, p)) for p in pts]
    rx, ry, rz = PICK_ORIENTATION_DEG
    return sent, dict(
        x=[p[0] for p in sent], y=[p[1] for p in sent], z=[p[2] for p in sent],
        rx=rx, ry=ry, rz=rz, speed=JOINT_SPEED_DPS, durations=[SEG_PLAN_S] * len(sent),
    )


def _send_staccato(arm, pts, label):
    """Blocking one-joint-at-a-time move: lift -> traverse -> ... -> descend.
    Uses armik's fixed J1..J6 order (kept for the trigger-box path)."""
    sent, kw = _staccato_kw(arm, pts)
    r = arm.send_path(**kw)
    if not r:
        print(f"  {label}: send_path REFUSED -- {arm.last_error}")
        return False
    ex = arm.last_execution
    print(f"  {label}: {len(sent)} waypoints, {ex.duration_s:.1f}s, {ex.setpoints} joint moves")
    return True


def _send_ordered(arm, pts, label):
    """DEPRECATED for the normal cycle (replaced by _swing_j1 / _descend / _lift);
    kept only for run_nudge.

    One-joint-at-a-time move through `pts`, driving the joints in
    LOWER_JOINT_ORDER for waypoints that go DOWN and RAISE_JOINT_ORDER for
    waypoints that go up / across, instead of armik's fixed J1..J6. Mirrors
    armik._execute_single_joint: IK-plan every waypoint (no motion), then step
    each joint to its planned angle -- skipping joints already within
    SINGLE_JOINT_TOL_DEG -- one servo at a time, honouring the same delays."""
    sent, kw = _staccato_kw(arm, pts)
    pl = arm.plan_path(**kw)                          # IK only, no motion
    if not pl.ok:
        print(f"  {label}: plan_path REFUSED -- {pl.error}")
        return False
    seg_q = pl.segment_q                              # [q0, wp1, wp2, ...] degrees
    cur_tip_z = sent[0][2] - APPROACH_HEIGHT_CM       # tip z before the prepended lift
    orders = []
    for wi in range(1, len(seg_q)):
        z_now = sent[wi - 1][2]
        z_prev = sent[wi - 2][2] if wi >= 2 else cur_tip_z
        order = LOWER_JOINT_ORDER if z_now < z_prev - 0.5 else RAISE_JOINT_ORDER
        orders.append("lower" if order is LOWER_JOINT_ORDER else "raise")
        target = [float(v) for v in seg_q[wi]]
        cur = [float(v) for v in arm.get_angles()]
        for j in order:
            if abs(target[j - 1] - cur[j - 1]) <= config.SINGLE_JOINT_TOL_DEG:
                continue
            cand = list(cur)
            cand[j - 1] = target[j - 1]
            be = kinematics.check_workspace_bounds(
                kinematics.forward_kinematics(np.asarray(cand, float))[:3, 3])
            if be is not None:
                print(f"  {label}: J{j} refused -- {be}")
                return False
            try:
                arm._drive_joint(j, target[j - 1], JOINT_SPEED_DPS, label)
            except ArmError as exc:
                print(f"  {label}: {exc}")
                return False
            cur[j - 1] = target[j - 1]
            time.sleep(config.SINGLE_JOINT_DELAY)
        if wi < len(seg_q) - 1:
            time.sleep(config.SINGLE_JOINT_DELAY_BETWEEN_POINTS)
    print(f"  {label}: {len(sent)} waypoints [{', '.join(orders)}]")
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
    tip is inside an active trigger box for this cycle (position-polled)."""
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
    if not _send_ordered(arm, [_approach(seg["target"])], "  nudge reach (hover)"):
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
    if args.rviz:                       # mock-only: slow so RViz shows one joint at a time
        config.SINGLE_JOINT_DELAY = RVIZ_JOINT_MOVE_S
        config.SINGLE_JOINT_DELAY_BETWEEN_POINTS = RVIZ_BETWEEN_POINTS_S

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
    arm.jerk = JERK
    arm.random_twitch = JERK_RANDOM_TWITCH
    arm.twitch_intensity = JERK_TWITCH_INTENSITY_DEG
    arm.jerk_seed = JERK_SEED
    inj = _make_run_injector()                       # drives the hand-rolled joint stepping
    if inj.active:
        print(f"JERK on: jerk={JERK} twitch={JERK_RANDOM_TWITCH}@{JERK_TWITCH_INTENSITY_DEG}deg "
              f"seed={JERK_SEED}")

    bridge = None
    if args.rviz:
        from _rviz_bridge import RvizBridge
        # polyline the EE actually traces: from HOME, then per segment
        # swing in over the target at hover height -> descend onto it -> lift back
        flat_path = [tuple(map(float, home_tip))]
        for s in segments:
            p = s["target"]
            flat_path += [tuple(map(float, _approach(p))),
                          tuple(map(float, p)),
                          tuple(map(float, _approach(p)))]
        flat_path.append(tuple(map(float, home_tip)))    # end-of-run homing
        cube_pts = [tuple(map(float, c)) for c in CUBES_INITIAL_POINTS]
        try:
            bridge = RvizBridge(arm.get_angles, path_xyz_cm=flat_path,
                                cube_points=cube_pts, tip_source=arm.get_coords,
                                gripper_source=arm.get_gripper_value,
                                sequential_joints=True)
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

            if _has_trigger(ci):
                if not _send_staccato_with_trigger(arm, paths[ci], ci, grip_deg, label):
                    print("\naborting run."); go_home(arm); return 1
                continue

            tgt = seg["target"]

            # SWING: reach cycle 0 aligns J1 to the first cube (choreography step
            # 1); every carry swings J1 to the drop azimuth (step 4). Reach
            # cycles > 0 are already aligned by the previous carry's step-7 swing.
            if kind == "carry" or ci == 0:
                if not _swing_j1(arm, tgt, inj, f"{label}: swing J1"):
                    print("\naborting run."); go_home(arm); return 1

            # DESCEND J6..J2 onto the cube / drop point (steps 2 / 5).
            if not _descend(arm, tgt, inj, f"{label}: descend"):
                print("\naborting run."); go_home(arm); return 1

            cur = current_pos(arm)
            reached = (has_reached_cube(cur, tgt) if kind == "reach"
                       else has_reached_target(cur, tgt))
            if reached:
                if not _grip(arm, grip_deg,
                             "close on cube" if kind == "reach" else "release cube"):
                    return 1
            else:
                print(f"  !! tip at {tuple(round(v, 2) for v in cur)}, expected "
                      f"{tuple(round(v, 1) for v in tgt)} +/- {REACH_TOL_CM} cm "
                      f"-- gripper NOT fired")

            # LIFT (J2 only) -- plain shoulder rotation (steps 3 / 6).
            if not _lift(arm, tgt, inj, f"{label}: lift"):
                print("\naborting run."); go_home(arm); return 1

            # After a carry, swing J1 to the NEXT cube's azimuth (step 7).
            if kind == "carry" and ci + 1 < len(segments):
                if not _swing_j1(arm, segments[ci + 1]["target"], inj,
                                 "swing J1 -> next cube"):
                    print("\naborting run."); go_home(arm); return 1

        # End-of-run homing: J2..J4 to HOME, then J1, then the wrist (J5, J6) --
        # mirrors the descents (shoulder settled before the wrist tidies up).
        print("\ndone. homing (J2-J4, then J1, then wrist)...")
        _step_joints(arm, (2, 3, 4), HOME, inj, True, "home: J2-J4")
        _step_joints(arm, (SWING_JOINT,), HOME, inj, True, "home: J1")
        _step_joints(arm, (5, 6), HOME, inj, True, "home: wrist")
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
