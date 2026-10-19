#!/usr/bin/env python3
"""
MOVEMENT PROFILE: Low-human / Low-robot  (v2 -- a bird pushing cubes with its beak)
==================================================================================

Neither human nor robot: ANIMAL. The arm moves the cubes the way a crow or a
pigeon would -- the gripper stays SHUT the whole run and is used as a BEAK:

  * Phase 1 -- from HOME the arm settles into the look configuration and looks
    at the cubes, then at the target zone, then straight at the first cube
    (the opening of HighH-LowR_begin-looks-both-sides.py.bak);
  * per cube:
      1. it arcs over to a point PUSH_HOVER_CM above the cube's FAR side (the
         side away from the target zone), gazing at the cube on the way, and
         drops its beak down beside the cube;
      2. it PUSHES the cube in a straight line to its target with the tip of
         the closed gripper;
      3. part-way through the push it randomly pulls back a little off the cube
         and comes back to it ("adjusting", ADJUST_*), as a bird -- or a dog
         with a ball -- would, then carries on pushing;
      4. once the cube is in place it lifts its head while LOOKING AT the cube
         it just placed, and turns, on the way, to the next cube (as in
         HighH-LowR.py's return);
  * at the end it looks straight ahead (HighH-LowR.py's nod pose, no nod).

Built on HighH-LowR.py: the same rehearse-then-play tape, look configuration,
head swings, arcs and look-back return are copied from it unchanged (see the
comments there); only the gripper logic, the nudge and the nod are gone and
the beak push is new.

    python3 scripts/Profiles/LowH-LowR_v2.py --mock --yes      # no hardware
    python3 scripts/Profiles/LowH-LowR_v2.py --port /dev/ttyTHS1

LAYOUT
------
A pushed cube sweeps a corridor as wide as itself along its push line, so the
layout must keep every corridor clear of the cubes still waiting and the ones
already placed -- preflight checks this (and that every beak point is
reachable). The tip cannot reach the table close to the base (x < ~16 cm near
y = 0), so the push lanes sit at x >= 16. The default layout: two cubes pushed
side by side in lanes x = 16 and x = 21, then the third, further back, pushed
down the middle once the other two are out of its way.

Everything you tune is a CONSTANT below. The cube coordinates, CUBE_SIZE_CM
and PUSH_Z_CM are PLACEHOLDERS -- measure them on your arm first.
"""

from __future__ import annotations

import os as _os, sys as _sys
# scripts/Profiles/ is two levels below the package root -> three dirname() calls
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import argparse
import math
import threading
import time

import numpy as np

from armik import Arm, Plan, config, pose_coords
from armik.arm import Execution
from _gaze import (gaze_waypoints, gaze_then_level_waypoints, look_at_rpy, ease_to_rpy,
                   wrist_look_angles, wrist_track_angles)
from armik.kinematics import check_joint_limits
from _celebrate import celebration_durations, min_feasible_duration_s

# ===========================================================================
# CONSTANTS
# ===========================================================================

# -- run / connection -----------------------------------------------------------
HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
HOME_MOVE_S = 2.5                  # minimum homing duration (short returns)
HOME_RETURN_DPS = 35.0            # deg/s -- a big return gets proportionally MORE time
SETTLE_S = 0.3
PREFLIGHT = True                  # cheap here (points + corridor geometry, no arcs)
RANDOM_SEED = 1                   # int for a repeatable run, None for fresh each time

# -- cubes (MEASURE AND REPLACE) ----------------------------------------------
# (x, y, z) CM, at the cube's CENTRE at gripper-tip height, base frame.

Z_CUBE_COORD = -4.0
CUBE_SIZE_CM = 3.0                 # cube edge length -- sets where the beak touches the cube
                                   # and the corridor a push sweeps

CUBES_INITIAL_POINTS = [           # on the pick side (+y)
    (16.0, 12.0, Z_CUBE_COORD),
    (21.0, 12.0, Z_CUBE_COORD),
    (18.5, 18.0, Z_CUBE_COORD),
]
CUBES_TARGET_POINTS = [            # cube k is pushed in a straight line to target k
    (16.0, -12.0, Z_CUBE_COORD),
    (21.0, -12.0, Z_CUBE_COORD),
    (18.5, -5.0, Z_CUBE_COORD),
]
PUSH_ORDER = [0, 1, 2]             # which cube is pushed when -- the layout above needs
                                   # cube 3 last (its corridor runs through cubes 1 and 2)

# Gripper orientation (rx, ry, rz DEG) for the push: beak straight down.
PICK_ORIENTATION_DEG = (180.0, 0.0, -45.0)
ORIENT_LOCK = "world"              # see HighH-LowR.py
ORIENT_LOCK_SIGN = 1.0

# -- the beak --------------------------------------------------------------------
GRIP_BEAK_DEG = 0.0                # gripper opening for the WHOLE run (0 = fully shut)
PUSH_Z_CM = Z_CUBE_COORD           # beak-tip height while pushing
BEAK_CLEARANCE_CM = 1.0            # gap between beak and cube face where the beak comes down
PUSH_HOVER_CM = 6.0                # the approach arc ends this far above the push start;
                                   # the beak then drops straight down beside the cube
PECK_SPEED_CM_S = 10.0             # that drop
PECK_EASE_IN = 1.0                 # [0,10]
PECK_EASE_OUT = 3.0                # [0,10]
PECK_WAYPOINTS = 10

PUSH_SPEED_CM_S = 6.0              # cruise speed of the push itself
PUSH_EASE_IN = 3.0                 # [0,10] -- see EASE_IN's doc
PUSH_EASE_OUT = 3.0                # [0,10]
PUSH_STEP_CM = 0.5                 # waypoint spacing along a push stroke

# -- adjusting: pulling back off the cube mid-push ------------------------------
ADJUST_MIN = 1                     # back-offs per push, drawn at random in [MIN, MAX]
ADJUST_MAX = 2
ADJUST_FRAC_MIN = 0.2              # back-offs happen between these fractions of the push
ADJUST_FRAC_MAX = 0.8
ADJUST_MIN_GAP_FRAC = 0.15         # two back-offs on one push are at least this far apart
ADJUST_BACKOFF_CM = 2.0            # how far the beak pulls back off the cube ...
ADJUST_BACKOFF_JITTER_CM = 0.7     # ... +/- this
ADJUST_LIFT_CM = 1.5               # ... and how far it rises while doing so
ADJUST_PAUSE_S = 0.4               # beat held off the cube ...
ADJUST_PAUSE_JITTER_S = 0.25       # ... +/- this
ADJUST_SPEED_CM_S = 8.0            # pull-back / come-back speed
ADJUST_EASE_IN = 1.0               # [0,10]
ADJUST_EASE_OUT = 2.0              # [0,10]
ADJUST_WAYPOINTS = 10

POST_PUSH_LIFT_CM = 3.0            # once a cube is placed the beak first rises straight up
                                   # this far (clear of the cube), then the look-back return
                                   # takes it to the look configuration; 0 = no straight lift
POST_PUSH_LIFT_SPEED_CM_S = 6.0

# -- gaze (look-at) ----------------------------------------------------------
# See HighH-LowR.py. On the approach arc the beak looks at the cube it is about
# to push and levels out gripper-down before it arrives.
GAZE_ENABLED = True
GAZE_EASE_IN_S = 4.0
GAZE_EASE_OUT_S = 2.0

# -- the LOOK configuration + head swings --------------------------------------
LOOK_CONFIG_J123_DEG = [0.0, 50.0, -110.0]

HEAD_SWING_SPEED_DPS = 50.0      # average speed of a head swing (largest joint travel / time)
HEAD_SWING_EASE_IN = 3.0         # [0,10]
HEAD_SWING_EASE_OUT = 2.0        # [0,10]
HEAD_SWING_WAYPOINTS = 30
LOOK_PAUSE_S = 0.4               # beat held on each look before the next thing happens
LOOK_MIN_TIP_Z_CM = Z_CUBE_COORD + 1.0
LOOK_MAX_AIM_ERROR_DEG = 5.0

RETURN_SPEED_DPS = 40.0          # the look-back return to the configuration
RETURN_EASE_IN = 2.0
RETURN_EASE_OUT = 4.0
RETURN_WAYPOINTS = 30
NEXT_LOOK_START_FRAC = 0.9       # fraction of a return's PATH after which the head stops
                                 # looking back at the placed cube and turns to the next one

# -- the end: look straight ahead ------------------------------------------------
LOOK_AHEAD_WRIST_J456_DEG = [50.0, 0.0, 0.0]   # HighH-LowR.py's nod pose
LOOK_AHEAD_BLEND_START_FRAC = 0.7              # like NEXT_LOOK_START_FRAC, on the last return

# -- approach arcs (copied from HighH-LowR.py) -----------------------------------
MAX_HEIGHT_TRAJECTORY = 15.0
MIN_ARC_HEIGHT_CM = 9.0
CRUISE_SPEED_CM_S = 22.0
ARC_TIME_EQUALIZATION = 0.5
EASE_IN = 4.0                    # [0,10] start-of-move acceleration shape. 0 = abrupt,
EASE_OUT = 5.0                   # [0,10] end-of-move deceleration shape.  10 = long, gentle S
PATH_WAYPOINTS = 60
MIN_SEGMENT_S = 0.02

VARIATION = 0                    # [0,1] -- see HighH-LowR.py
APEX_HEIGHT_JITTER_FRAC = 0.0
BOW_JITTER_CM = 0.0
EASE_JITTER = 0.0

# -- gripper (only ever sent once, to GRIP_BEAK_DEG) --------------------------------
GRIP_SPEED = 10
GRIP_SETTLE_S = 1.0              # the jaws must be fully shut before the arm moves
GRIP_MIN_GAP_S = 0.2
GRIP_MOVING_REPEATS = 2
GRIP_MOVING_REPEAT_GAP_S = 0.1
REACH_TOL_CM = 3.0

_AZ_REF = None                    # (x, y) tip position whose azimuth is rz's zero; set in main()


# ===========================================================================
# GEOMETRY / PROFILE HELPERS
# ===========================================================================

def _smootherstep(p):
    p = np.clip(p, 0.0, 1.0)
    return 6 * p ** 5 - 15 * p ** 4 + 10 * p ** 3


def _yaw(pts_xy):
    """rz for a run of waypoints, per ORIENT_LOCK. Returns a scalar (held), a
    per-waypoint list, or None (free) -- send_path accepts all three."""
    rz0 = PICK_ORIENTATION_DEG[2]
    if ORIENT_LOCK == "free":
        return None
    if ORIENT_LOCK != "base":
        return rz0
    ax, ay = (_AZ_REF if _AZ_REF is not None else (1.0, 0.0))
    az0 = math.degrees(math.atan2(ay, ax))
    return [rz0 + ORIENT_LOCK_SIGN * (math.degrees(math.atan2(p[1], p[0])) - az0)
            for p in pts_xy]


def _yaw_one(xy):
    """Scalar rz at a single point (for preflight / the recoil move)."""
    r = _yaw([xy])
    return r[0] if isinstance(r, list) else r


def _polyline_points(verts, s_query):
    """Points at arc-length(s) `s_query` along the poly-line through `verts`
    (N x 3). Returns (points, total_length)."""
    verts = np.asarray(verts, dtype=float)
    seg = np.diff(verts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    s = np.clip(np.asarray(s_query, dtype=float), 0.0, total)
    j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, max(len(seglen) - 1, 0))
    denom = np.where(seglen[j] > 1e-9, seglen[j], 1.0)
    frac = (s - cum[j]) / denom
    return verts[j] + frac[:, None] * seg[j], total


def _chord_len(p, q):
    """Horizontal (xy) distance between two points."""
    return float(np.hypot(q[0] - p[0], q[1] - p[1]))


def _arc_height(d, d_max):
    """Apex height (above the chord) for a move of horizontal length `d`, given
    the widest move `d_max`. Shared parabola: a = -MAX_HEIGHT / (d_max/2)^2, and
    c_i = -a * (d/2)^2 = MAX_HEIGHT * (d/d_max)^2."""
    if d_max <= 1e-6:
        return MIN_ARC_HEIGHT_CM
    c = MAX_HEIGHT_TRAJECTORY * (d / d_max) ** 2
    return float(np.clip(c, MIN_ARC_HEIGHT_CM, MAX_HEIGHT_TRAJECTORY))


def _parabola_points(origin, target, arc_height, rng, n_waypoints=None):
    """(list of `n_waypoints` (x,y,z), arc_length) along the arc from `origin`
    to `target`: the straight xy chord + a symmetric vertical parabolic lift of
    apex `arc_height` above the chord (== a*x^2 + c with x = (u-0.5)*chord).
    With `rng` the apex height and a sideways bow are jittered by VARIATION.
    `n_waypoints` defaults to PATH_WAYPOINTS -- pass fewer for a short, fast
    move (e.g. the recoil) where PATH_WAYPOINTS segments would floor its total
    duration at (PATH_WAYPOINTS-1)*MIN_SEGMENT_S regardless of cruise speed."""
    n_waypoints = PATH_WAYPOINTS if n_waypoints is None else int(n_waypoints)
    o = np.asarray(origin, dtype=float)
    t = np.asarray(target, dtype=float)
    z0, z1 = float(o[2]), float(t[2])

    h = float(arc_height)
    bow = 0.0
    if rng is not None and VARIATION > 0.0:
        h = max(MIN_ARC_HEIGHT_CM,
                h * (1.0 + float(rng.uniform(-1.0, 1.0)) * APEX_HEIGHT_JITTER_FRAC * VARIATION))
        bow = float(rng.uniform(-1.0, 1.0)) * BOW_JITTER_CM * VARIATION

    M = 200
    u = np.linspace(0.0, 1.0, M)
    xy = o[:2][None, :] + (t[:2] - o[:2])[None, :] * u[:, None]

    # sideways bow, perpendicular to the chord in the horizontal plane
    chord = t[:2] - o[:2]
    n = float(np.linalg.norm(chord))
    if n > 1e-6 and bow != 0.0:
        perp = np.array([-chord[1], chord[0]]) / n
        xy = xy + perp[None, :] * (bow * 4.0 * (u * (1.0 - u)))[:, None]

    # straight chord in z + symmetric parabolic lift (0 at both ends, peak h at u=0.5)
    z = z0 + (z1 - z0) * u + 4.0 * h * u * (1.0 - u)

    dense = np.column_stack([xy, z])
    L = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    pts, _ = _polyline_points(dense, np.linspace(0.0, L, n_waypoints))
    return [tuple(float(v) for v in p) for p in pts], L


def get_path(origin_point, target_point, arc_height, rng=None, n_waypoints=None):
    """Arc from origin to target as `n_waypoints` (x,y,z) points (default
    PATH_WAYPOINTS). `arc_height` is this move's apex above the chord --
    compute it with _arc_height()."""
    return _parabola_points(origin_point, target_point, arc_height, rng, n_waypoints)[0]


def get_durations(origin_point, target_point, arc_height,
                  ease_in_accel=EASE_IN, ease_out_accel=EASE_OUT,
                  cruise=CRUISE_SPEED_CM_S, duration=None, n_waypoints=None):
    """`n_waypoints`-1 segment durations (s) for the arc between the two points
    (n_waypoints defaults to PATH_WAYPOINTS -- must match whatever `n_waypoints`
    get_path() was called with for the same arc), shaped by the EASE_IN /
    EASE_OUT dials (0..10, no physical meaning). Total time is `duration`
    seconds if given, otherwise derived from `cruise` cm/s."""
    n_waypoints = PATH_WAYPOINTS if n_waypoints is None else int(n_waypoints)
    _, L = _parabola_points(origin_point, target_point, arc_height, None, n_waypoints)

    a = float(np.clip(ease_in_accel, 0.0, 10.0)) / 10.0
    b = float(np.clip(ease_out_accel, 0.0, 10.0)) / 10.0
    r_in = 0.05 + 0.45 * a
    r_out = 0.05 + 0.45 * b
    if r_in + r_out > 1.0:
        k = 1.0 / (r_in + r_out)
        r_in *= k
        r_out *= k

    tau = np.linspace(0.0, 1.0, 2001)
    v = np.ones_like(tau)
    m_in = tau < r_in
    p = tau[m_in] / r_in
    v[m_in] = (1.0 - a) * p + a * _smootherstep(p)
    m_out = tau > 1.0 - r_out
    q = (1.0 - tau[m_out]) / r_out
    v[m_out] = (1.0 - b) * q + b * _smootherstep(q)
    v = np.maximum(v, 1e-6)

    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tau))])
    mean_v = float(s[-1])                     # == average of v over [0, 1]
    s_norm = s / s[-1]
    T = float(duration) if duration is not None else L / max(cruise * mean_v, 1e-6)

    ss = np.linspace(0.0, 1.0, n_waypoints)
    tau_k = np.interp(ss, s_norm, tau)
    t_k = tau_k * T
    return np.maximum(np.diff(t_k), MIN_SEGMENT_S).tolist()


# ===========================================================================
# STATE CHECKS
# ===========================================================================

def current_pos(arm):
    return tuple(float(v) for v in arm.get_coords()[:3])


def _within(xyz, centre, half_extents):
    d = np.abs(np.asarray(xyz, float) - np.asarray(centre, float))
    return bool(np.all(d <= np.asarray(half_extents, float)))


def has_reached_cube(end_effector_coords, cube_xyz):
    return _within(end_effector_coords, cube_xyz, (REACH_TOL_CM,) * 3)


def has_reached_target(end_effector_coords, target_xyz):
    return _within(end_effector_coords, target_xyz, (REACH_TOL_CM,) * 3)


# ===========================================================================
# MOTION
# ===========================================================================

def go_home(arm, target=HOME):
    """Homing move to `target`, with the duration scaled to the joint
    distance so a long
    return from the far side is not crammed into HOME_MOVE_S (which makes
    move_joints -- no speed pre-check -- outrun the servos and shake).
    Returns bool -- move_joints() can refuse (e.g. the arm's current pose is
    already outside a joint's soft limit), and that must not pass silently."""
    try:
        dq = max(abs(a - b) for a, b in zip(arm.get_angles(), target))
    except Exception:
        dq = 0.0
    dur = max(HOME_MOVE_S, dq / HOME_RETURN_DPS)
    if dur > HOME_MOVE_S + 0.05:
        print(f"  homing over {dur:.1f}s (joint travel {dq:.0f} deg)")
    if not arm.move_joints(target, duration=dur):
        print(f"  homing REFUSED -- {arm.last_error}")
        return False
    return True


def _fire_gripper(arm, deg):
    """Send the gripper command twice with a tiny gap -- pymycobot drops a
    gripper packet that is not followed by a short quiet window. No long wait
    (caller decides). Returns False only if send_gripper itself refused."""
    ok = arm.send_gripper(deg, speed=GRIP_SPEED)
    time.sleep(0.06)
    arm.send_gripper(deg, speed=GRIP_SPEED)
    return bool(ok)


def _grip(arm, deg, label, wait_s=GRIP_SETTLE_S):
    """Gripper command with the arm standing still, then wait_s before
    anything else happens (never less than GRIP_MIN_GAP_S)."""
    _say(f"  gripper -> {deg:.0f} deg ({label})")
    if _TAPE is not None:
        _TAPE.append(("grip", float(deg)))
    elif not _fire_gripper(arm, deg):
        print(f"  send_gripper REFUSED -- {arm.last_error}")
        return False
    if wait_s < GRIP_MIN_GAP_S:
        _say(f"  (gripper wait {wait_s}s < floor {GRIP_MIN_GAP_S}s -- using the floor)")
    _pause(max(max(wait_s, GRIP_MIN_GAP_S) - 0.06, 0.0))
    return True


def _arc_orientations(arm, tail, durs, gaze_target):
    """(rx, ry, rz) for the waypoints `tail` of an arc starting at the arm's
    current orientation: gazing at gaze_target and leveling out before arrival
    (if GAZE_ENABLED and a target is given), else easing to the fixed
    PICK_ORIENTATION_DEG / ORIENT_LOCK pose."""
    rx0, ry0 = PICK_ORIENTATION_DEG[:2]
    rz_raw = _yaw(tail)
    rz_seq = rz_raw if isinstance(rz_raw, list) else [rz_raw] * len(tail)
    fixed = [(rx0, ry0, rz) for rz in rz_seq]
    start_rpy = arm.get_coords()[3:]
    if GAZE_ENABLED and gaze_target is not None:
        return gaze_then_level_waypoints(tail, gaze_target, fixed, start_rpy, durs,
                                         GAZE_EASE_IN_S, GAZE_EASE_OUT_S)
    # eases FROM the arm's actual current orientation (which, after a gazed
    # arc, can be far from PICK_ORIENTATION_DEG) -- a no-op when it's already
    # there, e.g. the whole run has GAZE_ENABLED=False.
    return ease_to_rpy(fixed, start_rpy, durs, GAZE_EASE_IN_S)


class _GripTimer(threading.Thread):
    """Sends ONE gripper command while the arm is moving: started right before
    a blocking stream, it waits delay_s and then calls arm.send_gripper from
    this background thread (GRIP_MOVING_REPEATS times). ArmConnection puts
    every serial write behind one lock, so the packet lands between two
    setpoints. finish() -- called when the stream is over -- sends at once if
    the delay has not elapsed yet, so the command is never lost."""

    def __init__(self, arm, deg, delay_s, label):
        super().__init__(daemon=True)
        self.arm, self.deg, self.delay_s, self.label = arm, float(deg), float(delay_s), label
        self._now = threading.Event()

    def run(self):
        self._now.wait(max(self.delay_s, 0.0))
        print(f"  gripper -> {self.deg:.0f} deg ({self.label}, while moving)")
        for i in range(max(int(GRIP_MOVING_REPEATS), 1)):
            if i:
                time.sleep(GRIP_MOVING_REPEAT_GAP_S)
            self.arm.send_gripper(self.deg, speed=GRIP_SPEED)

    def finish(self):
        self._now.set()
        self.join()


def _execute_plan(arm, plan, grip=None):
    """arm._execute(plan), with an optional gripper command fired DURING it.
    grip = (deg, delay_s, label) or None. The timer starts with the stream, so
    delay_s counts from the moment the arm actually starts moving."""
    timer = _GripTimer(arm, *grip) if grip is not None else None
    if timer is not None:
        timer.start()
    try:
        return arm._execute(plan)
    finally:
        if timer is not None:
            timer.finish()


# ===========================================================================
# THE TAPE -- rehearse the whole run first, then play it back
# ===========================================================================
# Working out a motion (the look searches, the IK for an arc) takes real time
# on the Jetson, and doing it between actions left the arm standing still.
# So the choreography runs TWICE:
#   1. REHEARSAL on a second, simulated Arm(mock=True): nothing is streamed and
#      nothing sleeps. Every action -- a finished Plan, a gripper command, a
#      deliberate pause, a log line -- is appended to a list, the tape, and the
#      simulated arm is put at the motion's end pose so the next step plans
#      from the right place.
#   2. PLAYBACK on the real arm: the tape is executed in order. No kinematics
#      and no planning happen between motions, so the only gaps left are the
#      deliberate pauses.
# Every motion function reaches the arm only through _do_plan / _pause / _say /
# _grip below, which record while _TAPE is a list and act when it is None.

_TAPE = None


def _say(text):
    if _TAPE is not None:
        _TAPE.append(("say", text))
    else:
        print(text)


def _pause(seconds):
    if _TAPE is not None:
        _TAPE.append(("pause", float(seconds)))
    else:
        time.sleep(seconds)


def _do_plan(arm, plan, grip=None):
    """Execute a finished Plan -- or, in rehearsal, record it and move the
    simulated arm to its end pose. grip: optional (deg, delay_s, label)
    gripper command fired while it runs; plan=None means just that command.
    Returns an Execution."""
    if _TAPE is not None:
        _TAPE.append(("plan", plan, grip, arm.jerk))
        if plan is not None:
            arm.conn.raw.set_angles_directly([float(v) for v in plan.q_waypoints[-1]])
        return Execution(ok=True)
    return _run_plan(arm, plan, grip)


def _run_plan(arm, plan, grip):
    if plan is None:                       # nothing to move: just the gripper command
        timer = _GripTimer(arm, *grip)
        timer.start()
        timer.finish()
        return Execution(ok=True)
    return _execute_plan(arm, plan, grip)


def _play(arm, tape):
    """Run a recorded tape on the real arm. Returns bool."""
    for step in tape:
        kind = step[0]
        if kind == "say":
            print(step[1])
        elif kind == "pause":
            time.sleep(step[1])
        elif kind == "grip":
            if not _fire_gripper(arm, step[1]):
                print(f"  send_gripper REFUSED -- {arm.last_error}")
                return False
        elif kind == "plan":
            _, plan, grip, jerk = step
            arm.jerk = jerk
            ex = _run_plan(arm, plan, grip)
            arm.jerk = 0.0
            arm.last_plan, arm.last_execution = plan, ex
            if not ex.ok:
                print(f"  execution ABORTED -- {ex.error}")
                return False
            if ex.late_deadlines:
                print(f"  !! {ex.late_deadlines} late control-loop deadline(s) during this move")
        elif kind == "check":
            _, what, target = step
            cur = current_pos(arm)
            if not _within(cur, target, (REACH_TOL_CM,) * 3):
                print(f"  !! tip at {tuple(round(v, 2) for v in cur)}, expected "
                      f"{tuple(round(v, 1) for v in target)} +/- {REACH_TOL_CM} cm ({what})")
    return True


def _plan_arc(arm, pts, durs, label, gaze_target=None):
    """Plan (no motion) the arc through pts -- pts[0] is the implicit start --
    and return the Plan, or None if it was refused.
    gaze_target: if given (and GAZE_ENABLED), the gripper tip points at this
    3D point for the whole arc instead of holding PICK_ORIENTATION_DEG. A wide
    carry occasionally asks for a look-at pose this arm's elbow/wrist can't
    reach (or can only reach too fast) -- if the gazed plan is REFUSED, this
    falls back to the fixed PICK_ORIENTATION_DEG for THIS arc only, rather
    than aborting the run."""
    tail = pts[1:]
    xs = [p[0] for p in tail]
    ys = [p[1] for p in tail]
    zs = [p[2] for p in tail]

    def _plan(gaze):
        rx, ry, rz = _arc_orientations(arm, tail, durs, gaze)
        return arm.plan_path(x=xs, y=ys, z=zs, rx=rx, ry=ry, rz=rz, durations=list(durs))

    arm.last_error = None
    arm.last_execution = None
    pl = _plan(gaze_target)
    if not pl.ok and GAZE_ENABLED and gaze_target is not None:
        _say(f"  {label}: gaze pose unreachable ({pl.error}) "
             f"-- retrying this arc with fixed orientation")
        pl = _plan(None)
    arm.last_plan = pl
    if not pl.ok:
        arm.last_error = pl.error
        _say(f"  {label}: plan REFUSED -- {pl.error}")
        return None
    return pl


def _send_arc(arm, pts, durs, label, gaze_target=None, grip=None):
    """Parabolic move: _plan_arc(), then hand the Plan to _do_plan(), so the
    same code serves the rehearsal (record) and a live move (execute).
    grip: optional (deg, delay_s, label) gripper command fired while the arc
    runs. Returns bool."""
    if len(pts) < 2:
        _say(f"  {label}: negligible, skipped")
        return True
    pl = _plan_arc(arm, pts, durs, label, gaze_target)
    if pl is None:
        return False
    ex = _do_plan(arm, pl, grip)
    arm.last_execution = ex
    if not ex.ok:
        arm.last_error = ex.error
        _say(f"  {label}: execution ABORTED -- {ex.error}")
        return False
    _say(f"  {label}: {pl.path_length_cm:.1f} cm, {pl.duration_s:.2f} s, "
         f"peak {pl.peak_joint_dps:.0f} deg/s")
    if ex.late_deadlines:
        _say(f"  !! {ex.late_deadlines} late control-loop deadline(s) during this arc")
    return True


def _arc_from(origin, target, d_max, cruise_dur_max, rng):
    """(pts, durs) of a reach / carry arc from `origin` -- same shared-parabola
    height, ease jitter and ARC_TIME_EQUALIZATION as main()'s precompute."""
    ei = EASE_IN + float(rng.uniform(-1.0, 1.0)) * EASE_JITTER * VARIATION
    eo = EASE_OUT + float(rng.uniform(-1.0, 1.0)) * EASE_JITTER * VARIATION
    h = _arc_height(_chord_len(origin, target), d_max)
    pts = get_path(origin, target, h, rng)
    cruise_dur = sum(get_durations(origin, target, h, ei, eo))
    t_i = cruise_dur * (1.0 - ARC_TIME_EQUALIZATION) + cruise_dur_max * ARC_TIME_EQUALIZATION
    return pts, get_durations(origin, target, h, ei, eo, duration=t_i)


def _ease_curve(ease_in, ease_out):
    """(progress_at, time_at): the [0,10] ease dials as a pair of functions on
    [0,1] -- eased progress at a fraction of the move's TIME, and the inverse.
    Same shape as every arc (celebration_durations gives the time of evenly
    spaced progress steps; here it is read both ways)."""
    s_grid = np.linspace(0.0, 1.0, 201)
    t_grid = np.concatenate([[0.0], np.cumsum(celebration_durations(ease_in, ease_out, 1.0, 201))])
    t_grid /= t_grid[-1]
    return (lambda x: np.interp(np.clip(x, 0.0, 1.0), t_grid, s_grid),
            lambda p: np.interp(np.clip(p, 0.0, 1.0), s_grid, t_grid))


def _ticks(total_s):
    """Sample times for a move of total_s: one per control tick, never closer
    together than 1 / config.CONTROL_RATE_HZ (floor), ends included."""
    n = max(1, int(total_s * config.CONTROL_RATE_HZ))
    return np.linspace(0.0, total_s, n + 1)


def _timed_joint_move(arm, sample, label, grip=None):
    """Run a joint-space move given as sample(scale) -> (t, rows): setpoint
    times (s, from _ticks) and joint rows, row 0 being the current pose; scale
    >= 1 stretches every duration in it. One setpoint per control tick is what
    keeps the stream within what the serial link can carry.

    Direct arm._execute() (through _do_plan), which applies no hardware speed
    check of its own -- so the move is stretched (never sped up) to keep every
    joint within MAX_JOINT_SPEED_DPS. grip: optional (deg, delay_s, label)
    gripper command fired while it runs. Returns bool."""
    t, rows = sample(1.0)
    for r in rows[1:]:
        problems = check_joint_limits(r)
        if problems:
            _say(f"  {label} REFUSED -- violates joint limits: {'; '.join(problems)}")
            return False
    min_s = min_feasible_duration_s(rows, np.diff(t) / t[-1])
    if min_s > t[-1] + 1e-6:
        t, rows = sample(min_s / t[-1])
    plan = Plan(ok=True, q_waypoints=rows, timestamps=t, duration_s=float(t[-1]))
    ex = _do_plan(arm, plan, grip)
    if not ex.ok:
        _say(f"  {label} REFUSED -- {ex.error}")
        return False
    if ex.late_deadlines:
        _say(f"  !! {label}: {ex.late_deadlines} late control-loop deadline(s)")
    return True


def _eased_joint_move(arm, q_waypoints, speed_dps, ease_in, ease_out, label, grip=None):
    """One eased joint-space stroke from the arm's current pose along
    q_waypoints (rows evenly spaced along the stroke; the last row is the
    goal). Takes (largest total joint travel) / speed_dps seconds, shaped by
    the [0,10] ease dials. q_waypoints only define the path's SHAPE; the
    setpoints are sampled in time by _timed_joint_move. grip: optional
    (deg, delay_s, label) gripper command fired while the stroke runs (sent at
    once if there is nothing to move). Returns bool."""
    q0 = np.asarray(arm.get_angles(), dtype=float)
    path = np.vstack([q0[None, :], np.asarray(q_waypoints, dtype=float)])
    travel = float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0)))
    if travel < 0.5:
        if grip is not None:
            _do_plan(arm, None, grip)
        return True
    progress_at, _ = _ease_curve(ease_in, ease_out)
    idx = np.arange(len(path), dtype=float)
    base_s = travel / max(float(speed_dps), 1e-6)

    def sample(scale):
        t = _ticks(base_s * scale)
        u = progress_at(t / t[-1]) * (len(path) - 1)
        rows = np.column_stack([np.interp(u, idx, path[:, j]) for j in range(path.shape[1])])
        rows[0], rows[-1] = path[0], path[-1]
        return t, rows

    return _timed_joint_move(arm, sample, label, grip)


def _line_to(q_from, q_to, n):
    """n rows evenly spaced from q_from (excluded) to q_to (included)."""
    a, b = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    return a[None, :] + (b - a)[None, :] * np.linspace(0.0, 1.0, int(n) + 1)[1:, None]


def _look_q(q_now, target):
    """(q, aim_error_deg): the look configuration with the head (J4/J5)
    pointed at `target`; J6 as it is now. With GAZE_ENABLED off, or if no
    admissible head pose exists, the wrist is HOME's (gripper down) and the
    error is None."""
    q = [float(v) for v in q_now]
    q[:3] = [float(v) for v in LOOK_CONFIG_J123_DEG]
    if GAZE_ENABLED:
        q_look, err = wrist_look_angles(q, target, LOOK_MIN_TIP_Z_CM)
        if q_look is not None:
            return q_look, err
    q[3:] = [float(v) for v in HOME[3:]]
    return q, None


def _swing_head(arm, target, label):
    """At (or on the way to) the look configuration, swing the head to look
    at `target`, then hold LOOK_PAUSE_S. Returns bool."""
    q_now = arm.get_angles()
    q_look, err = _look_q(q_now, target)
    if not _eased_joint_move(arm, _line_to(q_now, q_look, HEAD_SWING_WAYPOINTS),
                             HEAD_SWING_SPEED_DPS, HEAD_SWING_EASE_IN, HEAD_SWING_EASE_OUT,
                             label):
        return False
    aim = f"aim error {err:.1f} deg" if err is not None else "no look (gripper down)"
    _say(f"  {label}: {tuple(round(float(v), 1) for v in target)}  ({aim})")
    _pause(LOOK_PAUSE_S)
    return True


def _phase_look_around(arm):
    """Phase 1: FIRST get to the look configuration (J1..J3 only, the wrist
    stays gripper-down), and only then look at the cubes (their average
    position) and the target zone (the targets' average). The look that
    follows is the first cube's own, straight at it."""
    q_now = np.asarray(arm.get_angles(), dtype=float)
    q_cfg = q_now.copy()
    q_cfg[:3] = LOOK_CONFIG_J123_DEG
    if not _eased_joint_move(arm, _line_to(q_now, q_cfg, RETURN_WAYPOINTS), RETURN_SPEED_DPS,
                             RETURN_EASE_IN, RETURN_EASE_OUT, "to the look configuration"):
        return False
    _say("  at the look configuration")
    _pause(LOOK_PAUSE_S)

    cubes = tuple(float(v) for v in np.mean(np.asarray(CUBES_INITIAL_POINTS, float), axis=0))
    targets = tuple(float(v) for v in np.mean(np.asarray(CUBES_TARGET_POINTS, float), axis=0))
    for target, label in ((cubes, "look at the cubes"),
                          (targets, "look at the target zone")):
        if not _swing_head(arm, target, label):
            return False
    return True


def _return_looking_back(arm, target, next_wrist=None, start_frac=1.0, next_name="next pose"):
    """After a push: J1..J3 travel back to LOOK_CONFIG_J123_DEG (J6 back to
    HOME's) while the head keeps looking at `target`, the cube just placed --
    see _gaze.wrist_track_angles.

    next_wrist: optional [J4, J5, J6] the head should turn to on the way (the
    look at the next cube, or the nod pose). Once the return has covered
    start_frac of its PATH the head leaves the target and turns there ON ITS
    OWN CLOCK: (largest wrist travel) / HEAD_SWING_SPEED_DPS seconds, eased by
    HEAD_SWING_EASE_IN / OUT. Nothing ties the end of that turn to the arm
    arriving -- it may finish early, or carry on after J1..J3 have stopped.
    While it turns, the wrist cross-fades from the (still evolving) look-back
    track to next_wrist, so there is no kink at the hand-over. Returns bool."""
    q0 = np.asarray(arm.get_angles(), dtype=float)
    q_end = q0.copy()
    q_end[:3] = LOOK_CONFIG_J123_DEG
    q_end[5] = HOME[5]
    if GAZE_ENABLED:
        path = np.vstack([q0[None, :], _line_to(q0, q_end, RETURN_WAYPOINTS)])
        floor_cm = current_pos(arm)[2] - 0.2               # never dip below the current height
        path = wrist_track_angles(path, target, floor_cm)  # J4/J5 re-aimed row by row
    else:
        q_end[3:5] = HOME[3:5]
        path = np.vstack([q0[None, :], _line_to(q0, q_end, RETURN_WAYPOINTS)])

    idx = np.arange(len(path), dtype=float)
    ret_progress, ret_time = _ease_curve(RETURN_EASE_IN, RETURN_EASE_OUT)
    turn_progress, _ = _ease_curve(HEAD_SWING_EASE_IN, HEAD_SWING_EASE_OUT)
    ret_s = float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0))) / max(RETURN_SPEED_DPS, 1e-6)

    turning = next_wrist is not None and start_frac < 1.0
    trig_s = turn_s = 0.0
    if turning:
        nxt = np.asarray(next_wrist, dtype=float)
        trig_s = float(ret_time(start_frac)) * ret_s       # when start_frac of the PATH is done
        at_trig = np.array([np.interp(start_frac * (len(path) - 1), idx, path[:, j])
                            for j in (3, 4, 5)])
        turn_s = max(float(np.max(np.abs(nxt - at_trig))) / max(HEAD_SWING_SPEED_DPS, 1e-6), 1e-3)
    base_s = max(ret_s, trig_s + turn_s)

    def sample(scale):
        t = _ticks(base_s * scale)
        u = ret_progress(t / (ret_s * scale)) * (len(path) - 1)     # holds the end once arrived
        rows = np.column_stack([np.interp(u, idx, path[:, j]) for j in range(path.shape[1])])
        rows[0] = path[0]
        rows[-1, :3] = path[-1, :3]
        if turning:
            w = turn_progress((t - trig_s * scale) / (turn_s * scale))
            rows[:, 3:6] += w[:, None] * (nxt[None, :] - rows[:, 3:6])
            rows[-1, 3:6] = nxt
        else:
            rows[-1] = path[-1]
        return t, rows

    if not _timed_joint_move(arm, sample, "return"):
        return False
    msg = "  returned to the look configuration" + (", looking back at the placed cube" if GAZE_ENABLED else "")
    if turning:
        when = ("the head finished %.1f s before the arm" % (ret_s - trig_s - turn_s)
                if trig_s + turn_s <= ret_s else
                "the head kept turning %.1f s after the arm arrived" % (trig_s + turn_s - ret_s))
        msg += f", then turning to {next_name} from {start_frac * 100:.0f}% of the way ({when})"
    _say(msg)
    return True


def _look_ahead(arm):
    """The end: at the look configuration, swing the head to
    LOOK_AHEAD_WRIST_J456_DEG (straight ahead) and hold -- HighH-LowR.py's nod
    pose, without the nod. Usually already there: the last return turns the
    head on the way (LOOK_AHEAD_BLEND_START_FRAC)."""
    base = np.asarray(list(LOOK_CONFIG_J123_DEG) + list(LOOK_AHEAD_WRIST_J456_DEG), dtype=float)
    if not _eased_joint_move(arm, _line_to(arm.get_angles(), base, HEAD_SWING_WAYPOINTS),
                             HEAD_SWING_SPEED_DPS, HEAD_SWING_EASE_IN, HEAD_SWING_EASE_OUT,
                             "look ahead"):
        _say("  could not turn to look straight ahead")
        return True
    _say("  looking straight ahead")
    _pause(LOOK_PAUSE_S)
    return True


# ===========================================================================
# THE BEAK PUSH
# ===========================================================================

def _push_geometry(k):
    """Points (cm) of cube k's push, all at PUSH_Z_CM unless noted:
      u       -- unit push direction in xy (initial -> target), as (ux, uy, 0)
      contact -- where the beak comes down: BEAK_CLEARANCE_CM off the cube's
                 far face (the face away from the target)
      end     -- where the beak stops: touching the cube's far face with the
                 cube centred on its target
      hover   -- PUSH_HOVER_CM above contact (the approach arc ends there)"""
    s = np.asarray(CUBES_INITIAL_POINTS[k], dtype=float)
    t = np.asarray(CUBES_TARGET_POINTS[k], dtype=float)
    d = t[:2] - s[:2]
    n = float(np.linalg.norm(d))
    u = np.array([d[0] / n, d[1] / n, 0.0]) if n > 1e-6 else np.array([0.0, -1.0, 0.0])
    half = CUBE_SIZE_CM / 2.0
    contact = np.array([s[0], s[1], PUSH_Z_CM]) - u * (half + BEAK_CLEARANCE_CM)
    end = np.array([t[0], t[1], PUSH_Z_CM]) - u * half
    hover = contact + np.array([0.0, 0.0, PUSH_HOVER_CM])
    tup = lambda v: tuple(float(x) for x in v)
    return {"u": u, "contact": tup(contact), "end": tup(end), "hover": tup(hover)}


def _line_move(arm, goal, speed, ease_in, ease_out, label, n_waypoints=None):
    """Straight Cartesian line from the tip's current position to `goal`, beak
    straight down (PICK_ORIENTATION_DEG), eased by the [0,10] dials.
    n_waypoints defaults to one per PUSH_STEP_CM. Returns bool."""
    start = current_pos(arm)
    L = float(np.linalg.norm(np.asarray(goal, float) - np.asarray(start, float)))
    if L < 0.05:
        return True
    n = n_waypoints or max(3, int(math.ceil(L / PUSH_STEP_CM)) + 1)
    pts = get_path(start, goal, 0.0, None, n_waypoints=n)
    durs = get_durations(start, goal, 0.0, ease_in, ease_out, cruise=speed, n_waypoints=n)
    return _send_arc(arm, pts, durs, label)


def _adjust_fractions(rng):
    """Sorted fractions of the push at which the beak pulls back: a random
    count in [ADJUST_MIN, ADJUST_MAX], at least ADJUST_MIN_GAP_FRAC apart."""
    m = int(rng.integers(ADJUST_MIN, ADJUST_MAX + 1)) if ADJUST_MAX > 0 else 0
    for _ in range(100):
        f = np.sort(rng.uniform(ADJUST_FRAC_MIN, ADJUST_FRAC_MAX, m))
        if m < 2 or np.all(np.diff(f) >= ADJUST_MIN_GAP_FRAC):
            return [float(v) for v in f]
    return [float(v) for v in f[:1]]          # could not space them -- just one


def _push_cube(arm, k, rng):
    """Beak down beside cube k, then push it in a straight line to its target,
    pulling back off it and coming back at random points ("adjusting").
    Returns bool."""
    g = _push_geometry(k)
    u = g["u"]
    if not _line_move(arm, g["contact"], PECK_SPEED_CM_S, PECK_EASE_IN, PECK_EASE_OUT,
                      "  beak down", n_waypoints=PECK_WAYPOINTS):
        return False

    c, e = np.asarray(g["contact"], float), np.asarray(g["end"], float)
    fracs = _adjust_fractions(rng)
    _say(f"  pushing cube #{k + 1} {np.linalg.norm(e - c):.1f} cm"
         + (f", adjusting at {', '.join(f'{f * 100:.0f}%' for f in fracs)}" if fracs else ""))
    for f in fracs:
        p = tuple(float(v) for v in c + f * (e - c))
        if not _line_move(arm, p, PUSH_SPEED_CM_S, PUSH_EASE_IN, PUSH_EASE_OUT, "  push"):
            return False
        back = (ADJUST_BACKOFF_CM
                + float(rng.uniform(-1.0, 1.0)) * ADJUST_BACKOFF_JITTER_CM)
        off = tuple(float(v) for v in
                    np.asarray(p) - u * max(back, 0.0) + np.array([0.0, 0.0, ADJUST_LIFT_CM]))
        if not _line_move(arm, off, ADJUST_SPEED_CM_S, ADJUST_EASE_IN, ADJUST_EASE_OUT,
                          "  pull back", n_waypoints=ADJUST_WAYPOINTS):
            return False
        _pause(max(ADJUST_PAUSE_S + float(rng.uniform(-1.0, 1.0)) * ADJUST_PAUSE_JITTER_S, 0.0))
        if not _line_move(arm, p, ADJUST_SPEED_CM_S, ADJUST_EASE_IN, ADJUST_EASE_OUT,
                          "  back to the cube", n_waypoints=ADJUST_WAYPOINTS):
            return False
    if not _line_move(arm, g["end"], PUSH_SPEED_CM_S, PUSH_EASE_IN, PUSH_EASE_OUT, "  push"):
        return False
    if _TAPE is not None:
        _TAPE.append(("check", "push end", g["end"]))
    return True


# ===========================================================================
# PREFLIGHT
# ===========================================================================

def _seg_point_dist(p, a, b):
    """xy distance from point p to the segment a-b."""
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def preflight(arm, order):
    rx0, ry0 = PICK_ORIENTATION_DEG[:2]
    print("\n--- preflight: beak points reachable + push corridors clear (no motion) ---")
    bad = 0

    # every point the beak goes through, incl. the furthest/highest pull-backs
    checks = []
    for k in order:
        g = _push_geometry(k)
        c, e = np.asarray(g["contact"], float), np.asarray(g["end"], float)
        checks += [(f"hover{k+1}", g["hover"]), (f"contact{k+1}", g["contact"]),
                   (f"end{k+1}", g["end"]),
                   (f"lifted{k+1}", tuple(e + np.array([0.0, 0.0, POST_PUSH_LIFT_CM])))]
        far = ADJUST_BACKOFF_CM + ADJUST_BACKOFF_JITTER_CM
        for f in (ADJUST_FRAC_MIN, ADJUST_FRAC_MAX):
            p = c + f * (e - c) - g["u"] * far + np.array([0.0, 0.0, ADJUST_LIFT_CM])
            checks.append((f"pullback{k+1}@{f:.1f}", tuple(p)))
    for name, (x, y, z) in checks:
        pl = arm.plan_coords(x=x, y=y, z=z, rx=rx0, ry=ry0, rz=_yaw_one((x, y)),
                             speed=config.DEFAULT_SPEED_CM_S)
        if pl.ok or "already at" in (pl.error or "").lower():
            print(f"  OK  {name:16s} ({x:5.1f},{y:6.1f},{z:4.1f})")
        else:
            print(f"  BAD {name:16s} ({x:5.1f},{y:6.1f},{z:4.1f})  {pl.error}")
            bad += 1

    # a pushed cube sweeps its corridor; nothing else may stand in it. Cubes
    # are treated as discs of their circumradius -- conservative.
    r_cube = CUBE_SIZE_CM / math.sqrt(2.0)
    placed = []
    for i, k in enumerate(order):
        s, t = CUBES_INITIAL_POINTS[k], CUBES_TARGET_POINTS[k]
        g = _push_geometry(k)
        for name, ob in ([(f"waiting cube {j+1}", CUBES_INITIAL_POINTS[j]) for j in order[i + 1:]]
                         + [(f"placed cube {j+1}", CUBES_TARGET_POINTS[j]) for j in placed]):
            d_cube = _seg_point_dist(ob, s, t)
            d_beak = _seg_point_dist(ob, g["contact"], g["end"])
            if d_cube < 2.0 * r_cube or d_beak < r_cube + BEAK_CLEARANCE_CM:
                print(f"  BAD push {k+1}: {name} at {ob} is in the way "
                      f"(cube {d_cube:.1f} cm / beak {d_beak:.1f} cm from the push line)")
                bad += 1
        placed.append(k)
    print(f"--- preflight: {'all clear' if bad == 0 else f'{bad} problem(s)'} ---")
    return bad == 0


# ===========================================================================
# MAIN
# ===========================================================================

def _choreography(arm, order, d_max, cruise_dur_max, rng):
    """The whole run, written against whatever `arm` it is given -- rehearsed
    on a simulated arm with _TAPE recording, then played back (see
    HighH-LowR.py). Returns False if anything was refused."""
    if not _grip(arm, GRIP_BEAK_DEG, "beak shut, for the whole run", wait_s=GRIP_SETTLE_S):
        return False

    _say("\n=== phase 1: looking around ===")
    if not _phase_look_around(arm):
        return False

    for i, k in enumerate(order):
        cube = CUBES_INITIAL_POINTS[k]
        g = _push_geometry(k)
        _say(f"\n=== cube {i + 1}/{len(order)}  push cube #{k + 1}  "
             f"{tuple(round(v, 1) for v in cube)} -> "
             f"{tuple(round(v, 1) for v in CUBES_TARGET_POINTS[k])} ===")

        # look at it (the first time: Phase 1's third look), then arc over to
        # above its far side, gazing at it
        if not _swing_head(arm, cube, "look at the cube"):
            return False
        pts, durs = _arc_from(current_pos(arm), g["hover"], d_max, cruise_dur_max, rng)
        if not _send_arc(arm, pts, durs, "approach", gaze_target=cube):
            return False

        if not _push_cube(arm, k, rng):
            return False

        if POST_PUSH_LIFT_CM > 0.0:
            e = g["end"]
            if not _line_move(arm, (e[0], e[1], e[2] + POST_PUSH_LIFT_CM),
                              POST_PUSH_LIFT_SPEED_CM_S, PECK_EASE_IN, PECK_EASE_OUT,
                              "  beak up", n_waypoints=PECK_WAYPOINTS):
                return False

        # head up and back to the look configuration, watching the cube just
        # placed -- and part-way there turning to the next cube (or ahead)
        placed = CUBES_TARGET_POINTS[k]
        if i == len(order) - 1:
            nxt = (LOOK_AHEAD_WRIST_J456_DEG, LOOK_AHEAD_BLEND_START_FRAC, "look straight ahead")
        else:
            q_cfg = list(LOOK_CONFIG_J123_DEG) + [float(v) for v in HOME[3:]]
            nxt = (_look_q(q_cfg, CUBES_INITIAL_POINTS[order[i + 1]])[0][3:],
                   NEXT_LOOK_START_FRAC, "the next cube")
        if not _return_looking_back(arm, placed, next_wrist=nxt[0], start_frac=nxt[1],
                                    next_name=nxt[2]):
            return False

    _say("\nall cubes placed.")
    _say("\n=== the end: looking straight ahead ===")
    return _look_ahead(arm)


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
    if sorted(PUSH_ORDER) != list(range(len(CUBES_INITIAL_POINTS))):
        print("PUSH_ORDER must list every cube index exactly once.")
        return 1
    if not args.mock and not args.yes:
        print("This will move the robot arm and actuate the gripper. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    rng = np.random.default_rng(RANDOM_SEED)
    order = [int(k) for k in PUSH_ORDER]

    p_home = pose_coords(HOME)                       # mm/deg (Z_RELATIVE_TO_JOINT1 assumed False)
    home_tip = (p_home[0] / 10.0, p_home[1] / 10.0, p_home[2] / 10.0)

    global _AZ_REF                                   # rz = PICK_ORIENTATION_DEG[2] at HOME's azimuth
    _AZ_REF = (home_tip[0], home_tip[1])

    print(f"push order (cube indices): {order}")
    p_look = pose_coords(list(LOOK_CONFIG_J123_DEG) + list(HOME[3:]))
    look_tip = (p_look[0] / 10.0, p_look[1] / 10.0, p_look[2] / 10.0)

    # the approach arcs (look configuration -> above a cube's far side) share
    # HighH-LowR.py's apex rule and ARC_TIME_EQUALIZATION
    hovers = [_push_geometry(k)["hover"] for k in order]
    d_max = max((_chord_len(look_tip, h) for h in hovers), default=1.0) or 1.0
    cruise_dur_max = max(sum(get_durations(look_tip, h, _arc_height(_chord_len(look_tip, h), d_max),
                                           EASE_IN, EASE_OUT)) for h in hovers)

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)

    bridge = None
    if args.rviz:
        from _rviz_bridge import RvizBridge
        flat_path = [tuple(map(float, look_tip))]
        for k in order:
            g = _push_geometry(k)
            e = g["end"]
            flat_path += [g["hover"], g["contact"], e, (e[0], e[1], e[2] + POST_PUSH_LIFT_CM),
                          tuple(map(float, look_tip))]
        cube_pts = [tuple(map(float, c)) for c in CUBES_INITIAL_POINTS]
        try:
            bridge = RvizBridge(arm.get_angles, path_xyz_cm=flat_path, cube_points=cube_pts,
                                tip_source=arm.get_coords,
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

        print("homing...")
        if not go_home(arm):
            print("\nhoming failed -- fix the arm's position (see error above), then try again.")
            return 1
        time.sleep(SETTLE_S)
        print(f"start pose (tip, cm/deg): {[round(v, 2) for v in arm.get_coords()]}")

        if PREFLIGHT and not preflight(arm, order):
            print("\npreflight failed -- fix the cube layout (see above). Nothing moved.")
            go_home(arm)
            return 1

        # ---- rehearsal: compute EVERYTHING now, on a simulated arm ----------
        global _TAPE
        print("\nrehearsing the whole run on a simulated arm (nothing moves yet)...")
        t_rehearse = time.perf_counter()
        sim = Arm(mock=True, quiet_mock=True)          # the rehearsal arm -- no MOCK warning
        sim.conn.raw.set_angles_directly([float(v) for v in HOME])
        _TAPE = []
        try:
            ok = _choreography(sim, order, d_max, cruise_dur_max, rng)
        finally:
            tape, _TAPE = _TAPE, None
            sim.close()
        if not ok:
            for step in tape:                      # show how far the rehearsal got
                if step[0] == "say":
                    print(step[1])
            print("\nrehearsal failed (see above) -- the arm has not moved beyond homing.")
            return 1
        n_moves = sum(1 for step in tape if step[0] == "plan" and step[1] is not None)
        print(f"precomputed {n_moves} motions in {time.perf_counter() - t_rehearse:.1f} s "
              f"-- playing back")

        # ---- playback: no computing between actions --------------------------
        if not _play(arm, tape):
            print("\naborting run."); go_home(arm); return 1
        return 0

    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping the arm.")
        arm.stop()
        return 1
    finally:
        if bridge is not None:
            bridge.stop()
        arm.close()


if __name__ == "__main__":
    _sys.exit(main())
