#!/usr/bin/env python3
"""
MOVEMENT PROFILE: High-human / Low-robot  (look, then reach) -- TEST VARIANT
============================================================================

TEST of HighH-LowR.py with the GRIPPER MOVING WHILE THE ARM MOVES:
  * three gripper openings: GRIP_OPEN_DEG to approach / release a cube,
    GRIP_CLOSED_DEG on a cube, and GRIP_LOOK_DEG (nearly shut) while it looks
    around, returns to the look configuration, and nods;
  * the gripper OPENS as the arm travels toward a cube (GRIP_OPEN_DELAY_S
    after the reach starts);
  * at a cube / a target the arm WAITS for the gripper to finish closing /
    opening before it moves off (GRIP_CLOSE_WAIT_S / GRIP_OPEN_WAIT_S);
  * after a drop, the gripper goes to GRIP_LOOK_DEG on the way back to the
    look configuration, GRIP_CLOSE_DELAY_S after the return starts -- late
    enough not to re-grab the cube just dropped;
  * on the LAST return the head looks back at the target only for the first
    part (NOD_BLEND_START_FRAC), then turns toward the nod pose while the arm
    is still travelling, so it arrives ready to nod;
  * Phase 1 looks at the cubes' average, the targets' average, then straight
    at the first cube.
arm.move_joints() / arm.send_path() only return once the motion is over (they
stream setpoints in a loop), so a gripper call written after them runs too
late. Here the gripper command is sent from a short background thread while
the stream runs -- see _GripTimer / _execute_plan.

Move cubes from one side of the frame to the other the way a person would:
LOOK at things first, reach in smooth parabolic arcs, and come back to a
resting posture between cubes.

Everything is organised around ONE arm configuration, LOOK_CONFIG_J123_DEG
(J1 base, J2, J3 only). J4..J6 -- the "head" -- are left free there, so the
gripper can turn to look at cubes and target positions. A look is a wrist-only
head turn (J4/J5, see _gaze.wrist_look_angles); the tip swings a little.

PHASE 1 -- look around
    From HOME the arm goes to the configuration and looks at the cubes (their
    average position), then at the target positions (their average); the
    next look is straight at the first cube (Phase 2, step 1).

PHASE 2 -- one cycle per cube
    1. at the configuration, the head swings to look at the cube it is about
       to grab;
    2. it reaches the cube and carries it to its target in parabolic arcs,
       gazing at where it is going and turning the gripper straight down just
       before it arrives (GAZE_EASE_IN_S / GAZE_EASE_OUT_S, as in
       HighH-LowR_gaze.py);
    3. it returns to the configuration while LOOKING BACK at the target
       position it just left -- then step 1 again for the next cube.
    One cube is NUDGED mid-run (scripted, no sensors): as the arm nears it the
    arm RECOILS, waits for the cube to "settle", then grabs it at its new
    position.

PHASE 3 -- success nod
    After the last cube's return, the head turns straight ahead
    (NOD_WRIST_J456_DEG) and nods up and down with J4 only.

    python3 scripts/Profiles/HighH-LowR_test.py --mock --yes      # no hardware
    python3 scripts/Profiles/HighH-LowR_test.py --port /dev/ttyTHS1

CYCLES
------
N_CYCLES = 2 * (number of cubes). Even cycle 2k = look at + reach cube k and
close; odd cycle 2k+1 = carry cube k to its target, open, and return to the
configuration.

GRIPPER TIMING
--------------
The gripper fires at the END of each reach / carry cycle (a clean pause at
the cube, like a hand), right after the arc finishes -- no background thread.

Everything you tune is a CONSTANT below. The cube coordinates,
PICK_ORIENTATION_DEG and MAX_HEIGHT_TRAJECTORY are PLACEHOLDERS -- measure them
on your arm first.
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
HOME_RETURN_DPS = 35.0            # deg/s -- a big return gets proportionally MORE time so
                                 # move_joints (no speed pre-check) does not outrun the
                                 # servos and shake. Lower if the last homing still shakes.
SETTLE_S = 0.3
PREFLIGHT = False #True
RANDOM_SEED = 1                # int for a repeatable run, None for fresh each time

# -- cubes (MEASURE AND REPLACE) ----------------------------------------------
# (x, y, z) CM, at the GRIPPER TIP, base frame, z from the table.

Z_CUBE_COORD = -4.0

CUBES_INITIAL_POINTS = [          # a row on the pick side
    (14.0, 23.0, Z_CUBE_COORD),
    (14.0, 15.5, Z_CUBE_COORD),
    (14.0, 10.0, Z_CUBE_COORD),
    #(15.0, 19.0, Z_CUBE_COORD),
]
CUBES_INITIAL_POINTS = CUBES_INITIAL_POINTS[::-1]

CUBES_TARGET_POINTS = [           # clean, evenly-spaced drop row, uniform z -- no overshoot
    (14.0, -23.0, Z_CUBE_COORD),
    (14.0, -15.5, Z_CUBE_COORD),
    (14.0, -10.0, Z_CUBE_COORD),
    #(15.0, -19.0, Z_CUBE_COORD),
]
CUBES_TARGET_POINTS = CUBES_TARGET_POINTS[::-1]

# Gripper orientation (rx, ry, rz DEG) held for EVERY move so the gripper stays
# pointing straight down. CALIBRATION: jog to gripper-straight-down, read
# arm.get_coords()[3:]  (this is in the current TOOL frame, config.TOOL_RPY_DEG).
PICK_ORIENTATION_DEG = (180.0, 0.0, -45.0)

# How the gripper YAW (rz) is handled -- rx/ry (pointing-down) are always held:
#   "world" : rz fixed in the base frame (today's behaviour) -- J6 counter-rotates
#             as J1 swings so the gripper keeps the same absolute heading.
#   "base"  : rz follows the tip azimuth atan2(y, x) so J6 stays ~put as J1 turns
#             (gripper heading fixed in J1's rotating frame, not the world's).
#   "free"  : rz unconstrained -- IK keeps wrist motion minimal.
ORIENT_LOCK = "world"
ORIENT_LOCK_SIGN = 1.0            # flip to -1.0 if "base" yaws the gripper the wrong way

# -- gaze (look-at) ----------------------------------------------------------
# When enabled, the gripper tip points AT its current goal instead of holding
# PICK_ORIENTATION_DEG -- the cube while reaching for it, the drop point while
# carrying it there. Supersedes ORIENT_LOCK/_yaw() (and PICK_ORIENTATION_DEG's
# rx/ry) on every arc that has a gaze target. False also turns off every look
# (the head swings / look-back below): the wrist then just stays gripper-down.
# See _gaze.py.
GAZE_ENABLED = True
GAZE_EASE_IN_S = 4.0              # seconds; 0 = snap onto a new target instantly,
                                 # larger = slower lock-on when the gaze target switches
GAZE_EASE_OUT_S = 2.0           # seconds before arrival that the gripper starts leveling
                                 # out to PICK_ORIENTATION_DEG's pitch/roll, so every gazed
                                 # arc still arrives gripper-straight-down; 0 = snap level
                                 # only on the arc's very last waypoint

# -- the LOOK configuration + head swings --------------------------------------
# The arm starts every cube cycle (and Phase 1, and the final nod) in this
# configuration. Only J1 (base), J2 and J3 are given: J4..J6 stay free so the
# head can swing between the cubes and the target positions.
LOOK_CONFIG_J123_DEG = [0.0, 0.0, -90.0]

HEAD_SWING_SPEED_DPS = 60.0      # average speed of a head swing (largest joint travel / time)
HEAD_SWING_EASE_IN = 5.0         # [0,10] acceleration into a swing -- see EASE_IN's doc below
HEAD_SWING_EASE_OUT = 5.0        # [0,10] deceleration out of a swing
HEAD_SWING_WAYPOINTS = 30        # samples per swing
LOOK_PAUSE_S = 0.4               # beat held on each look before the next thing happens
LOOK_MIN_TIP_Z_CM = Z_CUBE_COORD + 1.0   # a look may not swing the tip below this
LOOK_MAX_AIM_ERROR_DEG = 5.0     # preflight fails if the head cannot aim this well from
                                 # the configuration

# Return to the configuration after a drop, looking back at that target the
# whole way (J1..J3 travel to LOOK_CONFIG_J123_DEG, J4/J5 keep aiming).
RETURN_SPEED_DPS = 45.0          # average speed of the return (largest joint travel / time)
RETURN_EASE_IN = 5.0             # [0,10]
RETURN_EASE_OUT = 5.0            # [0,10]
RETURN_WAYPOINTS = 50            # samples along the return

# -- arc + velocity profile --------------------------------------------------
# All arcs are pieces of ONE shared parabola  y = a*x^2 + c  (b = 0, symmetric
# about the chord midpoint). The WIDEST move in the run rises to
# MAX_HEIGHT_TRAJECTORY; every shorter move keeps the same curvature `a` and so
# lifts less: apex_i = MAX_HEIGHT_TRAJECTORY * (chord_i / chord_widest) ** 2.
# Height is measured ABOVE the (possibly sloped, possibly diagonal) chord, and
# "chord" is the HORIZONTAL (xy) distance -- so a straight-down pick barely
# lifts, and moves in any xy direction (incl. right -> front) work unchanged.
# Keep MAX_HEIGHT_TRAJECTORY reachable at PICK_ORIENTATION_DEG: gripper-down the
# arm runs out of reach around world z ~ 17-18 cm near the workspace edge.
MAX_HEIGHT_TRAJECTORY = 15.0
MIN_ARC_HEIGHT_CM = 9.0          # floor, so short moves still clear the table / other cubes --
                                 # also keeps the elbow (J3) from folding past its real limit on
                                 # short, close-to-base carries; 2.0 let some short arcs' shared-
                                 # parabola apex drop to ~2.8cm, which needs J3 well past its real
                                 # hardware limit to hold PICK_ORIENTATION_DEG that low -- confirmed
                                 # via direct IK reproduction of every reach/carry arc in this run.
                                 # 8.0 was enough for the fixed straight-down pose, but left cube0's
                                 # carry arc's J3 within ~0.4 deg of that same limit once GAZE_ENABLED
                                 # adds its own (slightly more tilted) orientation on top -- 9.0 gives
                                 # the gazed pose enough clearance to stay reachable throughout,
                                 # confirmed the same way (plan_coords/ik.solve at every waypoint of
                                 # that arc, plus a full clean run)
CRUISE_SPEED_CM_S = 22.0          # peak tip speed; the ease dials stretch the move time
ARC_TIME_EQUALIZATION = 0.5   # 0..1: blends each reach/carry arc's own duration at
                              # CRUISE_SPEED_CM_S (0 = today, duration grows with arc
                              # length) toward the WIDEST arc's own duration at that
                              # speed (1 = every arc takes exactly that time). The
                              # widest arc's own pace is unchanged either way, so it's
                              # never pushed faster than the already-smooth cruise speed.
EASE_IN = 4.0                    # [0,10] start-of-move acceleration shape. 0 = abrupt,
EASE_OUT = 5.0                   # [0,10] end-of-move deceleration shape.  10 = long, gentle S
PATH_WAYPOINTS = 60              # samples per arc
MIN_SEGMENT_S = 0.02

# -- per-move variation ("never the same twice") -----------------------------
VARIATION = 0                  # [0,1] master scale; 0 = identical arcs every run
APEX_HEIGHT_JITTER_FRAC = 0.0    # +/- fraction of an arc's own apex height
BOW_JITTER_CM = 0.0             # +/- sideways bow, perpendicular to the chord
EASE_JITTER = 0.0              # +/- on EASE_IN / EASE_OUT per move

# -- order ------------------------------------------------------------------------
SHUFFLE_ORDER = False #True             # grab cubes in a random order (init<->target pairing kept)

# -- scripted nudge / flinch ------------------------------------------------------
NUDGE_ENABLED = True             # True enables the scripted nudge (fires on NUDGED_CUBE's
                                 # own reach, wherever pick order puts it)
NUDGED_CUBE = 1                  # 0, 1, or 2 -- which CUBES_INITIAL_POINTS cube gets
                                 # nudged; drives both the scripted recoil target and the
                                 # yellow RViz preview cube
NUDGE_OFFSET_CM = (2.0, 0.0, 0.0)   # where the nudged cube ends up (relative to its point)
NUDGE_AT_FRACTION = 0.8         # fraction of the reach arc completed before the recoil
NUDGE_RECOIL_CM = 4.0           # how far the arm hops back
NUDGE_RECOIL_ARC_HEIGHT_CM = 0.0   # recoil path's apex above its own chord -- small, so
                                 # it reads as a fast near-straight hop, not a lofted arc
NUDGE_RECOIL_EASE_IN = 0.0     # [0,10] recoil-specific ease-in (see EASE_IN doc) -- low, so
                              # the hop reaches NUDGE_RECOIL_SPEED_CM_S almost immediately
                              # instead of spending much of its short travel ramping up
NUDGE_RECOIL_EASE_OUT = 0.3    # [0,10] recoil-specific ease-out -- ditto, slowing into the stop
NUDGE_RECOIL_WAYPOINTS = 5     # fewer than PATH_WAYPOINTS -- get_durations floors a move's
                              # total time at (n_waypoints-1)*MIN_SEGMENT_S regardless of
                              # cruise speed, so the recoil's short hop needs far fewer
                              # segments than a full reach/carry arc to actually reach
                              # NUDGE_RECOIL_SPEED_CM_S instead of being floored near it
NUDGE_RECOIL_SPEED_CM_S = 25.0  # the recoil is fast
POST_NUDGE_ARC_HEIGHT_CM = 2.0   # re-approach-to-the-moved-cube arc's apex above its own
                                # chord -- bypasses _arc_height()'s shared-parabola floor
                                # (MIN_ARC_HEIGHT_CM) so this short hop stays a gentle curve
                                # toward the cube instead of a full lift-and-descend peak
POST_NUDGE_SPEED_CM_S = 12.0    # re-approach cruise speed -- its own dial, independent of
                                # CRUISE_SPEED_CM_S / ARC_TIME_EQUALIZATION (which every
                                # normal arc uses), so it can be tuned without also
                                # speeding up the rest of the run
POST_NUDGE_WAYPOINTS = 10       # fewer than PATH_WAYPOINTS -- same reasoning as
                                # NUDGE_RECOIL_WAYPOINTS: get_durations floors a move's total
                                # time at (n_waypoints-1)*MIN_SEGMENT_S regardless of cruise
                                # speed, so this short re-approach needs far fewer segments
                                # than a full reach/carry arc to actually reach
                                # POST_NUDGE_SPEED_CM_S instead of being floored well below it
NUDGE_RECOIL_JERK = 0.0        # brief arm.jerk on the recoil for a startled look (0 = clean)
NUDGE_SETTLE_S = 1.5           # pause after the recoil, "waiting for the cube to stop"

# where the nudged cube visually ends up -- always fed to RvizBridge as the yellow
# preview cube, independent of whether nudging is enabled this run
NUDGE_CUBE_PREVIEW_CM = tuple(
    float(c + o) for c, o in zip(CUBES_INITIAL_POINTS[NUDGED_CUBE], NUDGE_OFFSET_CM)
)

# -- success animation: a head NOD (Phase 3) ---------------------------------
# Played at the configuration once the last cube is placed: the head first
# swings to NOD_WRIST_J456_DEG, then nods with J4 only.
NOD_ENABLED = True
NOD_WRIST_J456_DEG = [90.0, 0.0, 0.0]   # wrist pose the nod is centred on. With the default
                                        # configuration J4 = +90 points the gripper straight
                                        # ahead (+X)
NOD_SPEED_DPS = 40.0          # AVERAGE J4 speed of each stroke, deg/s (peak is higher
                              # with strong ease)
NOD_EASE_IN = 5.0             # [0,10] acceleration into each stroke -- see EASE_IN's doc above
NOD_EASE_OUT = 5.0            # [0,10] deceleration out of each stroke -- see EASE_OUT's doc above
NOD_START_DIRECTION = -1      # +1 = the first nod goes UP, -1 = it goes DOWN
NOD_UP_COUNT = 2              # number of up nods
NOD_DOWN_COUNT = 2            # number of down nods. Nods alternate from NOD_START_DIRECTION
                              # until both counts are used up: opposite nods run extreme to
                              # extreme through the centre; once only one direction is
                              # left the head returns to the centre between repeats.
                              # Always ends back at the centre
NOD_UP_DEG = 15.0             # how far J4 tilts UP from the centre on an up nod
NOD_DOWN_DEG = 15.0           # how far it tilts DOWN on a down nod
NOD_UP_J4_SIGN = 1.0          # +1: increasing J4 tilts the gripper up (true for the
                              # default pose); flip to -1 if it nods the wrong way
NOD_WAYPOINTS_PER_STROKE = 20  # samples per stroke (one stroke = one key angle to the next)
NOD_BLEND_START_FRAC = 0.5    # on the LAST return: fraction of the way back after which the
                              # head stops looking at the target and turns toward
                              # NOD_WRIST_J456_DEG, arriving in the nod pose. 0 = from the
                              # start, 1 = never (return first, then a separate swing)

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 120.0           # open, to approach / release a cube
                                # (0 = shut .. config.MAX_GRIPPER_DEG = full open)
GRIP_CLOSED_DEG = 65.0          # closed ON a cube -- tune to the cube width
GRIP_LOOK_DEG = 10.0            # nearly shut: while looking around, on the way back to the
                                # look configuration, and for the nod
GRIP_SPEED = 10  #config.GRIPPER_DEFAULT_SPEED
GRIP_SETTLE_S = 0.35           # quiet time after a gripper command: it must LAND and the
                              # jaws start moving. Tunable down to GRIP_MIN_GAP_S, not below.
GRIP_MIN_GAP_S = 0.2          # hard floor -- pymycobot silently drops a gripper command
                              # that is not followed by a short quiet gap (why 0.0 failed).
GRIP_CLOSE_WAIT_S = 1.0        # the arm stays still this long after closing on a cube, so the
                              # jaws have finished before it lifts (depends on GRIP_SPEED)
GRIP_OPEN_WAIT_S = 1.0         # ... and after releasing a cube at its target
REACH_TOL_CM = 3.0             # has_reached_* tolerance, per axis

# -- gripper WHILE the arm moves (what this test variant is about) --------------
GRIP_OPEN_DELAY_S = 0.0          # seconds after a reach STARTS that the gripper begins to open
GRIP_CLOSE_DELAY_S = 0.6         # seconds after the return to the look configuration STARTS
                                 # that the gripper begins to close to GRIP_LOOK_DEG -- late
                                 # enough to be clear of the cube it just dropped
GRIP_MOVING_REPEATS = 2          # how many times a gripper command fired during motion is
                                 # sent. armik writes it without waiting for a reply and keeps
                                 # config.MIN_COMMAND_GAP_S of quiet around it (a blocking send
                                 # stalled the stream 0.5-1.6 s on the arm; a write <1 ms after
                                 # a setpoint was sometimes ignored). The repeat is a second
                                 # line of defence: raise it if the gripper still misses
GRIP_MOVING_REPEAT_GAP_S = 0.1   # gap between those sends

N_CYCLES = len(CUBES_INITIAL_POINTS) * 2

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


def _send_arc(arm, pts, durs, label, gaze_target=None, grip=None):
    """Parabolic move. pts[0] is the implicit start (not sent).
    gaze_target: if given (and GAZE_ENABLED), the gripper tip points at this
    3D point for the whole arc instead of holding PICK_ORIENTATION_DEG. A wide
    carry occasionally asks for a look-at pose this arm's elbow/wrist can't
    reach (or can only reach too fast) -- if the gazed plan is REFUSED, this
    falls back to the fixed PICK_ORIENTATION_DEG for THIS arc only, rather
    than aborting the run.
    grip: optional (deg, delay_s, label) gripper command fired while the arc
    runs.
    Plans with arm.plan_path() and hands the Plan to _do_plan(), so the same
    code serves the rehearsal (record) and a live move (execute)."""
    if len(pts) < 2:
        _say(f"  {label}: negligible, skipped")
        return True
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


def _eased_joint_move(arm, q_waypoints, speed_dps, ease_in, ease_out, label, grip=None):
    """One eased joint-space stroke from the arm's current pose along
    q_waypoints (rows evenly spaced along the stroke; the last row is the
    goal). Takes (largest total joint travel) / speed_dps seconds, shaped by
    the [0,10] ease dials like every arc (celebration_durations).

    The path is SAMPLED IN TIME, one setpoint per control tick
    (config.CONTROL_RATE_HZ) -- q_waypoints only define the path's shape, not
    how many setpoints are sent. Sending them as-is streamed short moves far
    faster than the serial link can carry (a 0.15 s head swing as 30 setpoints
    5 ms apart), which made the arm stutter.

    Direct arm._execute(), which applies no hardware speed check of its own --
    so the timeline is stretched (never sped up) to keep every joint within
    MAX_JOINT_SPEED_DPS. grip: optional (deg, delay_s, label)
    gripper command fired while the stroke runs (sent at once if there is
    nothing to move). Returns bool."""
    q0 = np.asarray(arm.get_angles(), dtype=float)
    path = np.vstack([q0[None, :], np.asarray(q_waypoints, dtype=float)])
    travel = float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0)))
    if travel < 0.5:
        if grip is not None:
            _do_plan(arm, None, grip)
        return True
    for r in path[1:]:
        problems = check_joint_limits(r)
        if problems:
            _say(f"  {label} REFUSED -- violates joint limits: {'; '.join(problems)}")
            return False

    # eased progress s(t): celebration_durations gives the time of evenly spaced
    # progress steps; invert it to read progress at evenly spaced TIMES
    s_grid = np.linspace(0.0, 1.0, 201)
    t_grid = np.concatenate([[0.0], np.cumsum(celebration_durations(ease_in, ease_out, 1.0, 201))])
    t_grid /= t_grid[-1]
    idx = np.arange(len(path), dtype=float)

    def _sample(total_s):
        n = max(1, int(total_s * config.CONTROL_RATE_HZ))      # floor: never faster than the rate
        t = np.linspace(0.0, total_s, n + 1)
        u = np.interp(t / total_s, t_grid, s_grid) * (len(path) - 1)
        rows = np.column_stack([np.interp(u, idx, path[:, j]) for j in range(path.shape[1])])
        rows[0], rows[-1] = path[0], path[-1]
        return t, rows

    total_s = travel / max(float(speed_dps), 1e-6)
    t, rows = _sample(total_s)
    min_s = min_feasible_duration_s(rows, np.diff(t) / total_s)
    if min_s > total_s + 1e-6:
        total_s = min_s
        t, rows = _sample(total_s)

    plan = Plan(ok=True, q_waypoints=rows, timestamps=t, duration_s=float(t[-1]))
    ex = _do_plan(arm, plan, grip)
    if not ex.ok:
        _say(f"  {label} REFUSED -- {ex.error}")
        return False
    if ex.late_deadlines:
        _say(f"  !! {label}: {ex.late_deadlines} late control-loop deadline(s)")
    return True


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
    """Phase 1: go to the look configuration and look at the cubes (their
    average position), then the target positions (their average). The look
    that follows is the first cycle's own, straight at the first cube."""
    cubes = tuple(float(v) for v in np.mean(np.asarray(CUBES_INITIAL_POINTS, float), axis=0))
    targets = tuple(float(v) for v in np.mean(np.asarray(CUBES_TARGET_POINTS, float), axis=0))
    for target, label in ((cubes, "look at the cubes"),
                          (targets, "look at the targets")):
        if not _swing_head(arm, target, label):
            return False
    return True


def _return_looking_back(arm, target, end_wrist=None):
    """After a drop: J1..J3 travel back to LOOK_CONFIG_J123_DEG (J6 back to
    HOME's) while the head keeps looking at `target`, the place just left --
    see _gaze.wrist_track_angles.
    end_wrist: optional [J4, J5, J6] to ARRIVE in (the nod pose, on the last
    return). The head then looks at the target only until NOD_BLEND_START_FRAC
    of the way back and cross-fades smoothly to end_wrist over the rest.
    Returns bool."""
    q0 = np.asarray(arm.get_angles(), dtype=float)
    q_end = q0.copy()
    q_end[:3] = LOOK_CONFIG_J123_DEG
    q_end[5] = HOME[5]
    if GAZE_ENABLED:
        rows = _line_to(q0, q_end, RETURN_WAYPOINTS)       # J4/J5 held, then re-aimed per row
        floor_cm = current_pos(arm)[2] - 0.2               # never dip below the drop height
        rows = wrist_track_angles(np.vstack([q0[None, :], rows]), target, floor_cm)[1:]
    else:
        q_end[3:5] = HOME[3:5]
        rows = _line_to(q0, q_end, RETURN_WAYPOINTS)
    blended = end_wrist is not None and NOD_BLEND_START_FRAC < 1.0
    if blended:
        frac = np.arange(1, len(rows) + 1) / len(rows)
        w = _smootherstep((frac - NOD_BLEND_START_FRAC) / (1.0 - NOD_BLEND_START_FRAC))
        rows[:, 3:6] += w[:, None] * (np.asarray(end_wrist, dtype=float)[None, :] - rows[:, 3:6])
    # close the gripper on the way, once clear of the cube just dropped
    if not _eased_joint_move(arm, rows, RETURN_SPEED_DPS, RETURN_EASE_IN, RETURN_EASE_OUT,
                             "return", grip=(GRIP_LOOK_DEG, GRIP_CLOSE_DELAY_S, "close")):
        return False
    _say("  returned to the look configuration"
         + (", looking back at the target" if GAZE_ENABLED else "")
         + (", then turning to the nod pose" if blended else ""))
    return True


def _nod_key_angles():
    """J4 key angles (deg) of the nod, starting and ending at the nod centre
    (NOD_WRIST_J456_DEG's J4) -- see the NOD_* constants for the sequencing
    rules."""
    centre = float(NOD_WRIST_J456_DEG[0])
    extreme = {"up": centre + NOD_UP_J4_SIGN * NOD_UP_DEG,
               "down": centre - NOD_UP_J4_SIGN * NOD_DOWN_DEG}
    other = {"up": "down", "down": "up"}
    left = {"up": int(NOD_UP_COUNT), "down": int(NOD_DOWN_COUNT)}
    nxt = "up" if NOD_START_DIRECTION > 0 else "down"
    keys, prev = [centre], None
    while left["up"] + left["down"] > 0:
        if left[nxt] == 0:
            nxt = other[nxt]
        if nxt == prev:                       # same direction again -> back to centre first
            keys.append(centre)
        keys.append(extreme[nxt])
        left[nxt] -= 1
        prev, nxt = nxt, other[nxt]
    keys.append(centre)
    return keys


def _play_nod(arm):
    """Phase 3: at the look configuration, swing the head to
    NOD_WRIST_J456_DEG, then nod -- J4 only, each stroke eased by
    NOD_EASE_IN / NOD_EASE_OUT at NOD_SPEED_DPS."""
    base = np.asarray(list(LOOK_CONFIG_J123_DEG) + list(NOD_WRIST_J456_DEG), dtype=float)
    if not _eased_joint_move(arm, _line_to(arm.get_angles(), base, HEAD_SWING_WAYPOINTS),
                             HEAD_SWING_SPEED_DPS, HEAD_SWING_EASE_IN, HEAD_SWING_EASE_OUT,
                             "nod pose"):
        _say("  nod skipped -- could not reach the nod pose")
        return True
    _pause(LOOK_PAUSE_S)
    if not NOD_ENABLED:
        return True

    keys = _nod_key_angles()
    _say(f"  nodding: J4 {' -> '.join(f'{k:.0f}' for k in keys)} deg")
    for a, b in zip(keys[:-1], keys[1:]):
        if abs(b - a) < 1e-6:
            continue
        q_a, q_b = base.copy(), base.copy()
        q_a[3], q_b[3] = a, b
        if not _eased_joint_move(arm, _line_to(q_a, q_b, NOD_WAYPOINTS_PER_STROKE),
                                 NOD_SPEED_DPS, NOD_EASE_IN, NOD_EASE_OUT, "nod"):
            break
    return True


def run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs, d_max, cruise_dur_max,
              grip=None):
    """Scripted flinch: approach part-way, recoil, wait, re-approach the moved cube."""
    n = len(pts)
    cut = max(2, int(round(NUDGE_AT_FRACTION * (n - 1))) + 1)
    gaze = seg["gaze"]                 # the original cube -- kept through approach + recoil
    _say(f"  NUDGE: approaching to {int(NUDGE_AT_FRACTION*100)}% ...")
    if not _send_arc(arm, pts[:cut], durs[:cut - 1], "  nudge approach", gaze_target=gaze,
                     grip=grip):
        return False

    here = current_pos(arm)
    travel = np.asarray(here, float) - np.asarray(pts[0], float)
    dirn = travel / (np.linalg.norm(travel) + 1e-9)
    recoil = tuple(float(v) for v in (
        np.asarray(here, float) - dirn * NUDGE_RECOIL_CM
        + np.array([0.0, 0.0, NUDGE_RECOIL_CM * 0.5])
    ))

    _say(f"  RECOIL -> {tuple(round(v, 1) for v in recoil)}")
    rpts = get_path(here, recoil, NUDGE_RECOIL_ARC_HEIGHT_CM, rng,
                    n_waypoints=NUDGE_RECOIL_WAYPOINTS)
    rdurs = get_durations(here, recoil, NUDGE_RECOIL_ARC_HEIGHT_CM,
                          NUDGE_RECOIL_EASE_IN, NUDGE_RECOIL_EASE_OUT,
                          cruise=NUDGE_RECOIL_SPEED_CM_S,
                          n_waypoints=NUDGE_RECOIL_WAYPOINTS)
    arm.jerk = NUDGE_RECOIL_JERK
    ok = _send_arc(arm, rpts, rdurs, "  recoil", gaze_target=gaze)
    arm.jerk = 0.0
    if not ok:
        return False

    _say(f"  waiting {NUDGE_SETTLE_S:.1f}s for the cube to settle ...")
    _pause(NUDGE_SETTLE_S)

    new_cube = tuple(float(c + o) for c, o in zip(seg["target"], NUDGE_OFFSET_CM))
    _say(f"  cube moved -> re-approaching {tuple(round(v, 1) for v in new_cube)}")
    after = current_pos(arm)
    h2 = POST_NUDGE_ARC_HEIGHT_CM
    p2 = get_path(after, new_cube, h2, rng, n_waypoints=POST_NUDGE_WAYPOINTS)
    d2 = get_durations(after, new_cube, h2, EASE_IN, EASE_OUT,
                       cruise=POST_NUDGE_SPEED_CM_S, n_waypoints=POST_NUDGE_WAYPOINTS)
    if not _send_arc(arm, p2, d2, "  nudge re-approach", gaze_target=new_cube):
        return False

    # the following carry cycle must start from where the cube actually is now
    nxt = ci + 1
    if nxt < len(segments) and segments[nxt]["kind"] == "carry":
        segments[nxt]["origin"] = new_cube
        hc = _arc_height(_chord_len(new_cube, segments[nxt]["target"]), d_max)
        paths[nxt] = get_path(new_cube, segments[nxt]["target"], hc, rng)
        cruise_durc = sum(get_durations(new_cube, segments[nxt]["target"], hc, EASE_IN, EASE_OUT))
        tc = cruise_durc * (1.0 - ARC_TIME_EQUALIZATION) + cruise_dur_max * ARC_TIME_EQUALIZATION
        all_durs[nxt] = get_durations(new_cube, segments[nxt]["target"], hc, EASE_IN, EASE_OUT,
                                      duration=tc)
    return True


# ===========================================================================
# PREFLIGHT
# ===========================================================================

def preflight(arm, segments, paths):
    rx0, ry0 = PICK_ORIENTATION_DEG[:2]
    print("\n--- preflight: planning every cube point + arc apex (no motion) ---")
    # (name, (x,y,z), gaze_target_or_None) -- direct cube/target points keep the
    # fixed-down probe (point == target there, so "look at it" is undefined; the
    # fixed pose is a sufficient reachability proxy for arrival anyway).
    checks = []
    for i, (s, t) in enumerate(zip(CUBES_INITIAL_POINTS, CUBES_TARGET_POINTS)):
        checks += [(f"init{i+1}", s, None), (f"tgt{i+1}", t, None)]
    if NUDGE_ENABLED:
        # the nudged cube also gets grabbed at init + offset
        for i, s in enumerate(CUBES_INITIAL_POINTS):
            checks.append((f"init{i+1}+nudge",
                           tuple(c + o for c, o in zip(s, NUDGE_OFFSET_CM)), None))
    for ci, (seg, pts) in enumerate(zip(segments, paths)):
        apex = max(pts, key=lambda p: p[2])         # highest point of the arc
        checks.append((f"apex c{ci}", apex, seg["gaze"]))
    bad = 0
    for name, (x, y, z), gaze in checks:
        if GAZE_ENABLED and gaze is not None:
            rx, ry, rz = look_at_rpy((x, y, z), gaze)
        else:
            rx, ry, rz = rx0, ry0, _yaw_one((x, y))
        pl = arm.plan_coords(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                             speed=config.DEFAULT_SPEED_CM_S)
        err = (pl.error or "").lower()
        if pl.ok:
            print(f"  OK  {name:14s} ({x:5.1f},{y:6.1f},{z:4.1f})  "
                  f"peak {pl.peak_joint_dps:.0f} deg/s")
        elif "already at" in err:
            # planning a move to the current pose -- reachable, just no motion
            print(f"  OK  {name:14s} ({x:5.1f},{y:6.1f},{z:4.1f})  (already there)")
        else:
            print(f"  BAD {name:14s} ({x:5.1f},{y:6.1f},{z:4.1f})  {pl.error}")
            bad += 1
    print(f"--- preflight: {len(checks) - bad}/{len(checks)} reachable ---")

    # every look must be possible from LOOK_CONFIG_J123_DEG with the wrist alone
    if GAZE_ENABLED:
        looks = [("cubes avg", tuple(np.mean(np.asarray(CUBES_INITIAL_POINTS, float), axis=0))),
                 ("targets avg", tuple(np.mean(np.asarray(CUBES_TARGET_POINTS, float), axis=0)))]
        looks += [(f"init{i+1}", s) for i, s in enumerate(CUBES_INITIAL_POINTS)]
        looks += [(f"tgt{i+1}", t) for i, t in enumerate(CUBES_TARGET_POINTS)]
        q_cfg = list(LOOK_CONFIG_J123_DEG) + list(HOME[3:])
        bad_looks = 0
        for name, tgt in looks:
            q_look, err = wrist_look_angles(q_cfg, tgt, LOOK_MIN_TIP_Z_CM)
            if q_look is None or err > LOOK_MAX_AIM_ERROR_DEG:
                print(f"  BAD look {name:12s} " + ("no admissible head pose" if q_look is None
                                                  else f"aim error {err:.1f} deg"))
                bad_looks += 1
            else:
                print(f"  OK  look {name:12s} aim error {err:.1f} deg")
        print(f"--- preflight: {len(looks) - bad_looks}/{len(looks)} looks possible from "
              f"LOOK_CONFIG_J123_DEG ---")
        bad += bad_looks
    return bad == 0


# ===========================================================================
# MAIN
# ===========================================================================

def _build_segments(order, look_tip):
    """Reach / carry pairs. Every reach starts from the look configuration
    (nominally look_tip -- the real start depends on where the head points)."""
    segs = []
    for k in order:
        segs.append({"kind": "reach", "origin": look_tip,
                     "target": CUBES_INITIAL_POINTS[k], "k": int(k),
                     "gaze": CUBES_INITIAL_POINTS[k]})
        segs.append({"kind": "carry", "origin": CUBES_INITIAL_POINTS[k],
                     "target": CUBES_TARGET_POINTS[k], "k": int(k),
                     "gaze": CUBES_TARGET_POINTS[k]})
    return segs


def _choreography(arm, segments, paths, all_durs, d_max, cruise_dur_max, rng):
    """The whole run -- Phase 1, every cube cycle, Phase 3 -- written against
    whatever `arm` it is given. main() runs it once on a simulated arm with
    _TAPE recording (the rehearsal) and then plays the tape back on the real
    one. Returns False if anything was refused."""
    if not _grip(arm, GRIP_LOOK_DEG, "for looking around"):
        return False

    _say("\n=== phase 1: looking around ===")
    if not _phase_look_around(arm):
        return False

    for ci, seg in enumerate(segments):
        kind = seg["kind"]

        grip_deg = GRIP_CLOSED_DEG if kind == "reach" else GRIP_OPEN_DEG
        label = "reach & grasp" if kind == "reach" else "carry & place"
        _say(f"\n=== cycle {ci}/{N_CYCLES - 1}  {label}  "
              f"cube #{seg['k'] + 1}  -> {tuple(round(v, 1) for v in seg['target'])} ===")

        # a reach starts by looking at the cube from the look configuration;
        # either way the arc starts from wherever the tip actually is
        if kind == "reach" and not _swing_head(arm, seg["target"], "look at the cube"):
            return False
        pts, durs = _arc_from(current_pos(arm), seg["target"], d_max, cruise_dur_max, rng)
        # the gripper opens WHILE the arm travels to the cube
        open_grip = (GRIP_OPEN_DEG, GRIP_OPEN_DELAY_S, "open") if kind == "reach" else None

        if kind == "reach" and NUDGE_ENABLED and seg["k"] == NUDGED_CUBE:
            if not run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs, d_max,
                             cruise_dur_max, grip=open_grip):
                return False
            if not _grip(arm, GRIP_CLOSED_DEG, "close on cube (new position)",
                         wait_s=GRIP_CLOSE_WAIT_S):
                return False
            continue

        if not _send_arc(arm, pts, durs, label, gaze_target=seg["gaze"], grip=open_grip):
            return False

        # grip IMMEDIATELY -- nothing (no position read, no extra round
        # trip) runs between the arm stopping and the gripper command.
        # ... then the arm WAITS for the jaws to finish before it moves off.
        if not _grip(arm, grip_deg, "close on cube" if kind == "reach" else "release cube",
                     wait_s=GRIP_CLOSE_WAIT_S if kind == "reach" else GRIP_OPEN_WAIT_S):
            return False

        # live check at playback: did the tip really get there?
        _TAPE.append(("check", "cube" if kind == "reach" else "target", tuple(seg["target"])))

        # back to the look configuration, looking at the target just left; the
        # LAST return also turns the head into the nod pose on the way
        last = ci == len(segments) - 1
        if kind == "carry" and not _return_looking_back(
                arm, seg["target"], end_wrist=NOD_WRIST_J456_DEG if last else None):
            return False

    _say("\nall cubes placed.")
    _say("\n=== phase 3: success nod ===")
    _play_nod(arm)
    return True


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
    if not args.mock and not args.yes:
        print("This will move the robot arm and actuate the gripper. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(CUBES_INITIAL_POINTS)
    order = list(rng.permutation(n)) if SHUFFLE_ORDER else list(range(n))

    p_home = pose_coords(HOME)                       # mm/deg (Z_RELATIVE_TO_JOINT1 assumed False)
    home_tip = (p_home[0] / 10.0, p_home[1] / 10.0, p_home[2] / 10.0)

    global _AZ_REF                                   # rz = PICK_ORIENTATION_DEG[2] at HOME's azimuth
    _AZ_REF = (home_tip[0], home_tip[1])

    print(f"pick order (cube indices): {[int(k) for k in order]}")
    print(f"orientation lock: {ORIENT_LOCK}")
    if NUDGE_ENABLED:
        print(f"NUDGE enabled: cube #{NUDGED_CUBE + 1} "
              f"at {CUBES_INITIAL_POINTS[NUDGED_CUBE]} -- nudge THAT cube as soon as "
              f"its own reach comes up")

    p_look = pose_coords(list(LOOK_CONFIG_J123_DEG) + list(HOME[3:]))
    look_tip = (p_look[0] / 10.0, p_look[1] / 10.0, p_look[2] / 10.0)
    segments = _build_segments(order, look_tip)

    # the widest move sets the shared parabola; every shorter arc lifts less
    d_max = max((_chord_len(s["origin"], s["target"]) for s in segments), default=1.0) or 1.0
    print(f"widest move {d_max:.1f} cm -> apex {MAX_HEIGHT_TRAJECTORY:.1f} cm  "
          f"(shared parabola a = {-MAX_HEIGHT_TRAJECTORY / (d_max / 2.0) ** 2:.4f})")

    # ARC_TIME_EQUALIZATION blends each reach/carry arc's own CRUISE_SPEED_CM_S
    # duration toward the WIDEST arc's own duration at that speed
    cruise_dur_max = 0.0
    for seg in segments:
        h = _arc_height(_chord_len(seg["origin"], seg["target"]), d_max)
        cruise_dur_max = max(cruise_dur_max,
                             sum(get_durations(seg["origin"], seg["target"], h, EASE_IN, EASE_OUT)))
    print(f"widest reach/carry arc takes {cruise_dur_max:.2f}s at CRUISE_SPEED_CM_S "
          f"(ARC_TIME_EQUALIZATION={ARC_TIME_EQUALIZATION:.2f} blends every arc toward this)")

    # ---- precompute every arc + its durations --------------------------------
    paths, all_durs = [], []
    for seg in segments:
        ei = EASE_IN + float(rng.uniform(-1.0, 1.0)) * EASE_JITTER * VARIATION
        eo = EASE_OUT + float(rng.uniform(-1.0, 1.0)) * EASE_JITTER * VARIATION
        h = _arc_height(_chord_len(seg["origin"], seg["target"]), d_max)
        paths.append(get_path(seg["origin"], seg["target"], h, rng))
        cruise_dur = sum(get_durations(seg["origin"], seg["target"], h, ei, eo))
        t_i = cruise_dur * (1.0 - ARC_TIME_EQUALIZATION) + cruise_dur_max * ARC_TIME_EQUALIZATION
        all_durs.append(get_durations(seg["origin"], seg["target"], h, ei, eo, duration=t_i))

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)

    bridge = None
    if args.rviz:
        from _rviz_bridge import RvizBridge
        flat_path = [tuple(map(float, p)) for seg_pts in paths for p in seg_pts]
        cube_pts = [tuple(map(float, c)) for c in CUBES_INITIAL_POINTS]
        try:
            bridge = RvizBridge(arm.get_angles, path_xyz_cm=flat_path,
                                cube_points=cube_pts, nudge_cube_point=NUDGE_CUBE_PREVIEW_CM,
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

        if PREFLIGHT and not preflight(arm, segments, paths):
            print("\npreflight failed -- fix the cube coordinates or PICK_ORIENTATION_DEG. "
                  "Nothing moved.")
            go_home(arm)
            return 1

        # ---- rehearsal: compute EVERYTHING now, on a simulated arm ----------
        global _TAPE
        print("\nrehearsing the whole run on a simulated arm (nothing moves yet)...")
        t_rehearse = time.perf_counter()
        sim = Arm(mock=True)
        sim.conn.raw.set_angles_directly([float(v) for v in HOME])
        _TAPE = []
        try:
            ok = _choreography(sim, segments, paths, all_durs, d_max, cruise_dur_max, rng)
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
