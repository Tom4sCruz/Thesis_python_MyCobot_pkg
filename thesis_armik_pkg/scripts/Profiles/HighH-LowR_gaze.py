#!/usr/bin/env python3
"""
MOVEMENT PROFILE: High-human / Low-robot  (parabolic rework)
==========================================================

Move cubes from one side of the frame to the other, the way a person doing it
casually would:

  * every move is a smooth PARABOLIC arc -- accelerate out of the start,
    decelerate into the end;
  * no two arcs are identical -- the apex height / position / a slight sideways
    bow are jittered per move (VARIATION);
  * the cubes are grabbed in a RANDOM order;
  * they are dropped at fixed, deliberately uneven points (CUBES_TARGET_POINTS --
    you bake the "looks like it over/undershot" appearance straight into those
    coordinates);
  * one cube is NUDGED mid-run: as the arm nears it, the arm RECOILS (a quick,
    startled hop backwards), waits for the cube to "settle", then grabs it at
    its new position. This is fully scripted -- the arm has no sensors.

    python3 scripts/Profiles/HighH-LowR.py --mock --yes      # no hardware
    python3 scripts/Profiles/HighH-LowR.py --port /dev/ttyTHS1

CYCLES
------
N_CYCLES = 2 * (number of cubes). Even cycle 2k = reach cube k and close;
odd cycle 2k+1 = carry cube k to its target and open. A final lead-out arc
returns to HOME.

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
import time

import numpy as np

from armik import Arm, Plan, config, pose_coords
from _gaze import gaze_waypoints, gaze_then_level_waypoints, look_at_rpy, ease_to_rpy
from _celebrate import (build_straight_arm_oscillation_waypoints, celebration_durations,
                        min_feasible_duration_s, solve_j6_level)

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
# rx/ry) on every arc that has a gaze target; the lead-out back to HOME has
# none and keeps the old fixed orientation. See _gaze.py.
GAZE_ENABLED = True
GAZE_EASE_IN_S = 4.0              # seconds; 0 = snap onto a new target instantly,
                                 # larger = slower lock-on when the gaze target switches
GAZE_EASE_OUT_S = 2.0           # seconds before arrival that the gripper starts leveling
                                 # out to PICK_ORIENTATION_DEG's pitch/roll, so every gazed
                                 # arc still arrives gripper-straight-down; 0 = snap level
                                 # only on the arc's very last waypoint

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
LEADOUT_SPEED_CM_S = 15.0        # the final arc back toward HOME is slower / gentler
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

# -- success animation (played after the last cube is placed, replaces the
# -- old lead-out-to-HOME ending entirely) -----------------------------------
# A gentle dip-and-rise: base (J1) fixed at 90deg, reference pose dead
# straight (J2=J3=J4=0) so the gripper points EXACTLY at world +X with no
# trade-off, and the shoulder/elbow/wrist-pitch chain moves only in the
# world YZ-plane. J4 (wrist-pitch) is STATIC -- it doesn't compensate
# anything (an earlier version had it chase shoulder+elbow's motion every
# waypoint, exact in simulation but unable to track that fast/precisely on
# real hardware). Shoulder+elbow (J2, J3) solve directly for the TIP's own
# position on the vertical line; J6 (tool-roll) keeps the gripper level --
# see _celebrate.py for why that's a different (and, here, exact) job from
# what J4 used to do. Direct joint-space motion; the ordinary Cartesian
# send_path()/plan_coords() can't do this (the straight pose is a real
# kinematic singularity).
CELEBRATE_ENABLED = False
CELEBRATE_BASE_J1_DEG = 90.0        # world azimuth the arm swings to first;
                                     # flip to -90 if it should face the other way
CELEBRATE_J4_STATIC_DEG = 0.0       # wrist-pitch -- held fixed, does not compensate
CELEBRATE_STAGING_J5_DEG = -90.0    # this reference pose gives EXACT +X pointing
                                     # with no trade-off at all (tip lands ~203mm
                                     # off the vertical line, but that's just the
                                     # fixed "head and neck" length laid out along
                                     # +X, not a tunable compromise -- see _celebrate.py)
CELEBRATE_LEVEL_AXIS_INDEX = 0      # which tool-frame axis (0 or 1) J6 keeps level --
                                     # whichever matches the real gripper's physical
                                     # finger-open/width direction; flip to 1 if the
                                     # gripper levels sideways instead of upright
CELEBRATE_STAGING_DURATION_S = 3.0  # move_joints() time into the straight pose;
                                     # 2.0s peaked ~126 deg/s on J1 (comparable to
                                     # the fastest reach/carry arcs), read as abrupt
                                     # for a move meant to look deliberate
CELEBRATE_DIP_MM = 30.0              # how far the gripper dips below the reference
                                     # height each swing -- PLACEHOLDER; the reference
                                     # sits at this sub-chain's full-extension
                                     # singularity, so it can only dip DOWN, and small
                                     # dips need disproportionately large J2/J3 swings
                                     # (square-root-type relation near full extension)
CELEBRATE_FINAL_DIP_MM = 40.0        # "slightly bent" resting dip, instead of
                                     # snapping back fully straight
CELEBRATE_ELBOW_BRANCH_SIGN = 1.0   # flip to -1.0 if the elbow bends the visually
                                     # wrong way
CELEBRATE_CYCLES = 2                # number of full dip-and-rise oscillations
CELEBRATE_EASE_IN = 1.0             # [0,10] -- see EASE_IN's doc above
CELEBRATE_EASE_OUT = 1.0            # [0,10] -- see EASE_OUT's doc above
CELEBRATE_OSCILLATE_WAYPOINTS = 40  # samples across all CELEBRATE_CYCLES
CELEBRATE_SETTLE_WAYPOINTS = 20     # samples for the final eased settle
CELEBRATE_DURATION_S = 5.0          # total time, staging move excluded

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 120.0           # 0 = closed .. config.MAX_GRIPPER_DEG = full open
GRIP_CLOSED_DEG = 65.0          # tune to the cube width
GRIP_SPEED = 90  #config.GRIPPER_DEFAULT_SPEED
GRIP_SETTLE_S = 0.35           # quiet time after a gripper command: it must LAND and the
                              # jaws start moving. Tunable down to GRIP_MIN_GAP_S, not below.
GRIP_MIN_GAP_S = 0.2          # hard floor -- pymycobot silently drops a gripper command
                              # that is not followed by a short quiet gap (why 0.0 failed).
REACH_TOL_CM = 3.0             # has_reached_* tolerance, per axis
LEADOUT_PAUSE_S = 0.5         # deliberate beat between the last release and homing

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

def go_home(arm):
    """Homing move, with the duration scaled to the joint distance so a long
    return from the far side is not crammed into HOME_MOVE_S (which makes
    move_joints -- no speed pre-check -- outrun the servos and shake).
    Returns bool -- move_joints() can refuse (e.g. the arm's current pose is
    already outside a joint's soft limit), and that must not pass silently."""
    try:
        dq = max(abs(a - b) for a, b in zip(arm.get_angles(), HOME))
    except Exception:
        dq = 0.0
    dur = max(HOME_MOVE_S, dq / HOME_RETURN_DPS)
    if dur > HOME_MOVE_S + 0.05:
        print(f"  homing over {dur:.1f}s (joint travel {dq:.0f} deg)")
    if not arm.move_joints(HOME, duration=dur):
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


def _grip(arm, deg, label):
    print(f"  gripper -> {deg:.0f} deg ({label})")
    if not _fire_gripper(arm, deg):
        print(f"  send_gripper REFUSED -- {arm.last_error}")
        return False
    if GRIP_SETTLE_S < GRIP_MIN_GAP_S:
        print(f"  (GRIP_SETTLE_S {GRIP_SETTLE_S}s < floor {GRIP_MIN_GAP_S}s -- using the floor)")
    time.sleep(max(max(GRIP_SETTLE_S, GRIP_MIN_GAP_S) - 0.06, 0.0))
    return True


def _send_arc(arm, pts, durs, label, gaze_target=None):
    """Blocking parabolic move. pts[0] is the implicit start (not sent).
    gaze_target: if given (and GAZE_ENABLED), the gripper tip points at this
    3D point for the whole arc instead of holding PICK_ORIENTATION_DEG. A wide
    carry occasionally asks for a look-at pose this arm's elbow/wrist can't
    reach (or can only reach too fast) -- if the gazed send_path is REFUSED,
    this falls back to the fixed PICK_ORIENTATION_DEG for THIS arc only,
    rather than aborting the run."""
    if len(pts) < 2:
        print(f"  {label}: negligible, skipped")
        return True
    tail = pts[1:]
    xs = [p[0] for p in tail]
    ys = [p[1] for p in tail]
    zs = [p[2] for p in tail]

    def _fixed_sequence():
        rx0, ry0 = PICK_ORIENTATION_DEG[:2]
        rz_raw = _yaw(tail)
        rz_seq = rz_raw if isinstance(rz_raw, list) else [rz_raw] * len(tail)
        return [(rx0, ry0, rz) for rz in rz_seq]

    def _fixed_orientation():
        # eases FROM the arm's actual current orientation (which, after a
        # gazed arc, can be far from PICK_ORIENTATION_DEG) -- a no-op when
        # it's already there, e.g. the whole run has GAZE_ENABLED=False.
        start_rpy = arm.get_coords()[3:]
        return ease_to_rpy(_fixed_sequence(), start_rpy, durs, GAZE_EASE_IN_S)

    if GAZE_ENABLED and gaze_target is not None:
        start_rpy = arm.get_coords()[3:]
        rx, ry, rz = gaze_then_level_waypoints(
            tail, gaze_target, _fixed_sequence(), start_rpy, durs,
            GAZE_EASE_IN_S, GAZE_EASE_OUT_S)
    else:
        rx, ry, rz = _fixed_orientation()
    r = arm.send_path(x=xs, y=ys, z=zs, rx=rx, ry=ry, rz=rz, durations=list(durs))
    if not r and GAZE_ENABLED and gaze_target is not None:
        print(f"  {label}: gaze pose unreachable ({arm.last_error}) "
              f"-- retrying this arc with fixed orientation")
        rx, ry, rz = _fixed_orientation()
        r = arm.send_path(x=xs, y=ys, z=zs, rx=rx, ry=ry, rz=rz, durations=list(durs))
    if not r:
        print(f"  {label}: send_path REFUSED -- {arm.last_error}")
        return False
    pl = arm.last_plan
    print(f"  {label}: {pl.path_length_cm:.1f} cm, {pl.duration_s:.2f} s, "
          f"peak {pl.peak_joint_dps:.0f} deg/s")
    if arm.last_execution and arm.last_execution.late_deadlines:
        print(f"  !! {arm.last_execution.late_deadlines} late control-loop "
              f"deadline(s) during this arc")
    return True


def _play_success_animation(arm):
    """Played once the last cube is placed, replacing the old lead-out-to-
    HOME ending entirely: move to the straight staging pose (base=90deg,
    arm fully extended, gripper pointing exactly at +X), then a gentle
    dip-and-rise that settles slightly bent. Direct joint-space motion --
    see _celebrate.py for why."""
    if not CELEBRATE_ENABLED:
        return True
    staging_j6 = solve_j6_level(CELEBRATE_BASE_J1_DEG, 0.0, 0.0, CELEBRATE_J4_STATIC_DEG,
                                CELEBRATE_STAGING_J5_DEG, CELEBRATE_LEVEL_AXIS_INDEX)
    staging_q = [CELEBRATE_BASE_J1_DEG, 0.0, 0.0, CELEBRATE_J4_STATIC_DEG,
                CELEBRATE_STAGING_J5_DEG, staging_j6]
    print("\n=== success animation ===")
    print("  moving to the straight staging pose...")
    if not arm.move_joints(staging_q, duration=CELEBRATE_STAGING_DURATION_S):
        print(f"  success animation: staging move REFUSED -- {arm.last_error} (skipping)")
        return True

    q_osc = build_straight_arm_oscillation_waypoints(
        base_j1_deg=CELEBRATE_BASE_J1_DEG, dip_mm=CELEBRATE_DIP_MM,
        final_dip_mm=CELEBRATE_FINAL_DIP_MM, cycles=CELEBRATE_CYCLES,
        elbow_branch_sign=CELEBRATE_ELBOW_BRANCH_SIGN,
        j4_static_deg=CELEBRATE_J4_STATIC_DEG, j5_deg=CELEBRATE_STAGING_J5_DEG,
        level_axis_index=CELEBRATE_LEVEL_AXIS_INDEX,
        n_oscillate_waypoints=CELEBRATE_OSCILLATE_WAYPOINTS,
        n_settle_waypoints=CELEBRATE_SETTLE_WAYPOINTS)
    n_waypoints = CELEBRATE_OSCILLATE_WAYPOINTS + CELEBRATE_SETTLE_WAYPOINTS + 1

    q_waypoints = np.vstack([np.asarray(staging_q, dtype=float)[None, :], q_osc])

    # CELEBRATE_DURATION_S is a minimum, not an exact value -- _execute()
    # (unlike plan_path/send_path) applies no hardware speed check of its
    # own, so a bigger CELEBRATE_DIP_MM/CYCLES can silently demand more
    # deg/s than the real servo can deliver; stretch the timeline (never
    # speed it up) so every joint stays within armik.config.MAX_JOINT_SPEED_DPS.
    unit_durs = celebration_durations(CELEBRATE_EASE_IN, CELEBRATE_EASE_OUT, 1.0, n_waypoints)
    min_duration_s = min_feasible_duration_s(q_waypoints, unit_durs)
    duration_s = max(CELEBRATE_DURATION_S, min_duration_s)
    if duration_s > CELEBRATE_DURATION_S + 1e-6:
        print(f"  success animation: CELEBRATE_DURATION_S={CELEBRATE_DURATION_S:.1f}s too "
              f"fast for this dip/cycle count -- auto-stretched to {duration_s:.1f}s")
    durs = [d * duration_s for d in unit_durs]

    timestamps = np.concatenate([[0.0], np.cumsum(durs)])
    plan = Plan(ok=True, q_waypoints=q_waypoints, timestamps=timestamps,
               duration_s=float(timestamps[-1]))
    ex = arm._execute(plan)
    if not ex.ok:
        print(f"  success animation: oscillation REFUSED -- {ex.error}")
    return True


def run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs, d_max, cruise_dur_max):
    """Scripted flinch: approach part-way, recoil, wait, re-approach the moved cube."""
    n = len(pts)
    cut = max(2, int(round(NUDGE_AT_FRACTION * (n - 1))) + 1)
    gaze = seg["gaze"]                 # the original cube -- kept through approach + recoil
    print(f"  NUDGE: approaching to {int(NUDGE_AT_FRACTION*100)}% ...")
    if not _send_arc(arm, pts[:cut], durs[:cut - 1], "  nudge approach", gaze_target=gaze):
        return False

    here = current_pos(arm)
    travel = np.asarray(here, float) - np.asarray(pts[0], float)
    dirn = travel / (np.linalg.norm(travel) + 1e-9)
    recoil = tuple(float(v) for v in (
        np.asarray(here, float) - dirn * NUDGE_RECOIL_CM
        + np.array([0.0, 0.0, NUDGE_RECOIL_CM * 0.5])
    ))

    print(f"  RECOIL -> {tuple(round(v, 1) for v in recoil)}")
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

    print(f"  waiting {NUDGE_SETTLE_S:.1f}s for the cube to settle ...")
    time.sleep(NUDGE_SETTLE_S)

    new_cube = tuple(float(c + o) for c, o in zip(seg["target"], NUDGE_OFFSET_CM))
    print(f"  cube moved -> re-approaching {tuple(round(v, 1) for v in new_cube)}")
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
    return bad == 0


# ===========================================================================
# MAIN
# ===========================================================================

def _build_segments(order, home_tip):
    segs = []
    prev_target = home_tip
    for k in order:
        segs.append({"kind": "reach", "origin": prev_target,
                     "target": CUBES_INITIAL_POINTS[k], "k": int(k),
                     "gaze": CUBES_INITIAL_POINTS[k]})
        segs.append({"kind": "carry", "origin": CUBES_INITIAL_POINTS[k],
                     "target": CUBES_TARGET_POINTS[k], "k": int(k),
                     "gaze": CUBES_TARGET_POINTS[k]})
        prev_target = CUBES_TARGET_POINTS[k]
    segs.append({"kind": "leadout", "origin": prev_target, "target": home_tip, "k": None,
                 "gaze": None})
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

    segments = _build_segments(order, home_tip)

    # the widest move sets the shared parabola; every shorter arc lifts less
    d_max = max((_chord_len(s["origin"], s["target"]) for s in segments), default=1.0) or 1.0
    print(f"widest move {d_max:.1f} cm -> apex {MAX_HEIGHT_TRAJECTORY:.1f} cm  "
          f"(shared parabola a = {-MAX_HEIGHT_TRAJECTORY / (d_max / 2.0) ** 2:.4f})")

    # ARC_TIME_EQUALIZATION blends each reach/carry arc's own CRUISE_SPEED_CM_S
    # duration toward the WIDEST arc's own duration at that speed
    cruise_dur_max = 0.0
    for seg in segments:
        if seg["kind"] == "leadout":
            continue
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
        if seg["kind"] == "leadout":
            all_durs.append(get_durations(seg["origin"], seg["target"], h, ei, eo,
                                          cruise=LEADOUT_SPEED_CM_S))
            continue
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

        if not _grip(arm, GRIP_OPEN_DEG, "open before first pick"):
            return 1

        for ci, seg in enumerate(segments):
            kind = seg["kind"]
            pts, durs = paths[ci], all_durs[ci]

            if kind == "leadout":
                time.sleep(LEADOUT_PAUSE_S)
                _play_success_animation(arm)
                break

            grip_deg = GRIP_CLOSED_DEG if kind == "reach" else GRIP_OPEN_DEG
            label = "reach & grasp" if kind == "reach" else "carry & place"
            print(f"\n=== cycle {ci}/{N_CYCLES - 1}  {label}  "
                  f"cube #{seg['k'] + 1}  -> {tuple(round(v, 1) for v in seg['target'])} ===")

            if kind == "reach" and NUDGE_ENABLED and seg["k"] == NUDGED_CUBE:
                if not run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs, d_max,
                                 cruise_dur_max):
                    print("\naborting run."); go_home(arm); return 1
                if not _grip(arm, GRIP_CLOSED_DEG, "close on cube (new position)"):
                    return 1
                continue

            if not _send_arc(arm, pts, durs, label, gaze_target=seg["gaze"]):
                print("\naborting run."); go_home(arm); return 1

            # grip IMMEDIATELY -- nothing (no position read, no extra round
            # trip) runs between the arm stopping and the gripper command.
            if not _grip(arm, grip_deg, "close on cube" if kind == "reach" else "release cube"):
                return 1

            cur = current_pos(arm)
            reached = (has_reached_cube(cur, seg["target"]) if kind == "reach"
                       else has_reached_target(cur, seg["target"]))
            if not reached:
                print(f"  !! tip at {tuple(round(v, 2) for v in cur)}, expected "
                      f"{tuple(round(v, 1) for v in seg['target'])} +/- {REACH_TOL_CM} cm")

        print("\nall cubes placed.")
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
