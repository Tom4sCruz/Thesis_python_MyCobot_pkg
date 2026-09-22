#!/usr/bin/env python3
"""
MOVEMENT PROFILE: Low-human / Low-robot  (chaotic, uncoordinated, unintentional)
=================================================================================

Move cubes from one side of the frame to the other, but with neither the
smoothness of a human hand nor the precision of a robot: chaotic, sloppy,
unintentional, no agency, variable speed, non-uniform, non-sequential.

  * every move is STILL a smooth parabolic arc (accelerate out, decelerate
    in -- the arm has no other way to move), but nothing about the SHAPE of
    that arc is consistent: each move independently picks a "wide-curved"
    (high apex, generous sideways bow) or "low-to-ground" (low apex,
    minimal bow) style, at random, regardless of distance;
  * EASE_IN / EASE_OUT (how abruptly it starts/stops) are drawn fresh,
    independently, across their FULL range every move -- sometimes it
    decelerates gently and early, sometimes abruptly and late;
  * cruise speed is drawn fresh per move too -- "sometimes faster, other
    times slower" -- there is no equalization toward a shared pace;
  * the cubes ARE grabbed and placed precisely (no overshoot/undershoot --
    unlike an earlier version of this profile) -- the chaos instead lives
    in GRIPPER TIMING (see below);
  * the cubes are grabbed in a RANDOM order and dropped at RANDOM positions
    (TARGET_POSITIONS -- placeholders, retune freely);
  * a cube may be NUDGED (moved by hand) mid-run. Unlike HighH-LowR.py,
    there is NO code here that reacts to this -- the arm has no sensors and
    this script deliberately implements no response, so it just continues
    toward the cube's ORIGINAL position as if nothing happened. The absence
    of a reaction is the point (no agency, no reaction to an external
    event).

    python3 scripts/Profiles/LowH-LowR.py --mock --yes      # no hardware
    python3 scripts/Profiles/LowH-LowR.py --port /dev/ttyTHS1

CYCLES
------
N_CYCLES = 2 * (number of cubes). Even cycle 2k = reach cube k and close;
odd cycle 2k+1 = carry cube k to its target and open. A final lead-out arc
returns to HOME.

GRIPPER TIMING -- THE CORE OF THIS PROFILE
-------------------------------------------
Every other profile in this repo fires the gripper exactly at the arc's
end. Here, each cycle draws a random timing OFFSET (seconds,
GRIP_CHAOS_SPREAD_S each way) for when its gripper command actually fires,
relative to that "correct" moment:

  offset <= 0 (fires EARLY, inside THIS arc): the arc is split near its
    TAIL, |offset| seconds before the end (_tail_split_index -- same
    technique as HighH-LowR_delay-points.py's GRIP_LEAD_S), the gripper
    fires there, then the rest of the arc completes. On a carry, this
    drops the cube mid-transit. On a reach, a large enough early offset
    closes the gripper on nothing before the cube is even reached.

  offset > 0 (fires LATE, deferred into the START of the NEXT arc): this
    arc completes with NO gripper action at all; the action is carried
    forward and fired near the START of the very next arc instead (
    _head_split_index, the forward-walking mirror of _tail_split_index).
    On a carry, this means the arm has already started back toward the
    next reach before the release fires -- it carries the cube partway
    back, then drops it. On a reach, the close fires after the following
    lift/departure has already begun.

One mechanism, no cube-state tracking: the script only ever controls WHEN
the gripper opens or closes relative to the arm's motion; whatever
actually happens to a cube at that moment is a real physical consequence,
not something simulated in code. Both split points are clamped to the
arc's own actual duration by construction, so a large offset just lands at
the very start/end rather than reaching more than one cycle ahead.

This never runs two things at once (no background thread) -- each call is
a normal, fully-blocking send_path()/send_gripper(), one after another,
exactly like the existing HighH-LowR_delay-points.py prototype this is
adapted from.

Everything you tune is a CONSTANT below. The cube coordinates,
PICK_ORIENTATION_DEG, and TARGET_POSITIONS are PLACEHOLDERS -- measure/
retune them on your arm first. PREFLIGHT defaults to True here (unlike the
other profiles) since a low-to-ground style combined with a fixed
straight-down orientation can force a joint past its real hardware limit
on a short, close-to-base move -- confirmed the hard way fixing
HighH-LowR.py earlier; verify reachability before trusting new
TARGET_POSITIONS on real hardware.
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

from armik import Arm, config, pose_coords

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
PREFLIGHT = True                  # on by default here -- see module docstring
RANDOM_SEED = None                # int for a repeatable run, None for fresh each time (default:
                                  # fresh -- "never traces the same trajectory twice")

# -- cubes (MEASURE AND REPLACE) ----------------------------------------------
# (x, y, z) CM, at the GRIPPER TIP, base frame, z from the table.

Z_CUBE_COORD = -4.0

CUBES_INITIAL_POINTS = [          # a row on the pick side (reused from HighH-LowR.py)
    (14.0, 23.0, Z_CUBE_COORD),
    (14.0, 15.5, Z_CUBE_COORD),
    (14.0, 10.0, Z_CUBE_COORD),
]
CUBES_INITIAL_POINTS = CUBES_INITIAL_POINTS[::-1]

# PLACEHOLDER -- deliberately scattered, not a tidy row, to emphasize chaotic/
# unintentional placement. Retune freely; keep count == len(CUBES_INITIAL_POINTS).
TARGET_POSITIONS = [
    (16.0, -9.0, Z_CUBE_COORD + 0.5),
    (12.0, -19.5, Z_CUBE_COORD - 0.5),
    (18.5, -14.0, Z_CUBE_COORD),
]

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

# -- arc shape: two styles, chosen at random per move, independent of distance ---
# Unlike HighH-LowR.py's single shared-parabola family (apex scales with chord
# length), every move here independently rolls "wide-curved" or "low-to-ground" --
# this is what produces "sometimes wide, sometimes low," not a smooth family.
STYLE_WIDE_PROB = 0.5             # chance a given move is "wide-curved" (else "low-to-ground")
APEX_WIDE_MIN_CM = 11.0
APEX_WIDE_MAX_CM = 13.0           # capped below HighH-LowR.py's ~17-18cm reach ceiling: HOME's
                                  # own tip height (~10cm) plus a too-high apex on a "reach from
                                  # home" move pushes the arc's midpoint z out of reach -- a
                                  # failure mode that can't occur in HighH-LowR.py's shared-
                                  # parabola design (a short reach-from-home arc never gets the
                                  # full apex there) but can here, since every move independently
                                  # rolls the full range. Verified via direct IK reproduction.
APEX_LOW_MIN_CM = 8.0             # verified via direct IK reproduction against every placeholder
APEX_LOW_MAX_CM = 10.0            # reach/carry arc below (7.0 still failed one); a long chord
                                  # crossing low near the base needs real clearance to hold a
                                  # fixed straight-down orientation without exceeding a joint's
                                  # real limit -- same lesson as HighH-LowR.py's MIN_ARC_HEIGHT_CM
                                  # fix. Re-verify (see scripts/Profiles/LowH-LowR.py's own
                                  # preflight, PREFLIGHT=True by default) once you retune
                                  # TARGET_POSITIONS -- the safe floor depends on the actual
                                  # coordinates, not just this constant.
BOW_WIDE_JITTER_CM = 6.0          # sideways bow, perpendicular to the chord
BOW_LOW_JITTER_CM = 0.5
# Keep MAX apex reachable at PICK_ORIENTATION_DEG: gripper-down the arm runs out
# of reach around world z ~ 17-18 cm near the workspace edge (same note as
# HighH-LowR.py -- physical constraint, not a tunable).

# -- velocity profile: everything drawn fresh per move, no shared pacing --------
CRUISE_SPEED_MIN_CM_S = 8.0        # "sometimes faster, other times slower" -- no
CRUISE_SPEED_MAX_CM_S = 28.0       # ARC_TIME_EQUALIZATION here; every arc keeps its own pace
LEADOUT_SPEED_CM_S = 10.0         # the final arc back toward HOME is slower / gentler
EASE_MIN, EASE_MAX = 0.0, 10.0     # EASE_IN/EASE_OUT drawn independently, UNIFORMLY across
                                   # this whole range per move (not jittered around a nominal --
                                   # this is what makes it slow down "too early" or "too late"
                                   # unpredictably; a low draw also LOOKS weakly-damped/abrupt,
                                   # without ever skipping the real zero-velocity landing)
# NOTE: an abrupt ease (near EASE_MIN) combined with a fast cruise draw (near
# CRUISE_SPEED_MAX_CM_S) can ask a short first/last segment for more deg/s than
# MAX_JOINT_SPEED_DPS allows, especially on a move that swings a large J1 angle
# over a short chord -- confirmed empirically, not just theoretically. Rather
# than narrow these ranges down to whatever's safe for TODAY's placeholder
# TARGET_POSITIONS (fragile -- the safe range depends on the actual chords),
# _draw_valid_move() below re-rolls a few times against a live arm.plan_path()
# check (the exact check send_path() will run) and falls back to a known-safe,
# still-varied combination if repeated draws keep failing. The ranges here stay
# wide on purpose; the retry is what makes that safe.
PATH_WAYPOINTS = 30                # samples per arc
MIN_SEGMENT_S = 0.02
MOVE_DRAW_MAX_TRIES = 12           # re-roll attempts before _draw_valid_move() falls back

# -- order ------------------------------------------------------------------------
SHUFFLE_ORDER = True              # grab cubes in a random order (init<->target pairing kept)

# -- gripper timing chaos (see module docstring) -----------------------------------
GRIP_CHAOS_SPREAD_S = 1.2         # offset drawn uniform in [-this, +this] each cycle

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 120.0           # 0 = closed .. config.MAX_GRIPPER_DEG = full open
GRIP_CLOSED_DEG = 65.0          # tune to the cube width
GRIP_SPEED = 90  #config.GRIPPER_DEFAULT_SPEED
GRIP_SETTLE_S = 0.35           # quiet time after a gripper command: it must LAND and the
                              # jaws start moving. Tunable down to GRIP_MIN_GAP_S, not below.
GRIP_MIN_GAP_S = 0.2          # hard floor -- pymycobot silently drops a gripper command
                              # that is not followed by a short quiet gap (why 0.0 failed).
REACH_TOL_CM = 3.0             # has_reached_* tolerance, per axis -- still checked/logged even
                               # though this profile aims for reliable geometric precision
LEADOUT_PAUSE_S = 0.5         # deliberate beat between the last action and homing

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
    """Scalar rz at a single point (for preflight)."""
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


def random_style(rng):
    """Roll this move's arc style. Returns (apex_height_cm, bow_cm, style_name)."""
    if rng.random() < STYLE_WIDE_PROB:
        h = float(rng.uniform(APEX_WIDE_MIN_CM, APEX_WIDE_MAX_CM))
        bow = float(rng.uniform(-BOW_WIDE_JITTER_CM, BOW_WIDE_JITTER_CM))
        return h, bow, "wide-curved"
    h = float(rng.uniform(APEX_LOW_MIN_CM, APEX_LOW_MAX_CM))
    bow = float(rng.uniform(-BOW_LOW_JITTER_CM, BOW_LOW_JITTER_CM))
    return h, bow, "low-to-ground"


def _parabola_points(origin, target, arc_height, bow):
    """(list of PATH_WAYPOINTS (x,y,z), arc_length) along the arc from `origin`
    to `target`: the straight xy chord + a symmetric vertical parabolic lift of
    apex `arc_height` above the chord, with a sideways `bow` (cm, perpendicular
    to the chord). Unlike HighH-LowR.py, height/bow are passed in directly --
    this profile rolls them once per move via random_style(), not from a
    jitter-around-nominal RNG call inside here."""
    o = np.asarray(origin, dtype=float)
    t = np.asarray(target, dtype=float)
    z0, z1 = float(o[2]), float(t[2])
    h = float(arc_height)

    M = 200
    u = np.linspace(0.0, 1.0, M)
    xy = o[:2][None, :] + (t[:2] - o[:2])[None, :] * u[:, None]

    chord = t[:2] - o[:2]
    n = float(np.linalg.norm(chord))
    if n > 1e-6 and bow != 0.0:
        perp = np.array([-chord[1], chord[0]]) / n
        xy = xy + perp[None, :] * (bow * 4.0 * (u * (1.0 - u)))[:, None]

    z = z0 + (z1 - z0) * u + 4.0 * h * u * (1.0 - u)

    dense = np.column_stack([xy, z])
    L = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    pts, _ = _polyline_points(dense, np.linspace(0.0, L, PATH_WAYPOINTS))
    return [tuple(float(v) for v in p) for p in pts], L


def get_path(origin_point, target_point, arc_height, bow):
    """Arc from origin to target as PATH_WAYPOINTS (x,y,z) points."""
    return _parabola_points(origin_point, target_point, arc_height, bow)[0]


def get_durations(origin_point, target_point, arc_height, bow,
                  ease_in_accel, ease_out_accel, cruise):
    """PATH_WAYPOINTS-1 segment durations (s) for the arc, shaped by the
    EASE_IN / EASE_OUT dials (0..10, no physical meaning), total time derived
    from `cruise` cm/s -- no ARC_TIME_EQUALIZATION in this profile, every arc
    keeps its own randomly-drawn pace."""
    _, L = _parabola_points(origin_point, target_point, arc_height, bow)

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
    mean_v = float(s[-1])
    s_norm = s / s[-1]
    T = L / max(cruise * mean_v, 1e-6)

    ss = np.linspace(0.0, 1.0, PATH_WAYPOINTS)
    tau_k = np.interp(ss, s_norm, tau)
    t_k = tau_k * T
    return np.maximum(np.diff(t_k), MIN_SEGMENT_S).tolist()


def _tail_split_index(durs, lead_s):
    """Largest i such that the remaining travel from pts[i] to the arc's end
    (sum(durs[i:])) is still >= lead_s -- firing the gripper exactly when the
    arm reaches pts[i] gives it about `lead_s` seconds head start before the
    final point. 0 (fire at the very start) if the whole arc is shorter than
    lead_s. Used for an EARLY (offset <= 0) gripper action, within THIS arc."""
    tail = 0.0
    for i in range(len(durs) - 1, -1, -1):
        tail += durs[i]
        if tail >= lead_s:
            return i
    return 0


def _head_split_index(durs, lead_s):
    """Mirror of _tail_split_index, walking forward from the start: smallest
    split such that the travel from the arc's start to pts[split] is >=
    lead_s. len(durs) (fire at the very end, i.e. no real split) if the whole
    arc is shorter than lead_s. Used to fire a gripper action DEFERRED from
    the previous cycle, near the start of this one."""
    head = 0.0
    for i in range(len(durs)):
        head += durs[i]
        if head >= lead_s:
            return i + 1
    return len(durs)


def _draw_valid_move(arm, origin, target, rng, cruise_override=None):
    """Roll style/ease/speed for one move, re-rolling against a live
    arm.plan_path() check (the exact validation send_path() will run --
    reachability AND MAX_JOINT_SPEED_DPS) up to MOVE_DRAW_MAX_TRIES times,
    falling back to a known-safe, still-style-varied combination if draws
    keep failing. See the note above EASE_MIN/EASE_MAX for why this exists
    instead of narrower fixed ranges. No motion -- plan_path() only plans.
    Returns (pts, durs, description_str)."""
    rx, ry = PICK_ORIENTATION_DEG[:2]

    def _try(ei, eo, cruise):
        h, bow, style = random_style(rng)
        pts = get_path(origin, target, h, bow)
        durs = get_durations(origin, target, h, bow, ei, eo, cruise)
        desc = f"{style} ease=({ei:.1f}/{eo:.1f}) speed={cruise:.1f}cm/s"
        if len(pts) < 2:
            return pts, durs, desc, True
        pl = arm.plan_path(x=[p[0] for p in pts[1:]], y=[p[1] for p in pts[1:]],
                           z=[p[2] for p in pts[1:]], rx=rx, ry=ry, rz=_yaw(pts[1:]),
                           durations=durs)
        return pts, durs, desc, pl.ok

    for _ in range(MOVE_DRAW_MAX_TRIES):
        ei = float(rng.uniform(EASE_MIN, EASE_MAX))
        eo = float(rng.uniform(EASE_MIN, EASE_MAX))
        cruise = cruise_override or float(rng.uniform(CRUISE_SPEED_MIN_CM_S, CRUISE_SPEED_MAX_CM_S))
        pts, durs, desc, ok = _try(ei, eo, cruise)
        if ok:
            return pts, durs, desc

    pts, durs, desc, ok = _try(8.0, 8.0, cruise_override or 6.0)
    return pts, durs, desc + f" [fallback after {MOVE_DRAW_MAX_TRIES} tries]"


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


def _send_arc(arm, pts, durs, label):
    """Blocking parabolic move. pts[0] is the implicit start (not sent)."""
    rx, ry = PICK_ORIENTATION_DEG[:2]
    if len(pts) < 2:
        print(f"  {label}: negligible, skipped")
        return True
    r = arm.send_path(
        x=[p[0] for p in pts[1:]], y=[p[1] for p in pts[1:]], z=[p[2] for p in pts[1:]],
        rx=rx, ry=ry, rz=_yaw(pts[1:]), durations=list(durs),
    )
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


# ===========================================================================
# PREFLIGHT
# ===========================================================================

def preflight(arm, segments, paths, all_durs):
    rx, ry = PICK_ORIENTATION_DEG[:2]
    print("\n--- preflight: planning every cube point (no motion) ---")
    checks = []
    for i, (s, t) in enumerate(zip(CUBES_INITIAL_POINTS, TARGET_POSITIONS)):
        checks += [(f"init{i+1}", s), (f"tgt{i+1}", t)]
    bad = 0
    for name, (x, y, z) in checks:
        pl = arm.plan_coords(x=x, y=y, z=z, rx=rx, ry=ry, rz=_yaw_one((x, y)),
                             speed=config.DEFAULT_SPEED_CM_S)
        err = (pl.error or "").lower()
        if pl.ok:
            print(f"  OK  {name:14s} ({x:5.1f},{y:6.1f},{z:4.1f})  "
                  f"peak {pl.peak_joint_dps:.0f} deg/s")
        elif "already at" in err:
            print(f"  OK  {name:14s} ({x:5.1f},{y:6.1f},{z:4.1f})  (already there)")
        else:
            print(f"  BAD {name:14s} ({x:5.1f},{y:6.1f},{z:4.1f})  {pl.error}")
            bad += 1

    # A single apex spot-check isn't enough -- a full arc can fail at some OTHER
    # waypoint along the way even when its highest point alone is reachable (seen
    # empirically while tuning APEX_LOW_MIN_CM/MAX_CM). plan_path() runs the exact
    # same chained-seed IK walk over every waypoint that send_path() will actually
    # execute, so this is a true dry-run, not an approximation of one.
    print("--- preflight: planning every full arc (matches send_path exactly) ---")
    n_arcs = 0
    for ci, (seg, pts, durs) in enumerate(zip(segments, paths, all_durs)):
        if len(pts) < 2:
            continue
        n_arcs += 1
        pl = arm.plan_path(x=[p[0] for p in pts[1:]], y=[p[1] for p in pts[1:]],
                           z=[p[2] for p in pts[1:]], rx=rx, ry=ry, rz=_yaw(pts[1:]),
                           durations=list(durs))
        label = f"{seg['kind']} c{ci}"
        if pl.ok:
            print(f"  OK  {label:14s} peak {pl.peak_joint_dps:.0f} deg/s")
        else:
            print(f"  BAD {label:14s} {pl.error}")
            bad += 1

    print(f"--- preflight: {len(checks) + n_arcs - bad}/{len(checks) + n_arcs} reachable ---")
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
                     "target": TARGET_POSITIONS[k], "k": int(k)})
        prev_target = TARGET_POSITIONS[k]
    segs.append({"kind": "leadout", "origin": prev_target, "target": home_tip, "k": None})
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

    if len(CUBES_INITIAL_POINTS) != len(TARGET_POSITIONS):
        print("CUBES_INITIAL_POINTS and TARGET_POSITIONS must be the same length.")
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
    print("no nudge-reaction code in this profile -- a live nudge gets no response, by design")

    segments = _build_segments(order, home_tip)

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
    bridge = None

    try:
        # power on FIRST -- _draw_valid_move() below needs a powered arm to plan
        # against (arm.plan_path() refuses "arm reports power off" otherwise)
        if not arm.conn.is_power_on():
            print("powering on...")
            arm.conn.power_on()
            time.sleep(1.5)

        # ---- precompute every arc + its durations (style/ease/speed all fresh per move) ----
        # _draw_valid_move() validates each candidate arc against a live arm.plan_path()
        # (reachability + MAX_JOINT_SPEED_DPS) before committing to it -- see the note
        # above EASE_MIN/EASE_MAX. Purely planning; no motion happens here.
        paths, all_durs, styles = [], [], []
        for seg in segments:
            cruise_override = LEADOUT_SPEED_CM_S if seg["kind"] == "leadout" else None
            pts, durs, desc = _draw_valid_move(arm, seg["origin"], seg["target"], rng, cruise_override)
            paths.append(pts)
            all_durs.append(durs)
            styles.append(desc)

        if args.rviz:
            from _rviz_bridge import RvizBridge
            flat_path = [tuple(map(float, p)) for seg_pts in paths for p in seg_pts]
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

        print("homing...")
        if not go_home(arm):
            print("\nhoming failed -- fix the arm's position (see error above), then try again.")
            return 1
        time.sleep(SETTLE_S)
        print(f"start pose (tip, cm/deg): {[round(v, 2) for v in arm.get_coords()]}")

        if PREFLIGHT and not preflight(arm, segments, paths, all_durs):
            print("\npreflight failed -- fix the cube/target coordinates or PICK_ORIENTATION_DEG. "
                  "Nothing moved.")
            go_home(arm)
            return 1

        if not _grip(arm, GRIP_OPEN_DEG, "open before first pick"):
            return 1

        pending = None   # (grip_deg, action_label, offset_s) deferred from the previous cycle

        for ci, seg in enumerate(segments):
            kind = seg["kind"]
            pts, durs = paths[ci], all_durs[ci]

            if kind == "leadout":
                print(f"\n=== lead-out arc -> HOME ===  [{styles[ci]}]")
                time.sleep(LEADOUT_PAUSE_S)
                if pending is not None:
                    p_deg, p_label, p_off = pending
                    split = _head_split_index(durs, p_off)
                    if split > 0:
                        if not _send_arc(arm, pts[:split + 1], durs[:split], "lead-out (deferred head)"):
                            print("\naborting run."); go_home(arm); return 1
                        pts, durs = pts[split:], durs[split:]
                    if not _grip(arm, p_deg, f"{p_label} (fired late)"):
                        return 1
                    pending = None
                _send_arc(arm, pts, durs, "lead-out")
                break

            grip_deg = GRIP_CLOSED_DEG if kind == "reach" else GRIP_OPEN_DEG
            action = "close on cube" if kind == "reach" else "release cube"
            label = "reach & grasp" if kind == "reach" else "carry & place"
            print(f"\n=== cycle {ci}/{N_CYCLES - 1}  {label}  "
                  f"cube #{seg['k'] + 1}  -> {tuple(round(v, 1) for v in seg['target'])}  "
                  f"[{styles[ci]}] ===")

            # fire anything deferred from the PREVIOUS cycle, near the start of this arc
            if pending is not None:
                p_deg, p_label, p_off = pending
                split = _head_split_index(durs, p_off)
                if split > 0:
                    if not _send_arc(arm, pts[:split + 1], durs[:split], f"{label} (deferred head)"):
                        print("\naborting run."); go_home(arm); return 1
                    pts, durs = pts[split:], durs[split:]
                if not _grip(arm, p_deg, f"{p_label} (fired late)"):
                    return 1
                pending = None

            # this cycle's own gripper action, timed with a fresh random offset
            offset = float(rng.uniform(-GRIP_CHAOS_SPREAD_S, GRIP_CHAOS_SPREAD_S))
            if offset <= 0.0:
                split = _tail_split_index(durs, -offset)
                if split > 0:
                    if not _send_arc(arm, pts[:split + 1], durs[:split], f"{label} (approach)"):
                        print("\naborting run."); go_home(arm); return 1
                    pts, durs = pts[split:], durs[split:]
                if not _grip(arm, grip_deg, f"{action} (offset {offset:+.2f}s)"):
                    return 1
                if not _send_arc(arm, pts, durs, f"{label} (final)"):
                    print("\naborting run."); go_home(arm); return 1
            else:
                if not _send_arc(arm, pts, durs, label):
                    print("\naborting run."); go_home(arm); return 1
                pending = (grip_deg, action, offset)
                print(f"  gripper action deferred ~{offset:.2f}s into the next leg")

            cur = current_pos(arm)
            reached = (has_reached_cube(cur, seg["target"]) if kind == "reach"
                       else has_reached_target(cur, seg["target"]))
            if not reached:
                print(f"  !! tip at {tuple(round(v, 2) for v in cur)}, expected "
                      f"{tuple(round(v, 1) for v in seg['target'])} +/- {REACH_TOL_CM} cm")

        print("\nall cubes placed. homing...")
        if not go_home(arm):
            print(f"  final homing failed -- {arm.last_error}")
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
