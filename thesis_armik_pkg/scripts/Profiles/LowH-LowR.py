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

LOOP_REACH_CYCLE replaces one reach's drawn arc with a scripted loop: forward
and fully stretched out in front, up, then over and down onto the cube (see
the LOOP REACH constants).

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

Designated cycles can override that randomness with a fixed, deterministic
timing instead -- NORMAL_GRIP_CYCLES forces a reliable close / release right
at arrival on the listed cycles (offset pinned to 0s, bypassing the random
draw), and EARLY_DROP_CYCLE forces a release at a fixed fraction of one designated
carry's OWN duration (the early/within-this-arc mirror of
DELAYED_OPEN_CYCLE's late/deferred-into-the-next-arc override, see below).

One mechanism, no cube-state tracking: the script only ever controls WHEN
the gripper opens or closes relative to the arm's motion; whatever
actually happens to a cube at that moment is a real physical consequence,
not something simulated in code. Both split points are clamped to the
arc's own actual duration by construction, so a large offset just lands at
the very start/end rather than reaching more than one cycle ahead.

This never runs two things at once (no background thread) -- each motion
and gripper command is fully blocking, one after another, exactly like the
existing HighH-LowR_delay-points.py prototype this is adapted from.

Every move is PRE-COMPUTED, as in HighH-LowR.py: the whole run is first
rehearsed on a simulated arm (every arc IK-planned, recorded to a "tape"),
then the tape is played back on the real arm with no planning between
actions -- the only gaps left are the deliberate gripper/lead-out pauses.

Everything you tune is a CONSTANT below. The cube coordinates,
PICK_ORIENTATION_DEG, and TARGET_POSITIONS are PLACEHOLDERS -- measure/
retune them on your arm first. PREFLIGHT defaults to True here (unlike the
other profiles) since a low-to-ground style combined with a fixed
straight-down orientation can force a joint past its real hardware limit
on a short, close-to-base move -- confirmed the hard way fixing
HighH-LowR.py earlier; verify reachability before trusting new
TARGET_POSITIONS on real hardware.

COMPILED RUN
------------
The rehearsed run is saved to compiled/LowH-LowR.json and reused on the next
launch until this script, armik/ or a _*.py helper changes (see
_compiled.py). --recompile forces a fresh rehearsal; --compile-only builds
it and exits without connecting to an arm (compile on the laptop, commit,
and the Jetson only plays it back):

    python3 scripts/Profiles/LowH-LowR.py --compile-only
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

from armik import Arm, Plan, config, ik, pose_coords
from armik.arm import Execution
from armik.kinematics import check_joint_limits
import _compiled

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
PREFLIGHT = False                  # on by default here -- see module docstring
RANDOM_SEED = 1                # int for a repeatable run, None for fresh each time (default:
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


TARGET_POSITIONS = [           # clean, evenly-spaced drop row, uniform z -- no overshoot
    (8.0, -22.0, Z_CUBE_COORD),
    (19.0, -17.5, Z_CUBE_COORD),
    (18.0, -5.0, Z_CUBE_COORD),
    #(15.0, -19.0, Z_CUBE_COORD),
]
TARGET_POSITIONS = TARGET_POSITIONS[::-1]


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
SHUFFLE_ORDER = False             # grab cubes in a random order (init<->target pairing kept)
PICKUP_ORDER = [0,2,1]

# -- gripper timing chaos (see module docstring) -----------------------------------
GRIP_CHAOS_SPREAD_S = 1.2         # offset drawn uniform in [-this, +this] each cycle

# -- special cycles: three one-off trajectory/timing variants, each on ONE ----------
# designated cycle (like HighH-LowR.py's NUDGE_CYCLE). -1 disables any of them.
# Cycle indices: 0=reach1,1=carry1,2=reach2,3=carry2,4=reach3,5=carry3 (3 cubes).

# 1) DIAGONAL ARC -- must be an EVEN (reach) cycle: the arm swings diagonally
# past the direct bearing before curving in to the next cube, starting from
# where it just placed the previous cube ("from the target position, reach
# forward... then move to the next cube").
DIAGONAL_ARC_CYCLE = 2
DIAGONAL_REACH_CM = 12.0    # how far the initial diagonal swing travels
DIAGONAL_ANGLE_DEG = 35.0   # angle of that swing off the direct bearing (0 = straight
                            # at the target, 90 = pure sideways)
DIAGONAL_APEX_CM = 10.0     # apex height of EACH sub-arc (out-swing and curve-back)
DIAGONAL_BOW_CM = 2.0       # slight sideways bow on the curve-back sub-arc

# 2) DETERMINISTIC DELAYED OPEN -- must be an ODD (carry) cycle: unlike every
# other cycle's RANDOM gripper-timing offset, this one deterministically
# withholds the release until a fixed, tunable FRACTION of the very next
# arc's own duration has elapsed (0.75 -> fires 3/4 of the way through the
# return-to-next-cube leg). Reuses the same deferred-gripper ("pending")
# machinery every other cycle already uses -- see the module docstring.
DELAYED_OPEN_CYCLE = 3
DELAYED_OPEN_FRACTION = 0.15  # 0..1 -- drop shortly after setting off toward the
                              # next cube

# 3) ROLLERCOASTER -- must be an ODD (carry) cycle: instead of forward/back
# oscillation, the arm performs an actual vertical circular loop (a
# loop-the-loop) partway along the origin->target chord, then continues on
# to the target -- "grab the cube, move towards target, then loop, like a
# rollercoaster."
ROLLERCOASTER_CYCLE = 5
ROLLERCOASTER_PASSES = 1              # number of full 360 deg revolutions
ROLLERCOASTER_RADIUS_MIN_CM, ROLLERCOASTER_RADIUS_MAX_CM = 5.0, 7.0
ROLLERCOASTER_LOOP_AT_FRAC_MIN, ROLLERCOASTER_LOOP_AT_FRAC_MAX = 0.35, 0.65
                                       # where along the chord (0=origin,1=target) the
                                       # loop sits -- re-rolled per retry, same IK-
                                       # difficulty rationale the old forward/back
                                       # fractions had: a loop can land in a hard IK
                                       # region regardless of ease/speed. Mirrors
                                       # random_style()'s role for the default 2-
                                       # waypoint case -- the geometry itself needs
                                       # retry diversity here, not just timing.
ROLLERCOASTER_LEAD_APEX_CM = 6.0      # apex height of the short lead-in/out sub-arcs
                                       # that carry the arm up to/down from loop height
ROLLERCOASTER_LEAD_BOW_CM = 0.0
ROLLERCOASTER_LOOP_CLEARANCE_CM = 8.0  # how far ABOVE the origin->target chord the loop's
                                        # own anchor point sits, on top of (not instead of)
                                        # the lead-in apex above -- matches APEX_LOW_MIN_CM's
                                        # own safe floor (see its comment above): a loop
                                        # anchored directly ON a long, low, near-base-crossing
                                        # chord sits in the exact same dead zone that floor
                                        # was already fixed for, and a taller lead-in apex
                                        # alone does NOT help, since a parabola's hump tapers
                                        # back down to the (low) anchor height at its tail --
                                        # confirmed via direct IK reproduction against this
                                        # profile's own placeholder chords. Keep clear of the
                                        # ~17-18cm reach ceiling too (see PICK_ORIENTATION_DEG
                                        # note above): clearance + 2*radius is the loop's own
                                        # peak height.

# 4) SIDEWAYS ARC -- any cycle (no parity restriction, it's just one arc):
# one arc whose parabolic bulge is split between vertical lift and
# horizontal sideways sway by a single tilt angle, instead of
# random_style()'s independent apex-height/bow draws.
SIDEWAYS_ARC_CYCLE = 0             # -1 disables
SIDEWAYS_ARC_AMPLITUDE_CM = 12.0   # the parabola's peak magnitude before the angle splits it
SIDEWAYS_ARC_ANGLE = 45.0          # 0 = upright (pure vertical lift, no sway, like a normal
                                    # arc); 90 = parallel to the ground (pure sideways sway,
                                    # no height change)

# 5) DETERMINISTIC NORMAL GRIP -- any cycles (reach or carry): overrides the
# usual RANDOM gripper-timing offset with a reliable action right at arrival
# (offset = 0s) -- a reach closes on the cube, a carry releases at its target;
# no early action mid-arc and none deferred into the next leg.
# DELAYED_OPEN_CYCLE / EARLY_DROP_CYCLE win if a cycle is listed in both.
NORMAL_GRIP_CYCLES = (0, 2, 4, 5)   # () disables

# 6) DETERMINISTIC EARLY DROP -- must be an ODD (carry) cycle: the release
# fires partway through THIS carry's own arc (a fixed fraction of its own
# duration elapsed), dropping the cube mid-transit to its target -- the
# early/within-this-arc mirror of DELAYED_OPEN_CYCLE's late/deferred-into-
# the-next-arc override.
EARLY_DROP_CYCLE = 1            # -1 disables
EARLY_DROP_FRACTION = 0.5       # 0..1 -- fraction of THIS carry elapsed before the drop

# 7) LOOP REACH -- must be an EVEN (reach) cycle: instead of the drawn arc, the
# arm 1) moves forward (+x) from where it is, 2) stretches fully to the front
# (LOOP_STRETCH_Q), 3) comes up through LOOP_TOP_CM, 4) curves over and down to
# LOOP_ABOVE_CUBE_CM above the cube, then lands on it straight down and closes
# (always at arrival -- this cycle ignores the random grip offset). A gripper
# action deferred from the previous carry (DELAYED_OPEN_CYCLE) fires during
# stage 1, DELAYED_OPEN_FRACTION of LOOP_STRETCH_S in.
# Stages 1-4 are JOINT-space moves: the stretched arm is a singularity, which
# Cartesian IK can neither reach exactly nor bend back out of. Every via pose
# points the gripper down, and the joint blends between them keep it down.
# Replaced the drawn arc from (8,-22) to cube #2 at (14,15.5), which swung J1
# 114 deg in 2.3 s (192 deg/s) past 7 cm from the base: the real servos lagged
# and the gripper arrived from the side. The loop keeps >= 20 cm from the base
# and ~105 deg/s.
LOOP_REACH_CYCLE = 4                       # -1 disables
LOOP_STRETCH_Q = [0.0, -90.0, 0.0, 0.0, 0.0, 0.0]   # tip ~(28.0, -6.5, -0.7) cm, pointing down
LOOP_STRETCH_S = 2.5                       # stages 1-2: forward + stretch (s)
LOOP_STRETCH_HOLD_S = 0.3                  # pause at full stretch
LOOP_TOP_CM = (20.0, 0.0, 15.0)            # stage 3's highest point -- pointing down, the arm
                                           # reaches r ~21.5 cm at z=15, ~17 cm at z=18
LOOP_UP_OVER_S = 3.5                       # stages 3-4: stretch -> top -> above the cube (s)
LOOP_ABOVE_CUBE_CM = 8.0                   # height above the cube where the vertical landing starts
LOOP_LAND_S = 1.2                          # the vertical landing (s)
LOOP_LAND_WAYPOINTS = 20

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


def _sideways_split(amplitude, angle_deg):
    """Split one parabola amplitude into (height, bow) by SIDEWAYS_ARC_ANGLE:
    0 deg -> all of it becomes vertical lift; 90 deg -> all of it becomes
    horizontal sideways bow, so the arc's plane lies flat, parallel to the
    ground."""
    a = math.radians(angle_deg)
    return amplitude * math.cos(a), amplitude * math.sin(a)


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


def _build_multi_arc(waypoints, apexes, bows):
    """Concatenate a _parabola_points() sub-arc between each consecutive
    pair in `waypoints` (len N), using per-segment apex/bow from `apexes`/
    `bows` (len N-1). Returns (pts, total_length_cm) in the same
    pts[0]=start convention as get_path() -- the shared boundary point
    between consecutive sub-arcs is de-duplicated. Used by the diagonal-arc
    and rollercoaster special cycles, which need more than one sub-arc."""
    all_pts = [tuple(float(v) for v in waypoints[0])]
    total_L = 0.0
    for i in range(len(waypoints) - 1):
        sub_pts, L = _parabola_points(waypoints[i], waypoints[i + 1], apexes[i], bows[i])
        all_pts.extend(sub_pts[1:])
        total_L += L
    return all_pts, total_L


def _multi_arc_durations(waypoints, apexes, bows, ease_in, ease_out, cruise):
    """Matching durations for _build_multi_arc(): one EASE-shaped
    get_durations() call per sub-segment, concatenated. Ease is only
    applied on the true first/last sub-segments (interior ones get abrupt
    0.0 ease) so the whole multi-arc reads as one continuous accel/decel
    envelope, not a stutter-stop at each interior waypoint -- send_path()'s
    own Hermite blending (already used by _send_arc) carries velocity
    through the interior points regardless."""
    durs = []
    n = len(waypoints) - 1
    for i in range(n):
        durs += get_durations(waypoints[i], waypoints[i + 1], apexes[i], bows[i],
                              ease_in if i == 0 else 0.0,
                              ease_out if i == n - 1 else 0.0,
                              cruise)
    return durs


def _rotate_xy(vec, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def diagonal_waypoints(origin, target):
    """[origin, a diagonal 'reach forward' point, target] -- the arm swings
    DIAGONAL_REACH_CM off the direct origin->target bearing (rotated
    DIAGONAL_ANGLE_DEG from it) before curving back in to the real target.
    The diagonal point stays at origin's height; each sub-arc's own
    parabolic lift gives the move its vertical character.

    The rotation can go to either side of the direct bearing; which side is
    picked dynamically, whichever keeps the diagonal point FARTHER from the
    base. A fixed rotation sign can accidentally swing toward the base
    depending on the specific origin/target geometry, forcing an
    unreachable close-in fold to hold a fixed orientation there -- the
    geometry-shape version of the same class of issue already fixed twice
    in this profile for apex height and cruise speed."""
    o, t = np.asarray(origin, float), np.asarray(target, float)
    chord = t[:2] - o[:2]
    chord_dir = chord / (np.linalg.norm(chord) + 1e-9)
    cand_a = o[:2] + _rotate_xy(chord_dir, DIAGONAL_ANGLE_DEG) * DIAGONAL_REACH_CM
    cand_b = o[:2] + _rotate_xy(chord_dir, -DIAGONAL_ANGLE_DEG) * DIAGONAL_REACH_CM
    diag_xy = cand_a if np.linalg.norm(cand_a) >= np.linalg.norm(cand_b) else cand_b
    diag_pt = (float(diag_xy[0]), float(diag_xy[1]), float(o[2]))
    return [tuple(float(v) for v in o), diag_pt, tuple(float(v) for v in t)]


def rollercoaster_loop_anchor(origin, target, loop_frac):
    """The loop-the-loop's anchor point -- horizontally at `loop_frac` of
    the way along the origin->target chord (0=origin, 1=target), but raised
    ROLLERCOASTER_LOOP_CLEARANCE_CM above the chord's own height there (see
    that constant's comment for why) -- and the chord's horizontal unit
    direction. loop_frac is passed in (not read from the module constants
    directly) because _draw_valid_rollercoaster() re-rolls it within
    ROLLERCOASTER_LOOP_AT_FRAC_MIN/MAX on every retry -- see the note above
    that constant for why this path shape needs geometry retries, not just
    ease/speed ones."""
    o, t = np.asarray(origin, float), np.asarray(target, float)
    chord_pt = o + (t - o) * loop_frac
    anchor = (float(chord_pt[0]), float(chord_pt[1]),
             float(chord_pt[2]) + ROLLERCOASTER_LOOP_CLEARANCE_CM)
    chord_xy = t[:2] - o[:2]
    n = float(np.linalg.norm(chord_xy))
    chord_dir = chord_xy / n if n > 1e-6 else np.array([1.0, 0.0])
    return anchor, chord_dir


def _loop_points(anchor, chord_dir, radius, passes):
    """PATH_WAYPOINTS*passes points tracing `passes` full vertical circular
    revolutions of `radius`, in the plane spanned by chord_dir (horizontal,
    aligned with the direction of travel) and z -- the actual "loop-the-
    loop": the arm rises up and over rather than oscillating back and
    forth. Starts and ends at `anchor` (the circle's bottom point), so it
    can be spliced between two lead sub-arcs like any other waypoint.
    Returns (points, arc_length_cm)."""
    M = max(int(round(PATH_WAYPOINTS * passes)), 8)
    theta = np.linspace(0.0, 2.0 * math.pi * passes, M)
    a = np.asarray(anchor, dtype=float)
    xy = a[:2][None, :] + chord_dir[None, :] * (radius * np.sin(theta))[:, None]
    z = a[2] + radius * (1.0 - np.cos(theta))
    pts = np.column_stack([xy, z])
    L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return [tuple(float(v) for v in p) for p in pts], L


def _loop_durations(n_points, length_cm, cruise):
    """Constant-speed durations for _loop_points()'s own point count -- no
    ease (it is always an interior sub-arc, entered/exited already moving,
    same "abrupt 0.0 ease at interior waypoints" convention as
    _multi_arc_durations)."""
    total_s = length_cm / max(cruise, 1e-6)
    dt = total_s / max(n_points - 1, 1)
    return [max(dt, MIN_SEGMENT_S)] * (n_points - 1)


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


def _draw_valid_diagonal(arm, origin, target, rng):
    """Like _draw_valid_move(), for DIAGONAL_ARC_CYCLE: the waypoint
    geometry is fixed (diagonal_waypoints()) -- only ease/speed are
    re-rolled per retry against the same live arm.plan_path() check."""
    rx, ry = PICK_ORIENTATION_DEG[:2]
    waypoints = diagonal_waypoints(origin, target)
    apexes = [DIAGONAL_APEX_CM, DIAGONAL_APEX_CM]
    bows = [0.0, DIAGONAL_BOW_CM]

    def _try(ei, eo, cruise):
        pts, _ = _build_multi_arc(waypoints, apexes, bows)
        durs = _multi_arc_durations(waypoints, apexes, bows, ei, eo, cruise)
        desc = f"diagonal ease=({ei:.1f}/{eo:.1f}) speed={cruise:.1f}cm/s"
        pl = arm.plan_path(x=[p[0] for p in pts[1:]], y=[p[1] for p in pts[1:]],
                           z=[p[2] for p in pts[1:]], rx=rx, ry=ry, rz=_yaw(pts[1:]),
                           durations=durs)
        return pts, durs, desc, pl.ok

    for _ in range(MOVE_DRAW_MAX_TRIES):
        ei = float(rng.uniform(EASE_MIN, EASE_MAX))
        eo = float(rng.uniform(EASE_MIN, EASE_MAX))
        cruise = float(rng.uniform(CRUISE_SPEED_MIN_CM_S, CRUISE_SPEED_MAX_CM_S))
        pts, durs, desc, ok = _try(ei, eo, cruise)
        if ok:
            return pts, durs, desc

    pts, durs, desc, ok = _try(8.0, 8.0, 6.0)
    return pts, durs, desc + f" [fallback after {MOVE_DRAW_MAX_TRIES} tries]"


def _draw_valid_rollercoaster(arm, origin, target, rng):
    """Like _draw_valid_move(), for ROLLERCOASTER_CYCLE: origin -> (short
    lead-in parabola) -> a vertical circular loop-the-loop -> (landing
    parabola, random_style()'d like every other arc) -> target. Unlike the
    diagonal arc, the loop's own geometry is NOT fixed across retries:
    loop_frac (where along the chord it sits) and radius are ALSO re-rolled
    (within ROLLERCOASTER_LOOP_AT_FRAC_MIN/MAX and
    ROLLERCOASTER_RADIUS_MIN/MAX_CM) on every retry, alongside ease/speed
    and the landing segment's apex/bow -- a loop can land in a genuinely
    hard IK region for some cube/target pairs regardless of ease/speed,
    same rationale random_style() already covers for the default 2-
    waypoint case."""
    rx, ry = PICK_ORIENTATION_DEG[:2]

    def _try(ei, eo, cruise, loop_frac, radius):
        anchor, chord_dir = rollercoaster_loop_anchor(origin, target, loop_frac)
        h, bow, _ = random_style(rng)

        lead_in_pts, _ = _parabola_points(origin, anchor, ROLLERCOASTER_LEAD_APEX_CM,
                                          ROLLERCOASTER_LEAD_BOW_CM)
        loop_pts, loop_L = _loop_points(anchor, chord_dir, radius, ROLLERCOASTER_PASSES)
        land_pts, _ = _parabola_points(anchor, target, h, bow)
        pts = lead_in_pts + loop_pts[1:] + land_pts[1:]

        lead_in_durs = get_durations(origin, anchor, ROLLERCOASTER_LEAD_APEX_CM,
                                     ROLLERCOASTER_LEAD_BOW_CM, ei, 0.0, cruise)
        loop_durs = _loop_durations(len(loop_pts), loop_L, cruise)
        land_durs = get_durations(anchor, target, h, bow, 0.0, eo, cruise)
        durs = lead_in_durs + loop_durs + land_durs

        desc = (f"rollercoaster ease=({ei:.1f}/{eo:.1f}) speed={cruise:.1f}cm/s "
                f"loop_at={loop_frac:.2f} r={radius:.1f}cm x{ROLLERCOASTER_PASSES}")
        pl = arm.plan_path(x=[p[0] for p in pts[1:]], y=[p[1] for p in pts[1:]],
                           z=[p[2] for p in pts[1:]], rx=rx, ry=ry, rz=_yaw(pts[1:]),
                           durations=durs)
        return pts, durs, desc, pl.ok

    for _ in range(MOVE_DRAW_MAX_TRIES):
        ei = float(rng.uniform(EASE_MIN, EASE_MAX))
        eo = float(rng.uniform(EASE_MIN, EASE_MAX))
        cruise = float(rng.uniform(CRUISE_SPEED_MIN_CM_S, CRUISE_SPEED_MAX_CM_S))
        loop_frac = float(rng.uniform(ROLLERCOASTER_LOOP_AT_FRAC_MIN, ROLLERCOASTER_LOOP_AT_FRAC_MAX))
        radius = float(rng.uniform(ROLLERCOASTER_RADIUS_MIN_CM, ROLLERCOASTER_RADIUS_MAX_CM))
        pts, durs, desc, ok = _try(ei, eo, cruise, loop_frac, radius)
        if ok:
            return pts, durs, desc

    mid_frac = (ROLLERCOASTER_LOOP_AT_FRAC_MIN + ROLLERCOASTER_LOOP_AT_FRAC_MAX) / 2.0
    mid_radius = (ROLLERCOASTER_RADIUS_MIN_CM + ROLLERCOASTER_RADIUS_MAX_CM) / 2.0
    pts, durs, desc, ok = _try(8.0, 8.0, 6.0, mid_frac, mid_radius)
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
    _say(f"  gripper -> {deg:.0f} deg ({label})")
    if _TAPE is not None:
        _TAPE.append(("grip", float(deg)))
    elif not _fire_gripper(arm, deg):
        print(f"  send_gripper REFUSED -- {arm.last_error}")
        return False
    if GRIP_SETTLE_S < GRIP_MIN_GAP_S:
        _say(f"  (GRIP_SETTLE_S {GRIP_SETTLE_S}s < floor {GRIP_MIN_GAP_S}s -- using the floor)")
    _pause(max(max(GRIP_SETTLE_S, GRIP_MIN_GAP_S) - 0.06, 0.0))
    return True


# ===========================================================================
# THE TAPE -- rehearse the whole run first, then play it back
# ===========================================================================
# Working out a motion (the IK for an arc) takes real time on the Jetson, and
# doing it between actions -- after every gripper command and arc split --
# left the arm standing still. So the choreography runs TWICE:
#   1. REHEARSAL on a second, simulated Arm(mock=True): nothing is streamed and
#      nothing sleeps. Every action -- a finished Plan, a gripper command, a
#      deliberate pause, a log line -- is appended to a list, the tape, and the
#      simulated arm is put at the motion's end pose so the next step plans
#      from the right place.
#   2. PLAYBACK on the real arm: the tape is executed in order. No kinematics
#      and no planning happen between motions, so the only gaps left are the
#      deliberate pauses.
# Every motion function reaches the arm only through _do_plan / _pause / _say /
# _grip, which record while _TAPE is a list and act when it is None.

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


def _do_plan(arm, plan):
    """Execute a finished Plan -- or, in rehearsal, record it and move the
    simulated arm to its end pose. Returns an Execution."""
    if _TAPE is not None:
        _TAPE.append(("plan", plan))
        arm.conn.raw.set_angles_directly([float(v) for v in plan.q_waypoints[-1]])
        return Execution(ok=True)
    return arm._execute(plan)


def _check(arm, what, target):
    """Log (don't abort) if the tip is not within REACH_TOL_CM of target --
    deferred to playback while rehearsing, since only the real arm can drift."""
    if _TAPE is not None:
        _TAPE.append(("check", what, tuple(target)))
        return
    cur = current_pos(arm)
    if not _within(cur, target, (REACH_TOL_CM,) * 3):
        print(f"  !! tip at {tuple(round(v, 2) for v in cur)}, expected "
              f"{tuple(round(v, 1) for v in target)} +/- {REACH_TOL_CM} cm ({what})")


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
            plan = step[1]
            ex = arm._execute(plan)
            arm.last_plan, arm.last_execution = plan, ex
            if not ex.ok:
                print(f"  execution ABORTED -- {ex.error}")
                return False
            if ex.late_deadlines:
                print(f"  !! {ex.late_deadlines} late control-loop deadline(s) during this move")
        elif kind == "check":
            _check(arm, step[1], step[2])
    return True


def _send_arc(arm, pts, durs, label):
    """Parabolic move: plan it, then hand the Plan to _do_plan(), so the same
    code serves the rehearsal (record) and a live move (execute).
    pts[0] is the implicit start (not sent). Returns bool."""
    rx, ry = PICK_ORIENTATION_DEG[:2]
    if len(pts) < 2:
        _say(f"  {label}: negligible, skipped")
        return True
    pl = arm.plan_path(
        x=[p[0] for p in pts[1:]], y=[p[1] for p in pts[1:]], z=[p[2] for p in pts[1:]],
        rx=rx, ry=ry, rz=_yaw(pts[1:]), durations=list(durs),
    )
    if not pl.ok:
        _say(f"  {label}: plan REFUSED -- {pl.error}")
        return False
    ex = _do_plan(arm, pl)
    if not ex.ok:
        _say(f"  {label}: execution ABORTED -- {ex.error}")
        return False
    _say(f"  {label}: {pl.path_length_cm:.1f} cm, {pl.duration_s:.2f} s, "
         f"peak {pl.peak_joint_dps:.0f} deg/s")
    if ex.late_deadlines:
        _say(f"  !! {ex.late_deadlines} late control-loop deadline(s) during this arc")
    return True


# ---- LOOP_REACH_CYCLE: joint-space moves -------------------------------------

def _ticks(total_s):
    """Setpoint times for a move of total_s: one per control tick, ends included."""
    n = max(1, int(round(total_s * config.CONTROL_RATE_HZ)))
    return np.linspace(0.0, total_s, n + 1)


def _ease_rows(q0, q1, total_s):
    """(t, rows): rest-to-rest smootherstep from joint pose q0 to q1."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    t = _ticks(total_s)
    return t, q0[None, :] + (q1 - q0)[None, :] * _smootherstep(t / total_s)[:, None]


def _spline_rows(knots, total_s):
    """(t, rows): one joint-space curve through every pose in `knots` -- at
    rest at the first and last, flowing through the ones between (Catmull-Rom
    tangents, knots spaced by joint distance), timed by smootherstep. Rows are
    linear blends of the knots, so if every knot points the gripper down
    (J2+J3+J4 and J1-J6 constant) the whole curve does too."""
    K = np.asarray(knots, dtype=float)
    seg = np.linalg.norm(np.diff(K, axis=0), axis=1)
    u_k = np.concatenate([[0.0], np.cumsum(seg)]) / max(float(seg.sum()), 1e-9)
    m = [np.zeros(K.shape[1])]
    m += [(K[i + 1] - K[i - 1]) / max(u_k[i + 1] - u_k[i - 1], 1e-9) for i in range(1, len(K) - 1)]
    m += [np.zeros(K.shape[1])]
    t = _ticks(total_s)
    rows = []
    for u in _smootherstep(t / total_s):
        i = int(min(max(np.searchsorted(u_k, u, side="right") - 1, 0), len(K) - 2))
        h = max(u_k[i + 1] - u_k[i], 1e-9)
        s = (u - u_k[i]) / h
        rows.append((2 * s ** 3 - 3 * s ** 2 + 1) * K[i] + (s ** 3 - 2 * s ** 2 + s) * h * m[i]
                    + (-2 * s ** 3 + 3 * s ** 2) * K[i + 1] + (s ** 3 - s ** 2) * h * m[i + 1])
    return t, np.array(rows)


def _joint_move(arm, t, rows, label):
    """Run (or record) a joint-space move; rows[0] is the arm's current pose.
    Goes straight to _do_plan(), so the joint limits and MAX_JOINT_SPEED_DPS
    are checked here. Returns bool."""
    for r in rows[1:]:
        problems = check_joint_limits(r)
        if problems:
            _say(f"  {label} REFUSED -- violates joint limits: {'; '.join(problems)}")
            return False
    dps = np.max(np.abs(np.diff(rows, axis=0)) / np.diff(t)[:, None], axis=0)
    over = [f"J{j + 1} {v:.0f} deg/s" for j, v in enumerate(dps) if v > config.MAX_JOINT_SPEED_DPS[j]]
    if over:
        _say(f"  {label} REFUSED -- too fast for the hardware: {', '.join(over)}")
        return False
    plan = Plan(ok=True, q_waypoints=rows, timestamps=t, duration_s=float(t[-1]),
                peak_joint_dps=float(dps.max()))
    ex = _do_plan(arm, plan)
    if not ex.ok:
        _say(f"  {label}: execution ABORTED -- {ex.error}")
        return False
    _say(f"  {label}: {t[-1]:.2f} s, peak {dps.max():.0f} deg/s")
    if ex.late_deadlines:
        _say(f"  !! {ex.late_deadlines} late control-loop deadline(s) during this move")
    return True


def _down_q(xyz, seed):
    """Joint pose with the tip at xyz (cm) and the gripper pointing down
    (PICK_ORIENTATION_DEG / ORIENT_LOCK), IK-seeded from `seed`; None if
    unreachable."""
    rx, ry = PICK_ORIENTATION_DEG[:2]
    d = np.array([xyz[0] * 10.0, xyz[1] * 10.0, xyz[2] * 10.0, rx, ry, _yaw_one(xyz[:2])])
    if config.Z_RELATIVE_TO_JOINT1:
        d[2] += config.JOINT_1_HEIGHT_CM * 10.0
    return ik.try_solve(d, np.ones(6, dtype=bool), np.asarray(seed, dtype=float))


def _loop_reach(arm, cube, pending, label, grip_deg, action):
    """LOOP_REACH_CYCLE: forward + full stretch, up over the top, down to above
    `cube`, straight down onto it, close. `pending` is a gripper action
    deferred from the previous carry -- it fires during stage 1. Returns bool."""
    above = (cube[0], cube[1], cube[2] + LOOP_ABOVE_CUBE_CM)
    q_top = _down_q(LOOP_TOP_CM, HOME)
    q_above = _down_q(above, q_top) if q_top is not None else None
    if q_above is None:
        _say(f"  {label}: loop REFUSED -- "
             f"{'LOOP_TOP_CM' if q_top is None else 'the point above the cube'} "
             f"is unreachable pointing down")
        return False

    # stages 1-2: forward + full stretch (split where a deferred gripper action fires)
    q0 = np.asarray(arm.get_angles(), dtype=float)
    q_s = np.asarray(LOOP_STRETCH_Q, dtype=float)
    tick = 1.0 / config.CONTROL_RATE_HZ
    if pending is not None:
        p_deg, p_label, p_off = pending
        t_fire = float(np.clip(p_off, 0.0, LOOP_STRETCH_S))
        q_mid = q0 + (q_s - q0) * float(_smootherstep(t_fire / LOOP_STRETCH_S))
        if t_fire >= tick and not _joint_move(arm, *_ease_rows(q0, q_mid, t_fire),
                                              f"{label} forward (deferred head)"):
            return False
        if not _grip(arm, p_deg, f"{p_label} (fired late)"):
            return False
        q0, rest_s = q_mid, LOOP_STRETCH_S - t_fire
    else:
        rest_s = LOOP_STRETCH_S
    if rest_s >= tick and not _joint_move(arm, *_ease_rows(q0, q_s, rest_s),
                                          f"{label} forward + full stretch"):
        return False
    _pause(LOOP_STRETCH_HOLD_S)

    # stages 3-4: up through LOOP_TOP_CM, over and down to above the cube
    if not _joint_move(arm, *_spline_rows([q_s, q_top, q_above], LOOP_UP_OVER_S),
                       f"{label} up + over"):
        return False

    # straight down onto the cube, eased over equal time steps
    n = max(int(LOOP_LAND_WAYPOINTS), 2)
    zs = above[2] + (cube[2] - above[2]) * _smootherstep(np.linspace(0.0, 1.0, n + 1))
    pts = [(cube[0], cube[1], float(z)) for z in zs]
    if not _send_arc(arm, pts, [LOOP_LAND_S / n] * n, f"{label} landing"):
        return False
    return _grip(arm, grip_deg, f"{action} (loop)")


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
        if len(pts) < 2 or (ci == LOOP_REACH_CYCLE and seg["kind"] == "reach"):
            continue                       # the loop is joint-space -- the rehearsal checks it
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

def _choreography(arm, segments, paths, all_durs, styles, rng):
    """The whole run, cycle by cycle. Runs once on a simulated arm with
    _TAPE recording (the rehearsal), and the tape is then played back on the
    real arm -- see THE TAPE above. Returns bool."""
    if not _grip(arm, GRIP_OPEN_DEG, "open before first pick"):
        return False

    pending = None   # (grip_deg, action_label, offset_s) deferred from the previous cycle

    for ci, seg in enumerate(segments):
        kind = seg["kind"]
        pts, durs = paths[ci], all_durs[ci]

        if kind == "leadout":
            _say(f"\n=== lead-out arc -> HOME ===  [{styles[ci]}]")
            _pause(LEADOUT_PAUSE_S)
            if pending is not None:
                p_deg, p_label, p_off = pending
                split = _head_split_index(durs, p_off)
                if split > 0:
                    if not _send_arc(arm, pts[:split + 1], durs[:split], "lead-out (deferred head)"):
                        return False
                    pts, durs = pts[split:], durs[split:]
                if not _grip(arm, p_deg, f"{p_label} (fired late)"):
                    return False
                pending = None
            _send_arc(arm, pts, durs, "lead-out")
            break

        grip_deg = GRIP_CLOSED_DEG if kind == "reach" else GRIP_OPEN_DEG
        action = "close on cube" if kind == "reach" else "release cube"
        label = "reach & grasp" if kind == "reach" else "carry & place"
        _say(f"\n=== cycle {ci}/{N_CYCLES - 1}  {label}  "
             f"cube #{seg['k'] + 1}  -> {tuple(round(v, 1) for v in seg['target'])}  "
             f"[{styles[ci]}] ===")

        # LOOP_REACH_CYCLE: a scripted joint-space loop instead of the drawn arc;
        # it fires anything deferred from the previous cycle itself
        if ci == LOOP_REACH_CYCLE and kind == "reach":
            if not _loop_reach(arm, seg["target"], pending, label, grip_deg, action):
                return False
            pending = None
            _check(arm, "cube", seg["target"])
            continue

        # fire anything deferred from the PREVIOUS cycle, near the start of this arc
        if pending is not None:
            p_deg, p_label, p_off = pending
            split = _head_split_index(durs, p_off)
            if split > 0:
                if not _send_arc(arm, pts[:split + 1], durs[:split], f"{label} (deferred head)"):
                    return False
                pts, durs = pts[split:], durs[split:]
            if not _grip(arm, p_deg, f"{p_label} (fired late)"):
                return False
            pending = None

        # this cycle's own gripper action -- DELAYED_OPEN_CYCLE forces a
        # deterministic, tunable-fraction defer instead of the usual random offset
        if ci == DELAYED_OPEN_CYCLE and kind == "carry":
            if not _send_arc(arm, pts, durs, label):
                return False
            next_durs = all_durs[ci + 1] if ci + 1 < len(all_durs) else []
            forced_offset = DELAYED_OPEN_FRACTION * sum(next_durs)
            pending = (grip_deg, action, forced_offset)
            _say(f"  gripper action DELIBERATELY delayed to "
                 f"{DELAYED_OPEN_FRACTION * 100:.0f}% into the next leg")
        elif ci == EARLY_DROP_CYCLE and kind == "carry":
            # deterministic early release, partway through THIS carry's own arc
            lead_s = (1.0 - EARLY_DROP_FRACTION) * sum(durs)
            split = _tail_split_index(durs, lead_s)
            if split > 0:
                if not _send_arc(arm, pts[:split + 1], durs[:split], f"{label} (approach)"):
                    return False
                pts, durs = pts[split:], durs[split:]
            if not _grip(arm, grip_deg, f"{action} (dropped at {EARLY_DROP_FRACTION * 100:.0f}%)"):
                return False
            if not _send_arc(arm, pts, durs, f"{label} (final)"):
                return False
            _say(f"  gripper action DELIBERATELY dropped "
                 f"{EARLY_DROP_FRACTION * 100:.0f}% into this leg")
        elif ci in NORMAL_GRIP_CYCLES:
            # deterministic, reliable close / release right at arrival (no early/late chaos)
            split = _tail_split_index(durs, 0.0)
            if split > 0:
                if not _send_arc(arm, pts[:split + 1], durs[:split], f"{label} (approach)"):
                    return False
                pts, durs = pts[split:], durs[split:]
            if not _grip(arm, grip_deg, f"{action} (normal grip)"):
                return False
            if not _send_arc(arm, pts, durs, f"{label} (final)"):
                return False
        else:
            # timed with a fresh random offset
            offset = float(rng.uniform(-GRIP_CHAOS_SPREAD_S, GRIP_CHAOS_SPREAD_S))
            if offset <= 0.0:
                split = _tail_split_index(durs, -offset)
                if split > 0:
                    if not _send_arc(arm, pts[:split + 1], durs[:split], f"{label} (approach)"):
                        return False
                    pts, durs = pts[split:], durs[split:]
                if not _grip(arm, grip_deg, f"{action} (offset {offset:+.2f}s)"):
                    return False
                if not _send_arc(arm, pts, durs, f"{label} (final)"):
                    return False
            else:
                if not _send_arc(arm, pts, durs, label):
                    return False
                pending = (grip_deg, action, offset)
                _say(f"  gripper action deferred ~{offset:.2f}s into the next leg")

        _check(arm, "cube" if kind == "reach" else "target", seg["target"])

    return True


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
    ap.add_argument("--recompile", action="store_true",
                    help="ignore the compiled run in compiled/ and rehearse it again")
    ap.add_argument("--compile-only", action="store_true",
                    help="rehearse (or check) the compiled run, save it, and exit -- "
                         "no arm is connected")
    args = ap.parse_args()

    if args.rviz and not args.mock:
        print("--rviz is mock-only; ignoring it (add --mock to visualize).")
        args.rviz = False

    if len(CUBES_INITIAL_POINTS) != len(TARGET_POSITIONS):
        print("CUBES_INITIAL_POINTS and TARGET_POSITIONS must be the same length.")
        return 1
    if DIAGONAL_ARC_CYCLE >= 0 and DIAGONAL_ARC_CYCLE % 2 != 0:
        print(f"DIAGONAL_ARC_CYCLE must be an EVEN (reach) cycle, got {DIAGONAL_ARC_CYCLE}.")
        return 1
    if DELAYED_OPEN_CYCLE >= 0 and DELAYED_OPEN_CYCLE % 2 != 1:
        print(f"DELAYED_OPEN_CYCLE must be an ODD (carry) cycle, got {DELAYED_OPEN_CYCLE}.")
        return 1
    if ROLLERCOASTER_CYCLE >= 0 and ROLLERCOASTER_CYCLE % 2 != 1:
        print(f"ROLLERCOASTER_CYCLE must be an ODD (carry) cycle, got {ROLLERCOASTER_CYCLE}.")
        return 1
    if EARLY_DROP_CYCLE >= 0 and EARLY_DROP_CYCLE % 2 != 1:
        print(f"EARLY_DROP_CYCLE must be an ODD (carry) cycle, got {EARLY_DROP_CYCLE}.")
        return 1
    if LOOP_REACH_CYCLE >= 0 and LOOP_REACH_CYCLE % 2 != 0:
        print(f"LOOP_REACH_CYCLE must be an EVEN (reach) cycle, got {LOOP_REACH_CYCLE}.")
        return 1

    if not args.mock and not args.yes and not args.compile_only:
        print("This will move the robot arm and actuate the gripper. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(CUBES_INITIAL_POINTS)
    order = list(rng.permutation(n)) if SHUFFLE_ORDER else PICKUP_ORDER

    p_home = pose_coords(HOME)                       # mm/deg (Z_RELATIVE_TO_JOINT1 assumed False)
    home_tip = (p_home[0] / 10.0, p_home[1] / 10.0, p_home[2] / 10.0)

    global _AZ_REF                                   # rz = PICK_ORIENTATION_DEG[2] at HOME's azimuth
    _AZ_REF = (home_tip[0], home_tip[1])

    print(f"pick order (cube indices): {[int(k) for k in order]}")
    print(f"orientation lock: {ORIENT_LOCK}")
    print("no nudge-reaction code in this profile -- a live nudge gets no response, by design")

    segments = _build_segments(order, home_tip)

    # ---- the run: load the compiled one, or rehearse it now on a simulated arm ----
    # (see _compiled.py -- reused until any source it depends on changes)
    compiled = None if args.recompile else _compiled.load(__file__, RANDOM_SEED)
    if compiled is not None:
        paths, all_durs, styles, tape = (compiled[k] for k in ("paths", "all_durs", "styles", "tape"))
        n_moves = sum(1 for step in tape if step[0] == "plan")
        print(f"loaded the compiled run ({n_moves} motions) -- skipping the rehearsal")
    else:
        global _TAPE
        print("\nrehearsing the whole run on a simulated arm (nothing moves yet)...")
        t_rehearse = time.perf_counter()
        sim = Arm(mock=True, quiet_mock=True)          # the rehearsal arm -- no MOCK warning
        sim.conn.raw.set_angles_directly([float(v) for v in HOME])
        try:
            # ---- every arc + its durations (style/ease/speed all fresh per move) ----
            # _draw_valid_move() validates each candidate arc against arm.plan_path()
            # (reachability + MAX_JOINT_SPEED_DPS) before committing to it -- see the
            # note above EASE_MIN/EASE_MAX. Purely planning; no motion happens here.
            # plan_path() plans from the arm's CURRENT pose, so the number of re-rolls
            # (and with it every later draw from the shared rng) depends on where the
            # arm is -- done on the simulated arm at exactly HOME (where the real arm
            # is after homing), so RANDOM_SEED really reproduces the run.
            paths, all_durs, styles = [], [], []
            for ci, seg in enumerate(segments):
                if ci == DIAGONAL_ARC_CYCLE and seg["kind"] == "reach":
                    pts, durs, desc = _draw_valid_diagonal(sim, seg["origin"], seg["target"], rng)
                elif ci == ROLLERCOASTER_CYCLE and seg["kind"] == "carry":
                    pts, durs, desc = _draw_valid_rollercoaster(sim, seg["origin"], seg["target"], rng)
                else:
                    cruise_override = LEADOUT_SPEED_CM_S if seg["kind"] == "leadout" else None
                    pts, durs, desc = _draw_valid_move(sim, seg["origin"], seg["target"], rng,
                                                       cruise_override)
                if ci == LOOP_REACH_CYCLE and seg["kind"] == "reach":
                    # still drawn above, so the shared rng -- and every later arc --
                    # stays the same as without the loop. The loop itself is planned
                    # in _loop_reach(); here: a preview path for RViz, and its stage
                    # 1 duration, which a deferred gripper action (DELAYED_OPEN_
                    # FRACTION of "the next leg") is measured against.
                    t = seg["target"]
                    p_s = pose_coords(LOOP_STRETCH_Q)
                    pts = [seg["origin"], (p_s[0] / 10.0, p_s[1] / 10.0, p_s[2] / 10.0),
                           tuple(LOOP_TOP_CM), (t[0], t[1], t[2] + LOOP_ABOVE_CUBE_CM), t]
                    durs, desc = [LOOP_STRETCH_S], "loop: forward + stretch, up, over, down"
                paths.append(pts)
                all_durs.append(durs)
                styles.append(desc)

            _TAPE = []
            try:
                ok = _choreography(sim, segments, paths, all_durs, styles, rng)
            finally:
                tape, _TAPE = _TAPE, None
        finally:
            sim.close()
        if not ok:
            for step in tape:                      # show how far the rehearsal got
                if step[0] == "say":
                    print(step[1])
            print("\nrehearsal failed (see above) -- nothing moved.")
            return 1
        n_moves = sum(1 for step in tape if step[0] == "plan")
        print(f"precomputed {n_moves} motions in {time.perf_counter() - t_rehearse:.1f} s")
        saved = _compiled.save(__file__, RANDOM_SEED,
                               {"paths": paths, "all_durs": all_durs, "styles": styles, "tape": tape})
        if saved:
            print(f"compiled -> {_os.path.relpath(saved)}")
    if args.compile_only:
        return 0

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
    bridge = None

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

        if PREFLIGHT and not preflight(arm, segments, paths, all_durs):
            print("\npreflight failed -- fix the cube/target coordinates or PICK_ORIENTATION_DEG. "
                  "Nothing moved.")
            go_home(arm)
            return 1

        # ---- playback: no computing between actions --------------------------
        if not _play(arm, tape):
            print("\naborting run."); go_home(arm); return 1

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
