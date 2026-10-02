#!/usr/bin/env python3
"""
MOVEMENT PROFILE: High-human / High-robot  (parabolic + deterministic)
=====================================================================

Move cubes from one side of the frame to the other, fast and precise, tracing
the exact same fluid arc every run -- a skilled operator / a well-tuned
collaborative robot:

  * every move is a symmetric arc (shape set by ARC_SHAPE_DEGREE -- 2 is a
    classic PARABOLA, 4 or 6 flatten the top and steepen the sides) whose APEX
    reaches a FIXED world height, one PER ARC from TRAJECTORY_MAX_HEIGHTS_CM --
    unlike HighH-LowR, where a long carry lifts much higher than a short one;
  * each arc is traversed at CONSTANT tip speed -- no ease-in / ease-out
    (CRUISE_SPEED_CM_S; the final lead-out to HOME at LEADOUT_SPEED_CM_S). Each
    arc still starts and ends at a full stop, so the very first / last setpoint
    of an arc unavoidably ramps;
  * every run is IDENTICAL -- no shuffle, no per-move variation (SHUFFLE_ORDER
    off, VARIATION 0). The same trajectory, every time;
  * the cubes are grabbed strictly LEFT -> RIGHT (order = CUBES_INITIAL_POINTS
    as listed);
  * they are dropped in a CLEAN, EVENLY-SPACED row at a uniform height
    (CUBES_TARGET_POINTS) -- no overshoot: has_reached_* gates the gripper, and
    the tip settles on the target before it fires;
  * a scripted NUDGE / flinch (as the arm nears a cube it RECOILS, waits for the
    cube to "settle", then grabs it at its new position) is AVAILABLE but
    DISABLED here (NUDGE_ENABLED = False). Set NUDGE_ENABLED = True to enable it;
    NUDGED_CUBE picks WHICH cube. Fully scripted -- the arm has no sensors.

    python3 scripts/Profiles/HighH-HighR.py --mock --yes      # no hardware
    python3 scripts/Profiles/HighH-HighR.py --port /dev/ttyTHS1

CYCLES
------
N_CYCLES = 2 * (number of cubes). Even cycle 2k = reach cube k and close;
odd cycle 2k+1 = carry cube k to its target and open. A final lead-out arc
returns to HOME.

GRIPPER TIMING
--------------
By default the gripper fires at the END of each reach / carry cycle (a clean
pause at the cube, like a hand). Optionally a TRIGGER_BOXES entry fires it
*during* a cycle, the moment the tip enters the box: that cycle's send_path
then runs on a background thread so the arm never stops. Empty TRIGGER_BOXES
(the default) keeps everything single-threaded.

Everything you tune is a CONSTANT below. The cube coordinates,
PICK_ORIENTATION_DEG and TRAJECTORY_MAX_HEIGHTS_CM are PLACEHOLDERS -- measure
them on your arm first.
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
PREFLIGHT = False #True
RANDOM_SEED = 0                   # fixed -> the same run every time ("High Robot").
                                 # Inert while VARIATION = 0, but explicit.

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

# -- arc shape -------------------------------------------------------------------
# z(u) = z0 + (z1-z0)*u + h*(1 - |2u-1|^ARC_SHAPE_DEGREE) -- ARC_SHAPE_DEGREE=2 is
# the classic parabola (and reduces to exactly the old 4*h*u*(1-u) formula); 4 or 6
# flattens the top and steepens the sides ("more square"), which also means less
# joint travel spent easing toward/away from the peak. Only affects the NORMAL
# reach/carry/leadout arcs below -- run_nudge()'s recoil hop and re-approach arc
# always force degree=2, unchanged.
ARC_SHAPE_DEGREE = 4

# -- arc + velocity profile --------------------------------------------------
# Every arc is a symmetric vertical bump over the straight xy chord (shape set by
# ARC_SHAPE_DEGREE above). Unlike HighH-LowR's shared parabola (apex scaled by
# chord length), here each arc's target apex is read from TRAJECTORY_MAX_HEIGHTS_CM
# -- one world-z height PER ARC, in run order (reach0, carry0, reach1, carry1, ...,
# leadout) -- so every arc can reach its own height, or all the same if you fill
# the list with one repeated value. _apex_h() back-solves (exactly, by bisection
# -- see its docstring) the above-chord height h that puts a given arc's actual
# peak at its target apex, for any chord slope and any ARC_SHAPE_DEGREE.
# Keep these heights reachable at PICK_ORIENTATION_DEG: gripper-down the arm runs
# out of reach around world z ~ 17-18 cm near the workspace edge.
TRAJECTORY_MAX_HEIGHTS_CM = [8.0] * 7  # one per
                                # arc, in run order -- length MUST equal len(segments)
                                # == 2*len(CUBES_INITIAL_POINTS)+1 (reach/carry per
                                # cube, plus the leadout); same value everywhere
                                # reproduces the old single-APEX_Z_CM behavior.
                                # Index 0 (reach0, HOME -> cube0) needs to stay clearly
                                # above HOME's own tip height (world z ~10.3cm) or that
                                # arc collapses to a near-straight line -- see _apex_h()'s
                                # docstring.
MIN_ARC_HEIGHT_CM = 2.0          # floor, so a near-flat arc still clears the table / cubes
# CONSTANT tip speed -- get_durations gives every arc equal per-segment times, so
# there is no ease-in / ease-out. Check each arc's "peak N deg/s" in a --mock run
# against config.MAX_JOINT_SPEED_DPS; lower toward 25 / 20 if any arc is refused
# ("segment ... too fast for the hardware") or nears a joint limit.
CRUISE_SPEED_CM_S = 18.0
LEADOUT_SPEED_CM_S = 10.0        # the final arc back toward HOME cruises slower / gentler
PATH_WAYPOINTS = 60              # samples per arc
MIN_SEGMENT_S = 0.02

# -- per-move variation -------------------------------------------------------
# Kept at 0 for this profile: "High Robot" means the SAME smooth trajectory
# every run. Raising VARIATION would break that -- do not.
VARIATION = 0                  # [0,1] master scale; 0 = identical arcs every run
APEX_HEIGHT_JITTER_FRAC = 0.0    # +/- fraction of an arc's own apex height
BOW_JITTER_CM = 0.0             # +/- sideways bow, perpendicular to the chord

# -- order ------------------------------------------------------------------------
SHUFFLE_ORDER = False            # deterministic: grab cubes left -> right, as listed

# -- scripted nudge / flinch ------------------------------------------------------
# DISABLED for this version. To enable: set NUDGE_ENABLED = True; NUDGED_CUBE picks
# WHICH cube (0/1/2) -- it fires on that cube's own reach, wherever pick order puts
# it. run_nudge then does approach-to-fraction -> recoil hop -> settle -> re-approach
# the moved cube.
NUDGE_ENABLED = True             # True enables the scripted nudge
NUDGED_CUBE = 1                  # 0, 1, or 2 -- which CUBES_INITIAL_POINTS cube gets
                                 # nudged; drives both the scripted recoil target and the
                                 # yellow RViz preview cube
NUDGE_OFFSET_CM = (2.0, 0.0, 0.0)   # where the nudged cube ends up (relative to its point)
NUDGE_AT_FRACTION = 0.9         # fraction of the reach arc completed before the recoil
NUDGE_RECOIL_CM = 4.0           # how far the arm hops back
NUDGE_RECOIL_ARC_HEIGHT_CM = 0.0   # recoil path's apex above its own chord -- small, so
                                 # it reads as a fast near-straight hop, not a lofted arc
NUDGE_RECOIL_WAYPOINTS = 10      # fewer than PATH_WAYPOINTS -- get_durations floors a move's
                              # total time at (n_waypoints-1)*MIN_SEGMENT_S regardless of
                              # cruise speed, so the recoil's short hop needs far fewer
                              # segments than a full reach/carry arc to actually reach
                              # NUDGE_RECOIL_SPEED_CM_S instead of being floored near it
NUDGE_RECOIL_SPEED_CM_S = 25.0  # the recoil is fast
POST_NUDGE_ARC_HEIGHT_CM = 2.0   # re-approach-to-the-moved-cube arc's apex above its own
                                # chord -- bypasses _apex_h()'s fixed-world-z target so this
                                # short hop stays a gentle curve instead of lifting toward
                                # that arc's TRAJECTORY_MAX_HEIGHTS_CM entry and back down
POST_NUDGE_SPEED_CM_S = 14.0    # re-approach cruise speed -- its own dial, independent of
                                # CRUISE_SPEED_CM_S (which every normal arc uses), so it can
                                # be tuned without also speeding up the rest of the run
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

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 120.0           # 0 = closed .. config.MAX_GRIPPER_DEG = full open
GRIP_CLOSED_DEG = 65.0          # tune to the cube width
GRIP_SPEED = 90  #config.GRIPPER_DEFAULT_SPEED
GRIP_SETTLE_S = 0.35           # quiet time after a gripper command: it must LAND and the
                              # jaws start moving. Tunable down to GRIP_MIN_GAP_S, not below.
GRIP_MIN_GAP_S = 0.2          # hard floor -- pymycobot silently drops a gripper command
                              # that is not followed by a short quiet gap (why 0.0 failed).
REACH_TOL_CM = 6.0             # has_reached_* tolerance, per axis
LEADOUT_PAUSE_S = 0.5         # deliberate beat between the last release and homing

# -- gripper trigger boxes (optional) ------------------------------------------
# [[(cx,cy,cz), (l,w,h), cycle_n], ...] -- on cycle cycle_n, the gripper fires
# the moment the tip enters this box (that cycle runs on a background thread so
# the arm keeps moving). Empty -> gripper always fires at the cycle end.
TRIGGER_BOXES = []

N_CYCLES = len(CUBES_INITIAL_POINTS) * 2

_AZ_REF = None                    # (x, y) tip position whose azimuth is rz's zero; set in main()


# ===========================================================================
# GEOMETRY / PROFILE HELPERS
# ===========================================================================

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


def _apex_h(origin, target, apex_z):
    """Above-the-chord height h that puts the TOP of the arc at world
    z = `apex_z`, for ANY chord (level or sloped) at ANY ARC_SHAPE_DEGREE.

    z(u) = z0 + (z1-z0)*u + h*(1-|2u-1|^n). On the half where the linear term
    climbs (v = 2u-1 >= 0, dz = |z1-z0|), its interior maximum is at
        v* = (dz / (2*h*n)) ** (1/(n-1))          (clipped to 1 -- beyond that
    the arc has no interior peak, it just climbs to the higher endpoint), which
    gives a peak this far above the chord's midpoint:
        peak_above_mid(h) = (dz/2)*v* + h*(1 - v**n).
    This is monotonically increasing in h, so the h that makes it equal
    `apex_z - (z0+z1)/2` is found by bisection. This is EXACT (unlike a fixed
    formula per degree) -- it reduces to the old closed-form quadratic result at
    n=2, and to the simple h = apex_z - z0 for a level chord (v*=0) at any n.
    (An earlier version of this function used the level-chord formula for every
    chord regardless of degree, which is only exact when level -- on the two
    sloped arcs here, HOME<->cube, it let the actual peak overshoot `apex_z` by
    several cm at degree > 2, enough to put a waypoint out of reach.) Floored at
    MIN_ARC_HEIGHT_CM so a near-flat arc still humps clear of the table."""
    z0, z1 = float(origin[2]), float(target[2])
    dz = abs(z1 - z0)
    n = float(ARC_SHAPE_DEGREE)
    target_above_mid = apex_z - 0.5 * (z0 + z1)
    if target_above_mid <= 0.0:
        return MIN_ARC_HEIGHT_CM

    def peak_above_mid(h):
        if h <= 1e-9:
            return dz / 2.0     # no bump: z(u) is just the linear chord, peak at the high end
        if dz <= 1e-9:
            return h            # level chord: v*=0 trivially, peak is exactly h at u=0.5
        v = min((dz / (2.0 * h * n)) ** (1.0 / (n - 1.0)), 1.0)
        return 0.5 * dz * v + h * (1.0 - v ** n)

    lo, hi = 0.0, max(target_above_mid, 1.0)
    while peak_above_mid(hi) < target_above_mid and hi < 1e6:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if peak_above_mid(mid) < target_above_mid:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    return float(max(h, MIN_ARC_HEIGHT_CM))


def _parabola_points(origin, target, arc_height, rng, n_waypoints=None, degree=None):
    """(list of `n_waypoints` (x,y,z), arc_length) along the arc from `origin`
    to `target`: the straight xy chord + a symmetric vertical bump of apex
    `arc_height` above the chord (`degree`=2 is a classic parabola; 4 or 6
    flatten the top and steepen the sides -- "more square"). With `rng` the
    apex height and a sideways bow are jittered by VARIATION.
    `n_waypoints` defaults to PATH_WAYPOINTS -- pass fewer for a short, fast
    move (e.g. the recoil) where PATH_WAYPOINTS segments would floor its total
    duration at (PATH_WAYPOINTS-1)*MIN_SEGMENT_S regardless of cruise speed.
    `degree` defaults to ARC_SHAPE_DEGREE -- the recoil/re-approach arcs in
    run_nudge() pass degree=2 explicitly to stay a plain parabola."""
    n_waypoints = PATH_WAYPOINTS if n_waypoints is None else int(n_waypoints)
    degree = ARC_SHAPE_DEGREE if degree is None else degree
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

    # straight chord in z + symmetric bump (0 at both ends, peak h at u=0.5);
    # degree=2 reduces to exactly 4*h*u*(1-u), the classic parabola
    z = z0 + (z1 - z0) * u + h * (1.0 - np.abs(2.0 * u - 1.0) ** degree)

    dense = np.column_stack([xy, z])
    L = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    pts, _ = _polyline_points(dense, np.linspace(0.0, L, n_waypoints))
    return [tuple(float(v) for v in p) for p in pts], L


def get_path(origin_point, target_point, arc_height, rng=None, n_waypoints=None, degree=None):
    """Arc from origin to target as `n_waypoints` (x,y,z) points (default
    PATH_WAYPOINTS). `arc_height` is this move's apex above the chord --
    compute it with _apex_h(). `degree` defaults to ARC_SHAPE_DEGREE."""
    return _parabola_points(origin_point, target_point, arc_height, rng,
                            n_waypoints, degree)[0]


def get_durations(origin_point, target_point, arc_height,
                  cruise=CRUISE_SPEED_CM_S, n_waypoints=None, degree=None):
    """`n_waypoints`-1 EQUAL segment durations (n_waypoints defaults to
    PATH_WAYPOINTS -- must match whatever `n_waypoints` get_path() was called
    with for the same arc) -> CONSTANT tip speed, no ease. _parabola_points
    resamples the arc to equal arc-length steps, so equal time per step ==
    constant speed along the arc. Each arc still begins and ends at a full stop
    (the gripper fires between arcs), so the rest-to-rest Hermite blend still
    ramps the very first / last segment -- 'constant' is through the interior."""
    n_waypoints = PATH_WAYPOINTS if n_waypoints is None else int(n_waypoints)
    _, L = _parabola_points(origin_point, target_point, arc_height, None, n_waypoints, degree)
    seg_t = (L / max(cruise, 1e-6)) / (n_waypoints - 1)
    return [max(seg_t, MIN_SEGMENT_S)] * (n_waypoints - 1)


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
    move_joints -- no speed pre-check -- outrun the servos and shake)."""
    try:
        dq = max(abs(a - b) for a, b in zip(arm.get_angles(), HOME))
    except Exception:
        dq = 0.0
    dur = max(HOME_MOVE_S, dq / HOME_RETURN_DPS)
    if dur > HOME_MOVE_S + 0.05:
        print(f"  homing over {dur:.1f}s (joint travel {dq:.0f} deg)")
    arm.move_joints(HOME, duration=dur)


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


def _send_arc_with_trigger(arm, pts, durs, t_fire, grip_deg, label):
    """Run the arc on a background thread and fire the gripper at t_fire so the
    arm never stops."""
    rx, ry = PICK_ORIENTATION_DEG[:2]
    kw = dict(x=[p[0] for p in pts[1:]], y=[p[1] for p in pts[1:]],
              z=[p[2] for p in pts[1:]], rx=rx, ry=ry, rz=_yaw(pts[1:]),
              durations=list(durs))
    th = _PathThread(arm, kw)
    t0 = time.perf_counter()
    th.start()

    while time.perf_counter() - t0 < t_fire and th.is_alive():
        time.sleep(0.02)

    fired = False
    if th.is_alive() and th.exc is None:
        print(f"  {label}: trigger box entered ~t={time.perf_counter()-t0:.2f}s "
              f"-> gripper {grip_deg:.0f}")
        _fire_gripper(arm, grip_deg)
        fired = True

    th.join()
    if th.exc is not None:
        raise th.exc
    if not th.result:
        print(f"  {label}: send_path (threaded) REFUSED -- {arm.last_error}")
        return False
    if not fired:
        print(f"  {label}: path ended before the box -- firing gripper {grip_deg:.0f} now")
        _fire_gripper(arm, grip_deg)
    time.sleep(max(GRIP_SETTLE_S, GRIP_MIN_GAP_S))
    pl = arm.last_plan
    print(f"  {label}: {pl.path_length_cm:.1f} cm, {pl.duration_s:.2f} s, "
          f"peak {pl.peak_joint_dps:.0f} deg/s")
    return True


def _trigger_time(pts, durs, cycle_n):
    """Cumulative time at the first arc sample inside an active trigger box for
    this cycle, or None."""
    t = 0.0
    for i in range(1, len(pts)):
        t += durs[i - 1]
        if is_in_trigger_box(pts[i], cycle_n):
            return t
    return None


def run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs):
    """Scripted flinch: approach part-way, recoil, wait, re-approach the moved cube."""
    n = len(pts)
    cut = max(2, int(round(NUDGE_AT_FRACTION * (n - 1))) + 1)
    print(f"  NUDGE: approaching to {int(NUDGE_AT_FRACTION*100)}% ...")
    if not _send_arc(arm, pts[:cut], durs[:cut - 1], "  nudge approach"):
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
                    n_waypoints=NUDGE_RECOIL_WAYPOINTS, degree=2)
    rdurs = get_durations(here, recoil, NUDGE_RECOIL_ARC_HEIGHT_CM,
                          cruise=NUDGE_RECOIL_SPEED_CM_S, n_waypoints=NUDGE_RECOIL_WAYPOINTS,
                          degree=2)
    arm.jerk = NUDGE_RECOIL_JERK
    ok = _send_arc(arm, rpts, rdurs, "  recoil")
    arm.jerk = 0.0
    if not ok:
        return False

    print(f"  waiting {NUDGE_SETTLE_S:.1f}s for the cube to settle ...")
    time.sleep(NUDGE_SETTLE_S)

    new_cube = tuple(float(c + o) for c, o in zip(seg["target"], NUDGE_OFFSET_CM))
    print(f"  cube moved -> re-approaching {tuple(round(v, 1) for v in new_cube)}")
    after = current_pos(arm)
    h2 = POST_NUDGE_ARC_HEIGHT_CM
    p2 = get_path(after, new_cube, h2, rng, n_waypoints=POST_NUDGE_WAYPOINTS, degree=2)
    d2 = get_durations(after, new_cube, h2, cruise=POST_NUDGE_SPEED_CM_S,
                       n_waypoints=POST_NUDGE_WAYPOINTS, degree=2)
    if not _send_arc(arm, p2, d2, "  nudge re-approach"):
        return False

    # the following carry cycle must start from where the cube actually is now
    nxt = ci + 1
    if nxt < len(segments) and segments[nxt]["kind"] == "carry":
        segments[nxt]["origin"] = new_cube
        hc = _apex_h(new_cube, segments[nxt]["target"], TRAJECTORY_MAX_HEIGHTS_CM[nxt])
        paths[nxt] = get_path(new_cube, segments[nxt]["target"], hc, rng)
        all_durs[nxt] = get_durations(new_cube, segments[nxt]["target"], hc)
    return True


# ===========================================================================
# PREFLIGHT
# ===========================================================================

def preflight(arm, segments, paths):
    rx, ry = PICK_ORIENTATION_DEG[:2]
    print("\n--- preflight: planning every cube point + arc apex (no motion) ---")
    checks = []
    for i, (s, t) in enumerate(zip(CUBES_INITIAL_POINTS, CUBES_TARGET_POINTS)):
        checks += [(f"init{i+1}", s), (f"tgt{i+1}", t)]
    if NUDGE_ENABLED:
        # the nudged cube also gets grabbed at init + offset
        for i, s in enumerate(CUBES_INITIAL_POINTS):
            checks.append((f"init{i+1}+nudge",
                           tuple(c + o for c, o in zip(s, NUDGE_OFFSET_CM))))
    for ci, (seg, pts) in enumerate(zip(segments, paths)):
        apex = max(pts, key=lambda p: p[2])         # highest point of the arc
        checks.append((f"apex c{ci}", apex))
    bad = 0
    for name, (x, y, z) in checks:
        pl = arm.plan_coords(x=x, y=y, z=z, rx=rx, ry=ry, rz=_yaw_one((x, y)),
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
                     "target": CUBES_INITIAL_POINTS[k], "k": int(k)})
        segs.append({"kind": "carry", "origin": CUBES_INITIAL_POINTS[k],
                     "target": CUBES_TARGET_POINTS[k], "k": int(k)})
        prev_target = CUBES_TARGET_POINTS[k]
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

    if len(TRAJECTORY_MAX_HEIGHTS_CM) != len(segments):
        print(f"TRAJECTORY_MAX_HEIGHTS_CM must have {len(segments)} entries (one per arc "
              f"in run order), got {len(TRAJECTORY_MAX_HEIGHTS_CM)}.")
        return 1

    print(f"arc shape: degree {ARC_SHAPE_DEGREE:.0f}, "
          f"per-arc max heights {TRAJECTORY_MAX_HEIGHTS_CM}")

    # ---- precompute every arc + its durations --------------------------------
    paths, all_durs = [], []
    for ci, seg in enumerate(segments):
        cruise = LEADOUT_SPEED_CM_S if seg["kind"] == "leadout" else CRUISE_SPEED_CM_S
        h = _apex_h(seg["origin"], seg["target"], TRAJECTORY_MAX_HEIGHTS_CM[ci])
        paths.append(get_path(seg["origin"], seg["target"], h, rng))
        all_durs.append(get_durations(seg["origin"], seg["target"], h, cruise=cruise))

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
        go_home(arm)
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
                print(f"\n=== lead-out arc -> HOME ===")
                time.sleep(LEADOUT_PAUSE_S)
                _send_arc(arm, pts, durs, "lead-out")
                break

            grip_deg = GRIP_CLOSED_DEG if kind == "reach" else GRIP_OPEN_DEG
            label = "reach & grasp" if kind == "reach" else "carry & place"
            print(f"\n=== cycle {ci}/{N_CYCLES - 1}  {label}  "
                  f"cube #{seg['k'] + 1}  -> {tuple(round(v, 1) for v in seg['target'])} ===")

            if kind == "reach" and NUDGE_ENABLED and seg["k"] == NUDGED_CUBE:
                if not run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs):
                    print("\naborting run."); go_home(arm); return 1
                if not _grip(arm, GRIP_CLOSED_DEG, "close on cube (new position)"):
                    return 1
                continue

            t_fire = _trigger_time(pts, durs, ci)
            if t_fire is not None:
                if not _send_arc_with_trigger(arm, pts, durs, t_fire, grip_deg, label):
                    print("\naborting run."); go_home(arm); return 1
                continue

            if not _send_arc(arm, pts, durs, label):
                print("\naborting run."); go_home(arm); return 1

            cur = current_pos(arm)
            reached = (has_reached_cube(cur, seg["target"]) if kind == "reach"
                       else has_reached_target(cur, seg["target"]))
            if reached:
                if not _grip(arm, grip_deg, "close on cube" if kind == "reach" else "release cube"):
                    return 1
            else:
                print(f"  !! tip at {tuple(round(v, 2) for v in cur)}, expected "
                      f"{tuple(round(v, 1) for v in seg['target'])} +/- {REACH_TOL_CM} cm "
                      f"-- gripper NOT fired")

        print("\nall cubes placed. homing...")
        go_home(arm)
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
