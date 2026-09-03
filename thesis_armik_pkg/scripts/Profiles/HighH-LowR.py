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
By default the gripper fires at the END of each reach / carry cycle (a clean
pause at the cube, like a hand). Optionally a TRIGGER_BOXES entry fires it
*during* a cycle, the moment the tip enters the box: that cycle's send_path
then runs on a background thread so the arm never stops. Empty TRIGGER_BOXES
(the default) keeps everything single-threaded.

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
PREFLIGHT = True
RANDOM_SEED = None                # int for a repeatable run, None for fresh each time

# -- cubes (MEASURE AND REPLACE) ----------------------------------------------
# (x, y, z) CM, at the GRIPPER TIP, base frame, z from the table.
CUBES_INITIAL_POINTS = [          # a row on the pick side
    (15.0, 10.0, 0.0),
    (15.0, 13.0, 0.0),
    (15.0, 16.0, 0.0),
    (15.0, 19.0, 0.0),
]
CUBES_TARGET_POINTS = [           # deterministic, deliberately uneven drop points
    (15.0, -10.0, 0.0),
    (15.0, -13.0, 0.5),
    (15.0, -16.0, 0.0),
    (15.0, -19.0, 0.5),
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
MIN_ARC_HEIGHT_CM = 2.0          # floor, so short moves still clear the table / other cubes
CRUISE_SPEED_CM_S = 25.0          # peak tip speed; the ease dials stretch the move time
LEADOUT_SPEED_CM_S = 10.0        # the final arc back toward HOME is slower / gentler
EASE_IN = 5.0                    # [0,10] start-of-move acceleration shape. 0 = abrupt,
EASE_OUT = 5.0                   # [0,10] end-of-move deceleration shape.  10 = long, gentle S
PATH_WAYPOINTS = 30              # samples per arc
MIN_SEGMENT_S = 0.02

# -- per-move variation ("never the same twice") -----------------------------
VARIATION = 0                  # [0,1] master scale; 0 = identical arcs every run
APEX_HEIGHT_JITTER_FRAC = 0.0    # +/- fraction of an arc's own apex height
BOW_JITTER_CM = 0.0             # +/- sideways bow, perpendicular to the chord
EASE_JITTER = 0.0              # +/- on EASE_IN / EASE_OUT per move

# -- order ------------------------------------------------------------------------
SHUFFLE_ORDER = True             # grab cubes in a random order (init<->target pairing kept)

# -- scripted nudge / flinch ------------------------------------------------------
NUDGE_CYCLE = -1                  # EVEN cycle index whose reach is interrupted; -1 disables
NUDGE_OFFSET_CM = (2.0, 0.0, 0.0)   # where the nudged cube ends up (relative to its point)
NUDGE_AT_FRACTION = 0.6         # fraction of the reach arc completed before the recoil
NUDGE_RECOIL_CM = 10.0           # how far the arm hops back
NUDGE_RECOIL_SPEED_CM_S = 25.0  # the recoil is fast
NUDGE_RECOIL_JERK = 0.0        # brief arm.jerk on the recoil for a startled look (0 = clean)
NUDGE_SETTLE_S = 1.5           # pause after the recoil, "waiting for the cube to stop"

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 110.0           # 0 = closed .. config.MAX_GRIPPER_DEG = full open
GRIP_CLOSED_DEG = 25.0          # tune to the cube width
GRIP_SPEED = 90  #config.GRIPPER_DEFAULT_SPEED
GRIP_SETTLE_S = 0.35           # quiet time after a gripper command: it must LAND and the
                              # jaws start moving. Tunable down to GRIP_MIN_GAP_S, not below.
GRIP_MIN_GAP_S = 0.2          # hard floor -- pymycobot silently drops a gripper command
                              # that is not followed by a short quiet gap (why 0.0 failed).
REACH_TOL_CM = 3.0             # has_reached_* tolerance, per axis
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


def _parabola_points(origin, target, arc_height, rng):
    """(list of PATH_WAYPOINTS (x,y,z), arc_length) along the arc from `origin`
    to `target`: the straight xy chord + a symmetric vertical parabolic lift of
    apex `arc_height` above the chord (== a*x^2 + c with x = (u-0.5)*chord).
    With `rng` the apex height and a sideways bow are jittered by VARIATION."""
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
    pts, _ = _polyline_points(dense, np.linspace(0.0, L, PATH_WAYPOINTS))
    return [tuple(float(v) for v in p) for p in pts], L


def get_path(origin_point, target_point, arc_height, rng=None):
    """Arc from origin to target as PATH_WAYPOINTS (x,y,z) points. `arc_height`
    is this move's apex above the chord -- compute it with _arc_height()."""
    return _parabola_points(origin_point, target_point, arc_height, rng)[0]


def get_durations(origin_point, target_point, arc_height,
                  ease_in_accel=EASE_IN, ease_out_accel=EASE_OUT,
                  cruise=CRUISE_SPEED_CM_S):
    """PATH_WAYPOINTS-1 segment durations (s) for the arc between the two points,
    shaped by the EASE_IN / EASE_OUT dials (0..10, no physical meaning)."""
    _, L = _parabola_points(origin_point, target_point, arc_height, None)

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
    T = L / max(cruise * mean_v, 1e-6)

    ss = np.linspace(0.0, 1.0, PATH_WAYPOINTS)
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


def run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs, d_max):
    """Scripted flinch: approach part-way, recoil, wait, re-approach the moved cube."""
    rx, ry = PICK_ORIENTATION_DEG[:2]
    n = len(pts)
    cut = max(2, int(round(NUDGE_AT_FRACTION * (n - 1))) + 1)
    print(f"  NUDGE cycle {ci}: approaching to {int(NUDGE_AT_FRACTION*100)}% ...")
    if not _send_arc(arm, pts[:cut], durs[:cut - 1], "  nudge approach"):
        return False

    here = np.asarray(current_pos(arm), float)
    travel = here - np.asarray(pts[0], float)
    dirn = travel / (np.linalg.norm(travel) + 1e-9)
    recoil = here - dirn * NUDGE_RECOIL_CM + np.array([0.0, 0.0, NUDGE_RECOIL_CM * 0.5])

    arm.jerk = NUDGE_RECOIL_JERK
    print(f"  RECOIL -> {np.round(recoil, 1).tolist()}  (jerk={arm.jerk})")
    ok = arm.send_coords(x=float(recoil[0]), y=float(recoil[1]), z=float(recoil[2]),
                         rx=rx, ry=ry, rz=_yaw_one((float(recoil[0]), float(recoil[1]))),
                         speed=NUDGE_RECOIL_SPEED_CM_S)
    arm.jerk = 0.0
    if not ok:
        print(f"  recoil REFUSED -- {arm.last_error}")
        return False

    print(f"  waiting {NUDGE_SETTLE_S:.1f}s for the cube to settle ...")
    time.sleep(NUDGE_SETTLE_S)

    new_cube = tuple(float(c + o) for c, o in zip(seg["target"], NUDGE_OFFSET_CM))
    print(f"  cube moved -> re-approaching {tuple(round(v, 1) for v in new_cube)}")
    after = current_pos(arm)
    h2 = _arc_height(_chord_len(after, new_cube), d_max)
    p2 = get_path(after, new_cube, h2, rng)
    d2 = get_durations(after, new_cube, h2, EASE_IN, EASE_OUT)
    if not _send_arc(arm, p2, d2, "  nudge re-approach"):
        return False

    # the following carry cycle must start from where the cube actually is now
    nxt = ci + 1
    if nxt < len(segments) and segments[nxt]["kind"] == "carry":
        segments[nxt]["origin"] = new_cube
        hc = _arc_height(_chord_len(new_cube, segments[nxt]["target"]), d_max)
        paths[nxt] = get_path(new_cube, segments[nxt]["target"], hc, rng)
        all_durs[nxt] = get_durations(new_cube, segments[nxt]["target"], hc, EASE_IN, EASE_OUT)
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
    if 0 <= NUDGE_CYCLE < N_CYCLES:
        # the cube reached on NUDGE_CYCLE also gets grabbed at init + offset
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
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")
    args = ap.parse_args()

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

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(CUBES_INITIAL_POINTS)
    order = list(rng.permutation(n)) if SHUFFLE_ORDER else list(range(n))

    p_home = pose_coords(HOME)                       # mm/deg (Z_RELATIVE_TO_JOINT1 assumed False)
    home_tip = (p_home[0] / 10.0, p_home[1] / 10.0, p_home[2] / 10.0)

    global _AZ_REF                                   # rz = PICK_ORIENTATION_DEG[2] at HOME's azimuth
    _AZ_REF = (home_tip[0], home_tip[1])

    print(f"pick order (cube indices): {[int(k) for k in order]}")
    print(f"orientation lock: {ORIENT_LOCK}")
    if 0 <= NUDGE_CYCLE < N_CYCLES:
        nudged_k = int(order[NUDGE_CYCLE // 2])
        print(f"NUDGE on cycle {NUDGE_CYCLE}: cube #{nudged_k + 1} "
              f"at {CUBES_INITIAL_POINTS[nudged_k]} -- nudge THAT cube as the arm nears it")

    segments = _build_segments(order, home_tip)

    # the widest move sets the shared parabola; every shorter arc lifts less
    d_max = max((_chord_len(s["origin"], s["target"]) for s in segments), default=1.0) or 1.0
    print(f"widest move {d_max:.1f} cm -> apex {MAX_HEIGHT_TRAJECTORY:.1f} cm  "
          f"(shared parabola a = {-MAX_HEIGHT_TRAJECTORY / (d_max / 2.0) ** 2:.4f})")

    # ---- precompute every arc + its durations --------------------------------
    paths, all_durs = [], []
    for seg in segments:
        ei = EASE_IN + float(rng.uniform(-1.0, 1.0)) * EASE_JITTER * VARIATION
        eo = EASE_OUT + float(rng.uniform(-1.0, 1.0)) * EASE_JITTER * VARIATION
        cruise = LEADOUT_SPEED_CM_S if seg["kind"] == "leadout" else CRUISE_SPEED_CM_S
        h = _arc_height(_chord_len(seg["origin"], seg["target"]), d_max)
        paths.append(get_path(seg["origin"], seg["target"], h, rng))
        all_durs.append(get_durations(seg["origin"], seg["target"], h, ei, eo, cruise=cruise))

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
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

            if kind == "reach" and ci == NUDGE_CYCLE:
                if not run_nudge(arm, seg, pts, durs, rng, ci, segments, paths, all_durs, d_max):
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
        arm.close()


if __name__ == "__main__":
    _sys.exit(main())
