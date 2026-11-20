#!/usr/bin/env python3
"""
DEMO, HUMANLIKE profile: the cube exchange with human-like reaching
===================================================================

The study's exchange (_demo.py: turns, Target Area, keys, --on-hand), moved
the way HighH-LowR.py / HighH-HighR.py move:

  * every reach and carry is one smooth ARC (a parabola above the straight
    chord, TRAJECTORY_DEGREE = 2) with eased start and stop;
  * the gripper LOOKS at where it is going -- the cube while reaching for it,
    the spot where the cube will land while carrying it (CARRY_GAZE_ENABLED)
    -- and levels out to the grasp orientation (straight down, turned to meet
    the cube) over the last GAZE_EASE_OUT_S of the arc. The jaws come in at
    an angle and finish perpendicular, so they don't catch a cube's edge;
    While carrying, the gripper tilts toward the landing spot by at most
    CARRY_MAX_TILT_DEG from straight down (0 = never tilts);
  * each arc runs on, in the SAME move, into a short straight descent of
    FINAL_DESCENT_CM onto the grasp / place height (one fluid motion, the
    ease-out slowing the tip into it); the carry likewise starts with the
    LIFT_CM lift off the table, and the nod is one stream;
  * every move of a transfer is PLANNED AHEAD (PREPLAN_ENABLED), before the
    first one, while the arm stands still -- each from the previous move's
    planned end -- so no planning happens between moves. A stored plan is
    used only if the arm is within PREPLAN_TOL_DEG of its planned start;
    otherwise that move is re-planned from where the arm really is;
  * the gripper is shut whenever the arm only looks, OPENS GRADUALLY during
    the reach (sent mid-move at GRIP_MOVING_SPEED, as HighH-LowR.py) and
    CLOSES GRADUALLY on the way back to rest after a release;
  * after a release the arm RETURNS to the look configuration LOOKING BACK at
    the cube it just left (as HighH-LowR.py's _return_looking_back), the head
    turning to the Target Area from REST_TURN_START_FRAC of the way;
  * after the exchange's last turn it returns to the look configuration and
    NODS (HighH-LowR.py's nod, the NOD_* dials);
  * while the participant is due the arm RESTS in the look configuration
    (IDLE_CONFIG_J123_DEG, as the profiles' LOOK_CONFIG_J123_DEG), its head
    turned toward the Target Area and the gripper nearly shut
    (GRIP_IDLE_DEG). Every pick starts there: (go to rest,) LOOK at the
    cube with the wrist, then reach for it.

    python3 scripts/Vision/demo_human.py --mock                  # real camera, simulated arm
    python3 scripts/Vision/demo_human.py --port /dev/ttyTHS1     # real arm (take_arm.sh first)
    python3 scripts/Vision/demo_human.py --port /dev/ttyTHS1 --on-hand
    ./scripts/laptop_demo.sh human [--on-hand]                   # laptop + Jetson bridge

An arc whose gaze pose can't be planned is retried without the gaze (easing
straight to the grasp orientation); if that fails too, the move is a plain
straight line -- a turn never fails just because of the arc.

Keys and the exchange: see _demo.py. Units: cm and degrees, robot frame.
"""

from __future__ import annotations

import math
import os as _os
import sys as _sys
import threading

# the gaze helpers live with the movement profiles (scripts/Profiles/_gaze.py)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                  "Profiles"))

import numpy as np

import _demo
import reach_cube as RC
import _gaze
from _celebrate import celebration_durations, min_feasible_duration_s
from armik import Plan, config
from armik.kinematics import check_joint_limits, matrix_to_rpy, rpy_to_matrix


# =============================================================================
# CONSTANTS
# =============================================================================

# -- arcs -------------------------------------------------------------------------
TRAJECTORY_DEGREE = 2            # lift above the chord = h * (1 - |x|^degree); 2 = parabola
ARC_HEIGHT_PER_CM = 0.4          # apex height above the chord per cm of horizontal travel ...
ARC_MIN_HEIGHT_CM = 5.0          # ... clipped to this range
ARC_MAX_HEIGHT_CM = 10.0
CRUISE_SPEED_CM_S = 12.0         # peak tip speed along an arc
EASE_IN = 1.0                    # [0,10] start-of-arc acceleration shape (0 abrupt, 10 long S)
EASE_OUT = 3.0                   # [0,10] end-of-arc deceleration shape
PATH_STEP_CM = 0.8               # waypoint spacing along an arc
PATH_MIN_WAYPOINTS = 60
MIN_SEGMENT_S = 0.02

# -- gaze ---------------------------------------------------------------------------
GAZE_ENABLED = True              # False = arcs ease straight to the grasp orientation
CARRY_GAZE_ENABLED = True        # also look at the landing spot while carrying the cube
GAZE_EASE_IN_S = 1.5             # time constant of turning the head onto a new target
GAZE_EASE_OUT_S = 1.5            # the gripper levels out over about this long before arriving
GAZE_STRENGTHS = (1.0, 0.6, 0.3) # tried in turn: 1 = look straight at the target, 0.5 = turn
                                 # halfway there. A full look is often out of reach (e.g. looking
                                 # back toward the base from a cube far out), so a weaker one is
                                 # tried before giving up on the gaze for that arc
GAZE_MAX_JOINT_DPS = 120.0       # ... and a gazing arc that needs a joint faster than this is
                                 # dropped the same way (the wrist whips near the edge of reach)
CARRY_MAX_TILT_DEG = 10.0        # while carrying a cube, the gripper tilts at most this far from
                                 # straight down toward the landing spot: 90 = no limit (but never
                                 # past horizontal, i.e. never looking up), 0 = stays perpendicular
                                 # to the floor (no tilt toward the target at all)

# -- the ends of an arc -------------------------------------------------------------
FINAL_DESCENT_CM = 2.0           # the last this-many cm onto the grasp / place height are
                                 # straight down -- part of the same move as the arc (0 = arc
                                 # right onto it)
LIFT_CM = 2.0                    # straight lift off the table at the start of the carry (same move)
RETREAT_CM = 4.0                 # straight lift at the start of the return after a release --
                                 # part of the same move (0 = none)
APPROACH_SPEED_CM_S = 3.0        # the final descent / lift / retreat
LINE_SPEED_CM_S = 6.0            # straight-line fallback when an arc can't be planned

# -- planning ahead ------------------------------------------------------------------
PREPLAN_ENABLED = True           # plan every move of a transfer before its first move (the arm is
                                 # standing still then), each from the previous move's planned end;
                                 # False = plan each move just before it, from the measured pose
PREPLAN_TOL_DEG = 2.0            # a stored plan is used only if every joint is within this of its
                                 # planned start; otherwise that move is re-planned from where the
                                 # arm really is

# -- gripper while the arm moves (as HighH-LowR.py) --------------------------------
GRIP_MOVING_SPEED = 20           # gripper speed (0-100) for the gradual open / close during a move
GRIP_OPEN_DELAY_S = 0.0          # seconds after a reach STARTS that the gripper begins to open
GRIP_CLOSE_DELAY_S = 0.6         # seconds after the move back to rest STARTS that it begins to close
GRIP_MOVING_REPEATS = 2          # sends per command (a gripper packet is sometimes dropped)
GRIP_MOVING_REPEAT_GAP_S = 0.1

# -- looking at the cube before a reach -------------------------------------------
LOOK_SPEED_DPS = 40.0            # wrist turn onto the cube: largest joint travel / time ...
LOOK_MIN_S = 1.0                 # ... but never quicker than this

# -- rest (waiting for the participant) ------------------------------------------
IDLE_CONFIG_J123_DEG = (0.0, 50.0, -110.0)   # J1 (base), J2, J3 -- as the profiles' look config
IDLE_LOOK_HEIGHT_CM = 0.0        # the head looks at the Target Area centre, this high above the table
IDLE_MIN_TIP_Z_CM = 8.0          # ... and may not put the tip lower than this above the table
GRIP_IDLE_DEG = 10.0             # nearly shut while resting (as the profiles' GRIP_LOOK_DEG)
REST_SPEED_DPS = 40.0            # joint move to the rest pose: largest joint travel / time ...
REST_MIN_S = 2.0                 # ... but never quicker than this

# -- the return after a release, looking back (as HighH-LowR.py) -------------------
RETURN_SPEED_DPS = 40.0          # average speed of the return (largest joint travel / time)
RETURN_EASE_IN = 2.0             # [0,10]
RETURN_EASE_OUT = 4.0            # [0,10]
RETURN_WAYPOINTS = 30            # samples along the return
REST_TURN_START_FRAC = 0.7       # fraction of the return after which the head leaves the cube
                                 # and turns to the Target Area (1.0 = keep looking at the cube)
HEAD_SWING_SPEED_DPS = 50.0      # average speed of a head swing (largest joint travel / time)
HEAD_SWING_EASE_IN = 3.0         # [0,10]
HEAD_SWING_EASE_OUT = 2.0        # [0,10]
HEAD_SWING_WAYPOINTS = 30

# -- the nod after the last turn (as HighH-LowR.py) ---------------------------------
NOD_ENABLED = True
NOD_START_DELAY_S = 0.0          # held at the nod pose before the first stroke (the nod is part of
                                 # the same move as the return, so 0 = it starts on arrival)
NOD_WRIST_J456_DEG = (50.0, 0.0, 0.0)   # wrist pose the nod is centred on (J1..J3 at
                                 # IDLE_CONFIG_J123_DEG)
NOD_SPEED_DPS = 200.0             # AVERAGE J4 speed of each stroke
NOD_EASE_IN = 2.0                # [0,10]
NOD_EASE_OUT = 1.0               # [0,10]
NOD_START_DIRECTION = 1          # +1 = the first nod goes UP, -1 = DOWN
NOD_UP_COUNT = 3                 # up nods ...
NOD_DOWN_COUNT = 2               # ... and down nods, alternating from NOD_START_DIRECTION
NOD_UP_DEG = 15.0                # how far J4 tilts UP from the centre on an up nod
NOD_DOWN_DEG = 20.0              # ... and DOWN on a down nod
NOD_UP_J4_SIGN = 1.0             # +1: increasing J4 tilts the gripper up
NOD_WAYPOINTS_PER_STROKE = 20
NOD_BLEND_START_FRAC = 0.7       # on the last return: fraction of the way after which the head
                                 # leaves the cube for NOD_WRIST_J456_DEG


# =============================================================================
# ARC GEOMETRY (as HighH-HighR.py)
# =============================================================================

def _smootherstep(p):
    p = np.clip(p, 0.0, 1.0)
    return 6 * p ** 5 - 15 * p ** 4 + 10 * p ** 3


def _polyline_points(verts, s_query):
    """Points at arc-length(s) s_query along the poly-line through verts."""
    verts = np.asarray(verts, dtype=float)
    seg = np.diff(verts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    s = np.clip(np.asarray(s_query, dtype=float), 0.0, float(cum[-1]))
    j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, max(len(seglen) - 1, 0))
    denom = np.where(seglen[j] > 1e-9, seglen[j], 1.0)
    return verts[j] + ((s - cum[j]) / denom)[:, None] * seg[j]


def arc_height(origin, target):
    d = math.hypot(target[0] - origin[0], target[1] - origin[1])
    return float(np.clip(ARC_HEIGHT_PER_CM * d, ARC_MIN_HEIGHT_CM, ARC_MAX_HEIGHT_CM))


def arc_points(origin, target, h, drop_to_z=None, rise_cm=0.0):
    """(points incl. origin, length) of the arc origin -> target: the straight
    chord + a lift h * (1 - |x|^TRAJECTORY_DEGREE), evenly spaced along it.
    drop_to_z: the path carries on straight down from target to this height
    (the final descent, in the same move). rise_cm: it starts with a straight
    rise of this much (the lift off the table, in the same move)."""
    o, t = np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
    pre = np.empty((0, 3))
    if rise_cm > 1e-3:
        w = np.linspace(0.0, 1.0, 200)[:-1]
        pre = np.tile(o, (len(w), 1))
        pre[:, 2] += float(rise_cm) * w
        o = o + np.array([0.0, 0.0, float(rise_cm)])
    u = np.linspace(0.0, 1.0, 2000)
    shape = 1.0 - np.abs(2.0 * u - 1.0) ** float(TRAJECTORY_DEGREE)
    dense = o[None, :] + (t - o)[None, :] * u[:, None]
    dense[:, 2] += h * shape
    dense = np.vstack([pre, dense])
    if drop_to_z is not None and t[2] - drop_to_z > 1e-3:
        w = np.linspace(0.0, 1.0, 200)[1:]
        drop = np.tile(t, (len(w), 1))
        drop[:, 2] = t[2] + (float(drop_to_z) - t[2]) * w
        dense = np.vstack([dense, drop])
    L = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    step = max(PATH_STEP_CM, 1.5 * CRUISE_SPEED_CM_S * MIN_SEGMENT_S)
    n = max(int(PATH_MIN_WAYPOINTS), int(math.ceil(L / step)) + 1)
    pts = _polyline_points(dense, np.linspace(0.0, L, n))
    return [tuple(float(v) for v in p) for p in pts], L


def arc_durations(n_points, length):
    """n_points-1 segment durations: cruise at CRUISE_SPEED_CM_S, eased in and
    out per EASE_IN / EASE_OUT (the waypoints are evenly spaced along the arc)."""
    a = float(np.clip(EASE_IN, 0.0, 10.0)) / 10.0
    b = float(np.clip(EASE_OUT, 0.0, 10.0)) / 10.0
    r_in, r_out = 0.05 + 0.45 * a, 0.05 + 0.45 * b
    if r_in + r_out > 1.0:
        k = 1.0 / (r_in + r_out)
        r_in, r_out = r_in * k, r_out * k
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
    T = length / max(CRUISE_SPEED_CM_S * float(s[-1]), 1e-6)
    t_k = np.interp(np.linspace(0.0, 1.0, n_points), s / s[-1], tau) * T
    return np.maximum(np.diff(t_k), MIN_SEGMENT_S).tolist()


def _tilt_deg(R):
    """Angle between the tool axis (3rd column) and straight down."""
    return math.degrees(math.acos(float(np.clip(-R[2, 2], -1.0, 1.0))))


def _limit_tilt(base, orients, max_deg):
    """Per waypoint, pull orients back toward base (both (rx_list, ry_list,
    rz_list); base = the no-gaze orientations) just enough that the gripper
    tilts at most max_deg (capped at 90) from straight down. Where base itself
    tilts more than that, base is used."""
    cap = float(np.clip(max_deg, 0.0, 90.0))
    out = ([], [], [])
    for b, g in zip(zip(*base), zip(*orients)):
        Rb, Rg = rpy_to_matrix(*b), rpy_to_matrix(*g)
        if _tilt_deg(Rg) <= cap + 1e-6:
            R = Rg
        elif _tilt_deg(Rb) >= cap:
            R = Rb
        else:
            lo, hi = 0.0, 1.0                  # tilt(slerp(lo)) <= cap < tilt(slerp(hi))
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if _tilt_deg(_gaze._slerp_matrix(Rb, Rg, mid)) <= cap:
                    lo = mid
                else:
                    hi = mid
            R = _gaze._slerp_matrix(Rb, Rg, lo)
        for lst, v in zip(out, matrix_to_rpy(R)):
            lst.append(float(v))
    return out


def _blend(base, gazed, s):
    """Per waypoint, slerp the no-gaze orientation `base` toward the gazing one
    by s (both are (rx_list, ry_list, rz_list)); s = 1 -> gazed exactly."""
    if s >= 1.0:
        return gazed
    out = ([], [], [])
    for b, g in zip(zip(*base), zip(*gazed)):
        R = _gaze._slerp_matrix(rpy_to_matrix(*b), rpy_to_matrix(*g), s)
        for lst, v in zip(out, matrix_to_rpy(R)):
            lst.append(float(v))
    return out


# =============================================================================
# MOTION
# =============================================================================

class _GripTimer(threading.Thread):
    """One gripper command sent WHILE the arm moves (as HighH-LowR.py): started
    right before a blocking move, it waits delay_s and sends from this thread
    (ArmConnection serialises the writes, so it lands between setpoints).
    finish() -- once the move is over -- sends at once if it hasn't yet."""

    def __init__(self, arm, deg, delay_s, label):
        super().__init__(daemon=True)
        self.arm, self.deg, self.delay_s, self.label = arm, float(deg), float(delay_s), label
        self._now = threading.Event()

    def run(self):
        self._now.wait(max(self.delay_s, 0.0))
        print(f"[run]   gripper -> {self.deg:.0f} deg ({self.label}, while moving)")
        for i in range(max(int(GRIP_MOVING_REPEATS), 1)):
            if i:
                threading.Event().wait(GRIP_MOVING_REPEAT_GAP_S)
            self.arm.send_gripper(self.deg, speed=GRIP_MOVING_SPEED)

    def finish(self):
        self._now.set()
        self.join()


def _with_grip(arm, grip, move):
    """move(), with grip = (deg, delay_s, label) sent during it (None = no grip)."""
    if grip is None:
        return move()
    timer = _GripTimer(arm, *grip)
    timer.start()
    try:
        return move()
    finally:
        timer.finish()


# -- timed joint-space moves (ported from HighH-LowR.py) ------------------------------

def _ease_curve(ease_in, ease_out):
    """(progress_at, time_at): the [0,10] ease dials as a pair of functions on
    [0,1] -- eased progress at a fraction of the move's time, and the inverse."""
    s_grid = np.linspace(0.0, 1.0, 201)
    t_grid = np.concatenate([[0.0], np.cumsum(celebration_durations(ease_in, ease_out, 1.0, 201))])
    t_grid /= t_grid[-1]
    return (lambda x: np.interp(np.clip(x, 0.0, 1.0), t_grid, s_grid),
            lambda p: np.interp(np.clip(p, 0.0, 1.0), s_grid, t_grid))


def _ticks(total_s):
    """One sample time per control tick over total_s, ends included."""
    n = max(1, int(total_s * config.CONTROL_RATE_HZ))
    return np.linspace(0.0, total_s, n + 1)


def _line_to(q_from, q_to, n):
    """n rows evenly spaced from q_from (excluded) to q_to (included)."""
    a, b = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    return a[None, :] + (b - a)[None, :] * np.linspace(0.0, 1.0, int(n) + 1)[1:, None]


def _timed_rows(sample, label):
    """(t, rows) of a joint-space move given as sample(scale) -> (t, rows),
    row 0 its start; stretched (never sped up) to keep every joint within the
    hardware speed limit. (None, reason) if a row breaks a joint limit."""
    t, rows = sample(1.0)
    for r in rows[1:]:
        problems = check_joint_limits(r)
        if problems:
            return None, f"{label}: joint limits: {'; '.join(problems)}"
    min_s = min_feasible_duration_s(rows, np.diff(t) / t[-1])
    if min_s > t[-1] + 1e-6:
        t, rows = sample(min_s / t[-1])
    return (t, rows), None


def _rows_plan(t, rows):
    return Plan(ok=True, q_waypoints=np.asarray(rows, dtype=float),
                timestamps=np.asarray(t, dtype=float), duration_s=float(t[-1]))


def _eased_rows(q0, q_waypoints, speed_dps, ease_in, ease_out, label):
    """(t, rows) of one eased joint-space stroke from q0 along q_waypoints
    (the last row is the goal): (largest joint travel) / speed_dps seconds.
    None if there is nothing to move; raises ValueError on a joint limit."""
    path = np.vstack([np.asarray(q0, dtype=float)[None, :], np.asarray(q_waypoints, dtype=float)])
    travel = float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0)))
    if travel < 0.5:
        return None
    progress_at, _ = _ease_curve(ease_in, ease_out)
    idx = np.arange(len(path), dtype=float)
    base_s = travel / max(float(speed_dps), 1e-6)

    def sample(scale):
        t = _ticks(base_s * scale)
        u = progress_at(t / t[-1]) * (len(path) - 1)
        rows = np.column_stack([np.interp(u, idx, path[:, j]) for j in range(path.shape[1])])
        rows[0], rows[-1] = path[0], path[-1]
        return t, rows

    tr, why = _timed_rows(sample, label)
    if tr is None:
        raise ValueError(why)
    return tr


def _concat_segs(a, b):
    """_Seg a then _Seg b as ONE stream (b starts where a ends: its first row
    is dropped and its times are offset by a's end). a's gripper command."""
    if b.plan is None:
        return a
    ta, ra = np.asarray(a.plan.timestamps, dtype=float), np.asarray(a.plan.q_waypoints, dtype=float)
    tb, rb = np.asarray(b.plan.timestamps, dtype=float), np.asarray(b.plan.q_waypoints, dtype=float)
    t = np.concatenate([ta, ta[-1] + tb[1:]])
    rows = np.vstack([ra, rb[1:]])
    return _Seg(_rows_plan(t, rows), a.grip, "; ".join(m for m in (a.msg, b.msg) if m), a.q0)


class _Seg:
    """One planned move: a Plan for arm._execute, the gripper command to send
    during it (or None) and what to print once it ran."""

    def __init__(self, plan, grip=None, msg="", q0=None):
        self.plan, self.grip, self.msg = plan, grip, msg
        self.q0 = None if q0 is None else np.asarray(q0, dtype=float)

    @property
    def q_start(self):
        return self.q0 if self.plan is None else np.asarray(self.plan.q_waypoints[0], dtype=float)

    @property
    def q_end(self):
        return self.q0 if self.plan is None else np.asarray(self.plan.q_waypoints[-1], dtype=float)


def _nod_key_angles():
    """J4 key angles of the nod, starting and ending at the nod centre."""
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


class HumanRunner(_demo.DemoRunner):
    """Gazing arcs between the cube and the spot; rests looking at the Target Area."""

    REST_WHILE_WAITING = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._fns, self._pre = {}, {}

    def _seeds(self, q0, down, home_q):
        # the rest pose is not gripper-down, but a reach starts right from it
        if self.at_rest:
            return [q0, home_q]
        if not self.holding:                     # a pick goes to rest first (_lead_steps)
            return [np.asarray(self.rest_pose(q0)[0], dtype=float), home_q]
        return [q0, home_q] if down else [home_q]

    def _lead_steps(self, t):
        """A pick starts at rest and looks at the cube first."""
        if t["holding"]:
            return super()._lead_steps(t)
        steps = [] if self.at_rest else [("rest", self._rest_step)]
        cube = t["cube_center"]
        return steps + [("look at cube", lambda: self._look_at(cube))]

    # -- the gripper while moving ---------------------------------------------------

    def _grip_reached(self, v, deg):
        """reach_cube's open / closed check on a 0-100 reading."""
        open_v = round(RC.GRIP_OPEN_DEG / config.MAX_GRIPPER_DEG * 100.0)
        target_v = round(deg / config.MAX_GRIPPER_DEG * 100.0)
        if deg >= RC.GRIP_OPEN_DEG:
            return v >= open_v - RC.GRIP_MOVED_TOL
        close_by = max(1.0, min(2.0 * RC.GRIP_MOVED_TOL, (open_v - target_v) / 2.0))
        return v <= open_v - close_by

    def _grip_check(self, deg):
        """After a move that sent the gripper to deg: fine if the reading says
        so, else the checked (re-sending) gripper command."""
        v = self.arm.get_gripper_value()
        if v is not None and self._grip_reached(v, deg):
            return True
        print(f"[run]   gripper not at {deg:.0f} deg yet (value {v}) -- checked command")
        return self._grip(deg)

    # -- moves --------------------------------------------------------------------

    def _line(self, x, y, z, orient, speed=APPROACH_SPEED_CM_S):
        tip = self.arm.get_coords()
        if math.dist(tip[:3], (x, y, z)) < 0.3:
            return True
        return self._move(x, y, z, orient, speed)

    # -- planning a move from a given joint pose ------------------------------------

    def _plan_arc(self, q0, end, gaze_target, level_orient, label, max_tilt=None, grip=None,
                  drop_to_z=None, rise_cm=0.0):
        """_Seg of one arc from joint pose q0 to `end` (x, y, z), looking at
        gaze_target (None = no gaze) and arriving at level_orient. max_tilt: the
        most the gripper may tilt from straight down (None = no limit). grip =
        (deg, delay_s, label): a gripper command sent during the move.
        rise_cm / drop_to_z: a straight rise at the start / descent at the end
        to this height, in the same move. None if no arc can be planned."""
        return self.plan_from(q0, lambda a: self._arc_on(a, q0, end, gaze_target, level_orient,
                                                         label, max_tilt, grip, drop_to_z, rise_cm))

    def _arc_on(self, a, q0, end, gaze_target, level_orient, label, max_tilt, grip, drop_to_z,
                rise_cm):
        tip = a.get_coords()
        start = tuple(float(v) for v in tip[:3])
        final = end if drop_to_z is None else (end[0], end[1], float(drop_to_z))
        if math.dist(start, final) < 0.5 and rise_cm <= 0.0:
            return _Seg(None, grip, "", q0)
        raised = (start[0], start[1], start[2] + max(rise_cm, 0.0))
        pts, L = arc_points(start, end, arc_height(raised, end), drop_to_z, rise_cm)
        durs = arc_durations(len(pts), L)
        tail = pts[1:]
        level = [tuple(level_orient)] * len(tail)
        start_rpy = tuple(float(v) for v in tip[3:6])
        xs, ys, zs = ([p[i] for p in tail] for i in range(3))

        def plan(orients):
            rx, ry, rz = orients
            return a.plan_path(x=xs, y=ys, z=zs, rx=rx, ry=ry, rz=rz, durations=durs)

        eased = _gaze.ease_to_rpy(level, start_rpy, durs, GAZE_EASE_IN_S, GAZE_EASE_OUT_S)
        pl = None
        if gaze_target is not None and GAZE_ENABLED:
            gazed = _gaze.gaze_then_level_waypoints(tail, gaze_target, level, start_rpy, durs,
                                                    GAZE_EASE_IN_S, GAZE_EASE_OUT_S)
            why = None
            for s in GAZE_STRENGTHS:
                orients = _blend(eased, gazed, s)
                if max_tilt is not None:
                    orients = _limit_tilt(eased, orients, max_tilt)
                pl = plan(orients)
                if pl.ok and pl.peak_joint_dps > GAZE_MAX_JOINT_DPS:
                    why = f"a joint would need {pl.peak_joint_dps:.0f} deg/s"
                    pl = None
                elif pl.ok:
                    if s < 1.0:
                        print(f"[plan]  {label}: gaze at {s:.0%} strength")
                    break
                else:
                    why = pl.error
            else:
                print(f"[plan]  {label}: gaze arc refused ({why}) -- without the gaze")
        if pl is None or not pl.ok:
            pl = plan(eased)
        if not pl.ok:
            print(f"[plan]  {label}: arc refused ({pl.error})")
            return None
        return _Seg(pl, grip, f"{label}: arc {pl.path_length_cm:.1f} cm in {pl.duration_s:.2f} s, "
                              f"peak {pl.peak_joint_dps:.0f} deg/s", q0)

    def _arc_fallback(self, end, level_orient, drop_to_z, rise_cm, grip, label):
        """No arc can be planned from here: straight lines instead (a turn never
        fails just because of the arc)."""
        print(f"[run]   {label}: no arc -- straight lines instead")
        tip = self.arm.get_coords()
        if rise_cm > 0.0 and not self._line(tip[0], tip[1], tip[2] + rise_cm, tuple(tip[3:6])):
            return False
        if not _with_grip(self.arm, grip, lambda: self._move(*end, level_orient, LINE_SPEED_CM_S)):
            return False
        return drop_to_z is None or self._line(end[0], end[1], drop_to_z, level_orient)

    def _plan_return(self, q0, target, next_wrist, start_frac, lift_cm):
        """_Seg of the return after a release, from joint pose q0 (as
        HighH-LowR.py's _return_looking_back): (a straight lift of lift_cm,
        gripper level, then) J1..J3 travel back to IDLE_CONFIG_J123_DEG (J6 to
        HOME's) while the head keeps looking at `target`, the place just left
        (_gaze.wrist_track_angles). Once start_frac of the path is done the
        head turns to next_wrist ([J4, J5, J6], None = the rest pose's) on its
        own clock (HEAD_SWING_SPEED_DPS). One stream; the gripper closes to
        GRIP_IDLE_DEG on the way. None if it breaks a joint limit."""
        q0 = np.asarray(q0, dtype=float)
        tip = self.plan_from(q0, lambda a: a.get_coords())
        head = q0[None, :]
        if lift_cm > 0.0:
            lp = self.plan_from(q0, lambda a: a.plan_coords(tip[0], tip[1], tip[2] + lift_cm,
                                                            *tip[3:6]))
            if lp.ok and lp.q_waypoints is not None and len(lp.q_waypoints) > 1:
                qs = np.asarray(lp.q_waypoints, dtype=float)
                pick = np.unique(np.linspace(0, len(qs) - 1, 9).round().astype(int))
                head = np.vstack([q0[None, :], qs[pick[1:]]])
            else:
                print(f"[plan]  return: lift not plannable ({lp.error}) -- without it")
        q_lift = head[-1]
        q_end = q_lift.copy()
        q_end[:3] = IDLE_CONFIG_J123_DEG
        q_end[5] = RC.HOME[5]
        if next_wrist is None:
            next_wrist = self.rest_pose(q_end)[0][3:6]
        rest = np.vstack([q_lift[None, :], _line_to(q_lift, q_end, RETURN_WAYPOINTS)])
        if GAZE_ENABLED:
            floor_cm = tip[2] - 0.2                       # never dip below the release height
            rest = _gaze.wrist_track_angles(rest, tuple(float(v) for v in target), floor_cm)
        path = np.vstack([head, rest[1:]])                # the lift stays as planned (level)

        idx = np.arange(len(path), dtype=float)
        ret_progress, ret_time = _ease_curve(RETURN_EASE_IN, RETURN_EASE_OUT)
        turn_progress, _ = _ease_curve(HEAD_SWING_EASE_IN, HEAD_SWING_EASE_OUT)
        ret_s = max(float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0)))
                    / max(RETURN_SPEED_DPS, 1e-6), REST_MIN_S)
        nxt = np.asarray(next_wrist, dtype=float)
        frac = float(np.clip(start_frac, 0.0, 1.0))
        trig_s = float(ret_time(frac)) * ret_s
        at_trig = np.array([np.interp(frac * (len(path) - 1), idx, path[:, j]) for j in (3, 4, 5)])
        turn_s = max(float(np.max(np.abs(nxt - at_trig))) / max(HEAD_SWING_SPEED_DPS, 1e-6), 1e-3)
        base_s = max(ret_s, trig_s + turn_s)

        def sample(scale):
            t = _ticks(base_s * scale)
            u = ret_progress(t / (ret_s * scale)) * (len(path) - 1)   # holds the end once arrived
            rows = np.column_stack([np.interp(u, idx, path[:, j]) for j in range(path.shape[1])])
            rows[0] = path[0]
            rows[-1, :3] = path[-1, :3]
            w = turn_progress((t - trig_s * scale) / (turn_s * scale))
            rows[:, 3:6] += w[:, None] * (nxt[None, :] - rows[:, 3:6])
            rows[-1, 3:6] = nxt
            return t, rows

        tr, why = _timed_rows(sample, "return")
        if tr is None:
            print(f"[plan]  return refused -- {why}")
            return None
        return _Seg(_rows_plan(*tr), (GRIP_IDLE_DEG, GRIP_CLOSE_DELAY_S, "close"),
                    f"return: looking back at the cube, head turning away from {frac:.0%} "
                    f"of the way", q0)

    def _plan_nod(self, q0):
        """_Seg of the whole nod as ONE stream: swing the head to
        NOD_WRIST_J456_DEG (skipped if already there), hold NOD_START_DELAY_S,
        then the J4 strokes (NOD_*)."""
        q0 = np.asarray(q0, dtype=float)
        base = np.asarray(list(IDLE_CONFIG_J123_DEG) + list(NOD_WRIST_J456_DEG), dtype=float)
        T, R = [0.0], [q0]

        def add(tr):
            if tr is not None:
                t, rows = tr
                T.extend(float(v) + T[-1] for v in t[1:])
                R.extend(rows[1:])
        try:
            add(_eased_rows(q0, _line_to(q0, base, HEAD_SWING_WAYPOINTS), HEAD_SWING_SPEED_DPS,
                            HEAD_SWING_EASE_IN, HEAD_SWING_EASE_OUT, "nod pose"))
            msg = "at the nod pose"
            if NOD_ENABLED:
                if NOD_START_DELAY_S > 0.0:
                    T.append(T[-1] + NOD_START_DELAY_S)
                    R.append(np.asarray(R[-1], dtype=float))
                keys = _nod_key_angles()
                for a_, b_ in zip(keys[:-1], keys[1:]):
                    q_a, q_b = base.copy(), base.copy()
                    q_a[3], q_b[3] = a_, b_
                    add(_eased_rows(q_a, _line_to(q_a, q_b, NOD_WAYPOINTS_PER_STROKE),
                                    NOD_SPEED_DPS, NOD_EASE_IN, NOD_EASE_OUT, "nod"))
                msg = f"nodded: J4 {' -> '.join(f'{k:.0f}' for k in keys)} deg"
        except ValueError as exc:
            print(f"[plan]  nod refused -- {exc}")
            return None
        if len(R) < 2:
            return _Seg(None, None, "", q0)
        return _Seg(_rows_plan(T, np.asarray(R, dtype=float)), None, msg, q0)

    # -- running a planned move ---------------------------------------------------------

    def _run_seg(self, key, fallback=None):
        """Run the move `key`: its pre-planned _Seg if the arm is within
        PREPLAN_TOL_DEG of the planned start, else planned now from the measured
        pose; fallback() if no plan can be made (None = the step fails)."""
        plan_fn = self._fns[key]
        q = np.asarray(self.arm.get_angles(), dtype=float)
        seg = self._pre.get(key) if PREPLAN_ENABLED else None
        if seg is not None:
            dev = float(np.max(np.abs(q - seg.q_start)))
            if dev > PREPLAN_TOL_DEG:
                print(f"[run]   {key}: the arm is {dev:.1f} deg from the planned start "
                      f"-- re-planning from here")
                seg = None
        if seg is None:
            seg = plan_fn(q)
        if seg is None:
            if fallback is not None:
                return fallback()
            self.arm.last_error = f"{key}: no plan"
            return False
        if seg.plan is None:                              # nothing to move
            if seg.grip is not None:
                _with_grip(self.arm, seg.grip, lambda: True)
            return True
        ex = _with_grip(self.arm, seg.grip, lambda: self.arm._execute(seg.plan))
        self.arm.last_plan, self.arm.last_execution = seg.plan, ex
        if not ex.ok:
            self.arm.last_error = ex.error
            return False
        if seg.msg:
            print(f"[run]   {seg.msg}")
        return True

    def _preplan(self, q_start):
        """Plan every move of the transfer now (the arm is standing still),
        each from the previous one's planned end."""
        self._pre = {}
        if not PREPLAN_ENABLED:
            return True
        q = np.asarray(q_start, dtype=float)
        for key, fn in self._fns.items():
            seg = fn(q)
            if seg is None:
                print(f"[plan]  {key}: no plan ahead -- it is planned when it is due")
                break
            self._pre[key] = seg
            q = seg.q_end
        print(f"[plan]  planned ahead: {', '.join(self._pre) or 'nothing'}")
        return True

    # -- the transfer -------------------------------------------------------------------

    def _lead_steps(self, t):
        """Plan everything; a pick then starts at rest and looks at the cube."""
        self._fns = self._move_fns(t)
        if t["holding"]:
            home = super()._lead_steps(t)
            q_pred = RC.HOME if home else None
            return [("plan ahead", lambda: self._preplan(
                q_pred if q_pred is not None else self.arm.get_angles()))] + home
        resting = self.at_rest                    # (cleared once the run starts)
        cube = t["cube_center"]

        def predicted_start():
            q = self.arm.get_angles()
            if not resting:
                q = self.rest_pose(q)[0]
            q_look, err = self.rest_pose(q, cube)
            return q_look if err is not None else q

        steps = [("plan ahead", lambda: self._preplan(predicted_start()))]
        if not resting:
            steps.append(("rest", self._rest_step))
        return steps + [("look at cube", lambda: self._look_at(cube))]

    def _move_fns(self, t):
        """{move: plan_fn(q) -> _Seg | None}, in the order they run."""
        fns = {}
        place_z, po = t["place_z"], t["po"]
        if not t["holding"]:
            o, gx, gy, grasp_z = t["o"], t["gx"], t["gy"], t["grasp_z"]
            cube = t["cube_center"]
            opening = (RC.GRIP_OPEN_DEG, GRIP_OPEN_DELAY_S, "open")
            fns["reach"] = lambda q: self._plan_arc(q, (gx, gy, grasp_z + FINAL_DESCENT_CM), cube,
                                                    o, "reach", grip=opening, drop_to_z=grasp_z)
        rise = 0.0 if t["holding"] else LIFT_CM
        look = t["dest_center"] if CARRY_GAZE_ENABLED else None
        tilt = CARRY_MAX_TILT_DEG
        if t["dest"][0] == "hand":
            dpx, dpy, dpz = t["dpx"], t["dpy"], t["dpz"]
            fns["drop pos"] = lambda q: self._plan_arc(q, (dpx, dpy, dpz), look, po, "carry",
                                                       max_tilt=tilt, rise_cm=rise)
            lift = 0.0
        else:
            px, py = t["px"], t["py"]
            fns["carry"] = lambda q: self._plan_arc(q, (px, py, place_z + FINAL_DESCENT_CM), look,
                                                    po, "carry", max_tilt=tilt,
                                                    drop_to_z=place_z, rise_cm=rise)
            lift = RETREAT_CM
        target = t["dest_center"]
        if self.is_last_turn:
            def ret_and_nod(q):
                ret = self._plan_return(q, target, NOD_WRIST_J456_DEG, NOD_BLEND_START_FRAC, lift)
                if ret is None or ret.plan is None:
                    return ret
                nod = self._plan_nod(ret.q_end)
                if nod is None:
                    print("[plan]  no nod -- the return runs alone")
                    return ret
                return _concat_segs(ret, nod)
            fns["return"] = ret_and_nod
        else:
            fns["return"] = lambda q: self._plan_return(q, target, None, REST_TURN_START_FRAC,
                                                        lift)
        return fns

    def _steps(self, t):
        steps = []
        place_z, po = t["place_z"], t["po"]
        if not t["holding"]:
            o, gx, gy, grasp_z = t["o"], t["gx"], t["gy"], t["grasp_z"]
            opening = (RC.GRIP_OPEN_DEG, GRIP_OPEN_DELAY_S, "open")
            steps += [("reach", lambda: self._run_seg("reach", lambda: self._arc_fallback(
                          (gx, gy, grasp_z + FINAL_DESCENT_CM), o, grasp_z, 0.0, opening, "reach"))
                       and self._grip_check(RC.GRIP_OPEN_DEG)),
                      ("grasp", self._grasp)]
        rise = 0.0 if t["holding"] else LIFT_CM
        if t["dest"][0] == "hand":
            dpx, dpy, dpz = t["dpx"], t["dpy"], t["dpz"]
            # step names match RC.HandWatch.WATCH_STEPS
            steps += [("drop pos", lambda: self._run_seg("drop pos", lambda: self._arc_fallback(
                          (dpx, dpy, dpz), po, None, rise, None, "carry"))),
                      ("wait for hand", self._wait_hand),
                      ("release", self._release)]
        else:
            px, py = t["px"], t["py"]
            steps += [("carry", lambda: self._run_seg("carry", lambda: self._arc_fallback(
                          (px, py, place_z + FINAL_DESCENT_CM), po, place_z, rise, None, "carry"))),
                      ("release", self._release)]
        steps.append(("return", self._return_step))    # + the nod, on the last turn
        return steps

    # -- rest -----------------------------------------------------------------------

    def rest_pose(self, q_now, target=None):
        """(q, aim_error_deg | None): the look configuration, head on `target`
        (x, y, z; default the Target Area centre), HOME's wrist if no head pose
        can aim there."""
        tz = self.table_z if self.table_z is not None else 0.0
        q = [float(v) for v in q_now]
        q[:3] = [float(v) for v in IDLE_CONFIG_J123_DEG]
        q[5] = float(RC.HOME[5])                 # J6 always the same at rest
        if target is None:
            ax, ay = _demo.area_center()
            target = (ax, ay, tz + IDLE_LOOK_HEIGHT_CM)
        q_look, err = _gaze.wrist_look_angles(q, tuple(float(v) for v in target),
                                              tz + IDLE_MIN_TIP_Z_CM)
        if q_look is not None:
            return q_look, err
        q[3:5] = [float(v) for v in RC.HOME[3:5]]
        return q, None

    def _rest(self):
        q0 = self.arm.get_angles()
        q, err = self.rest_pose(q0)
        travel = float(np.max(np.abs(np.asarray(q) - np.asarray(q0))))
        print(f"[run]   rest: " + (f"looking at the Target Area (aim error {err:.1f} deg)"
                                   if err is not None else "no head pose aims at the area"))
        if travel <= 0.5:
            return self._grip(GRIP_IDLE_DEG)
        closing = (GRIP_IDLE_DEG, GRIP_CLOSE_DELAY_S, "close")
        if not _with_grip(self.arm, closing,
                          lambda: self.arm.move_joints(q, max(REST_MIN_S, travel / REST_SPEED_DPS))):
            return False
        return self._grip_check(GRIP_IDLE_DEG)

    def _look_at(self, cube):
        """From the rest pose, turn the head (wrist only) onto the cube."""
        q0 = self.arm.get_angles()
        q, err = self.rest_pose(q0, cube)
        if err is None:
            print("[run]   look: no head pose aims at the cube -- reaching without the look")
            return True
        travel = float(np.max(np.abs(np.asarray(q) - np.asarray(q0))))
        print(f"[run]   look: at the cube (aim error {err:.1f} deg)")
        if travel <= 0.5:
            return True
        return bool(self.arm.move_joints(q, max(LOOK_MIN_S, travel / LOOK_SPEED_DPS)))

    # -- the return after a release ------------------------------------------------------

    def _return_step(self):
        ok = self._run_seg("return") and self._grip_check(GRIP_IDLE_DEG)
        self.at_rest = bool(ok) and not self.is_last_turn and not self.conn.abort.is_set()
        return ok


if __name__ == "__main__":
    _sys.exit(_demo.run_demo(HumanRunner, "demo_human", "Humanlike: gazing arcs"))
