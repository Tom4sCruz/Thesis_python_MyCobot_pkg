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
  * each arc ends FINAL_DESCENT_CM above the grasp / place height, and the
    last bit is a short, slow straight descent;
  * the gripper is shut whenever the arm only looks, OPENS GRADUALLY during
    the reach (sent mid-move at GRIP_MOVING_SPEED, as HighH-LowR.py) and
    CLOSES GRADUALLY on the way back to rest after a release;
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
from armik import config
from armik.kinematics import matrix_to_rpy, rpy_to_matrix


# =============================================================================
# CONSTANTS
# =============================================================================

# -- arcs -------------------------------------------------------------------------
TRAJECTORY_DEGREE = 2            # lift above the chord = h * (1 - |x|^degree); 2 = parabola
ARC_HEIGHT_PER_CM = 0.4          # apex height above the chord per cm of horizontal travel ...
ARC_MIN_HEIGHT_CM = 5.0          # ... clipped to this range
ARC_MAX_HEIGHT_CM = 10.0
CRUISE_SPEED_CM_S = 10.0         # peak tip speed along an arc
EASE_IN = 2.0                    # [0,10] start-of-arc acceleration shape (0 abrupt, 10 long S)
EASE_OUT = 2.0                   # [0,10] end-of-arc deceleration shape
PATH_STEP_CM = 0.8               # waypoint spacing along an arc
PATH_MIN_WAYPOINTS = 20
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
GAZE_MAX_JOINT_DPS = 110.0       # ... and a gazing arc that needs a joint faster than this is
                                 # dropped the same way (the wrist whips near the edge of reach)
CARRY_MAX_TILT_DEG = 30.0        # while carrying a cube, the gripper tilts at most this far from
                                 # straight down toward the landing spot: 90 = no limit (but never
                                 # past horizontal, i.e. never looking up), 0 = stays perpendicular
                                 # to the floor (no tilt toward the target at all)

# -- the ends of an arc -------------------------------------------------------------
FINAL_DESCENT_CM = 2.0           # arc ends this far above the grasp / place height; the
                                 # rest is a slow straight descent (0 = arc right onto it)
LIFT_CM = 2.0                    # straight lift off the table after grasping, before the carry
RETREAT_CM = 4.0                 # straight up after releasing, before going to rest
APPROACH_SPEED_CM_S = 3.0        # the final descent / lift / retreat
LINE_SPEED_CM_S = 6.0            # straight-line fallback when an arc can't be planned

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


def arc_points(origin, target, h):
    """(points incl. origin, length) of the arc origin -> target: the straight
    chord + a lift h * (1 - |x|^TRAJECTORY_DEGREE), evenly spaced along it."""
    o, t = np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
    u = np.linspace(0.0, 1.0, 2000)
    shape = 1.0 - np.abs(2.0 * u - 1.0) ** float(TRAJECTORY_DEGREE)
    dense = o[None, :] + (t - o)[None, :] * u[:, None]
    dense[:, 2] += h * shape
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


class HumanRunner(_demo.DemoRunner):
    """Gazing arcs between the cube and the spot; rests looking at the Target Area."""

    REST_WHILE_WAITING = True

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

    def _arc(self, end, gaze_target, level_orient, label, max_tilt=None, grip=None):
        """One arc from the live pose to `end` (x, y, z), looking at gaze_target
        (None = no gaze) and arriving at level_orient. max_tilt: the most the
        gripper may tilt from straight down (None = no limit). grip = (deg,
        delay_s, label): a gripper command sent during the move."""
        tip = self.arm.get_coords()
        start = tuple(float(v) for v in tip[:3])
        if math.dist(start, end) < 0.5:
            return True
        pts, L = arc_points(start, end, arc_height(start, end))
        durs = arc_durations(len(pts), L)
        tail = pts[1:]
        level = [tuple(level_orient)] * len(tail)
        start_rpy = tuple(float(v) for v in tip[3:6])
        xs, ys, zs = ([p[i] for p in tail] for i in range(3))

        def plan(orients):
            rx, ry, rz = orients
            return self.arm.plan_path(x=xs, y=ys, z=zs, rx=rx, ry=ry, rz=rz, durations=durs)

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
                        print(f"[run]   {label}: gaze at {s:.0%} strength")
                    break
                else:
                    why = pl.error
            else:
                print(f"[run]   {label}: gaze arc refused ({why}) -- without the gaze")
        if pl is None or not pl.ok:
            pl = plan(eased)
        if not pl.ok:
            print(f"[run]   {label}: arc refused ({pl.error}) -- straight line instead")
            return _with_grip(self.arm, grip,
                              lambda: self._move(*end, level_orient, LINE_SPEED_CM_S))
        ex = _with_grip(self.arm, grip, lambda: self.arm._execute(pl))
        self.arm.last_plan, self.arm.last_execution = pl, ex
        if not ex.ok:
            self.arm.last_error = ex.error
            return False
        print(f"[run]   {label}: arc {pl.path_length_cm:.1f} cm in {pl.duration_s:.2f} s, "
              f"peak {pl.peak_joint_dps:.0f} deg/s")
        return True

    def _steps(self, t):
        steps = []
        place_z, po = t["place_z"], t["po"]
        if not t["holding"]:
            o, gx, gy = t["o"], t["gx"], t["gy"]
            grasp_z = t["grasp_z"]
            cube = t["cube_center"]
            opening = (RC.GRIP_OPEN_DEG, GRIP_OPEN_DELAY_S, "open")
            steps += [("reach", lambda: self._arc((gx, gy, grasp_z + FINAL_DESCENT_CM), cube, o,
                                                  "reach", grip=opening)
                       and self._grip_check(RC.GRIP_OPEN_DEG)),
                      ("descend", lambda: self._line(gx, gy, grasp_z, o)),
                      ("grasp", self._grasp),
                      ("lift", lambda: self._line(gx, gy, grasp_z + LIFT_CM, o))]
        look = t["dest_center"] if CARRY_GAZE_ENABLED else None
        tilt = CARRY_MAX_TILT_DEG
        if t["dest"][0] == "hand":
            dpx, dpy, dpz = t["dpx"], t["dpy"], t["dpz"]
            # step names match RC.HandWatch.WATCH_STEPS
            steps += [("drop pos", lambda: self._arc((dpx, dpy, dpz), look, po, "carry",
                                                         max_tilt=tilt)),
                      ("wait for hand", self._wait_hand),
                      ("release", self._release),
                      self._end_step()]
        else:
            px, py = t["px"], t["py"]
            steps += [("carry", lambda: self._arc((px, py, place_z + FINAL_DESCENT_CM), look, po,
                                                  "carry", max_tilt=tilt)),
                      ("lower", lambda: self._line(px, py, place_z, po)),
                      ("release", self._release),
                      ("retreat", lambda: self._line(px, py, place_z + RETREAT_CM, po)),
                      self._end_step()]
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


if __name__ == "__main__":
    _sys.exit(_demo.run_demo(HumanRunner, "demo_human", "Humanlike: gazing arcs"))
