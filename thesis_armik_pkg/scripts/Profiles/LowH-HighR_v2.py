#!/usr/bin/env python3
"""
MOVEMENT PROFILE: Low-human / High-robot  (v2 -- reliable grab)
=============================================================

Same idea as LowH-HighR.py -- move cubes from one side of the frame to the
other like a classic industrial robot -- but with a grab that actually works:

  * EVERY move is a straight line, end-effector held straight down with a
    fixed world heading (PICK_ORIENTATION_DEG): per cube, both the reach and
    the carry run the SAME three steps:
        VERTICAL lift to CARRY_HEIGHT_CM  ->  HORIZONTAL straight line to
        directly above the cube / drop point  ->  VERTICAL descent onto it.
    The transit ends exactly above the target, so the gripper descends
    perpendicular to the table onto the cube instead of swiping into it, and
    lifting first means a transit never drags the gripper along the table
    after a release.
  * cubes grabbed in a fixed order (PICK_ORDER, left -> right), same trajectory
    every run, nothing randomised;
  * one cube is NUDGED mid-run (scripted): the lift + horizontal transit happen
    as normal, then the arm STOPS mid-descent onto it (NUDGE_AT_FRACTION of
    the way down), holds for NUDGE_WAIT_S as if watching the cube move, then
    moves on to the NEXT cube instead -- the nudged cube's reach and carry
    are both skipped; it is left behind ("defective"). Fully scripted; the
    arm has no sensors.
  * deliberate jerk is a STUTTER, as in LowH-HighR.py: every lift / transit /
    descent / homing move stops and continues at random points --
    nothing else is perturbed. See the jerk constants; JERK = 0 turns it off.
  * the END-of-run homing is straight lines too: vertical lift ->
    horizontal line to above HOME -> vertical to HOME's height -> a short
    joint move onto HOME's exact angles, stuttering the same way, with a
    PHASE_DELAY_S hold between the legs; the start-of-run / abort homing is
    one coordinated joint move.
  * the arm holds PHASE_DELAY_S at every hand-over between a vertical and a
    horizontal leg: after each lift (before the traverse) and above every
    cube / drop point (before the descent).

    python3 scripts/Profiles/LowH-HighR_v2.py --mock --yes      # no hardware
    python3 scripts/Profiles/LowH-HighR_v2.py --port /dev/ttyTHS1

Every move is PRE-COMPUTED, as in HighH-LowR.py: the whole run is first
rehearsed on a simulated arm, recording each finished plan (straight lines,
IK-solved and speed-capped), gripper command and pause on a "tape"; the tape
is then played back on the real arm with no planning between moves, so the
only gaps left are the deliberate ones. Each straight lift / transit / descent
is IK-solved finely and streamed one setpoint per control tick (arm._execute)
at a CONSTANT tip
speed (DESCENT_ / TRAVERSE_SPEED_CM_S), ramping in / out per MOVE_EASE_IN /
MOVE_EASE_OUT (NUDGE_EASE_OUT for the nudge's interrupted descent); the
start-of-run and abort homing stay live (one coordinated joint move).

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
import math
import time

import numpy as np

from armik import Arm, Plan, config, ik, pose_coords
from armik.kinematics import check_joint_limits, check_workspace_bounds

# ===========================================================================
# CONSTANTS
# ===========================================================================

# -- run / connection -----------------------------------------------------------
HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
SETTLE_S = 0.3
PREFLIGHT = False

# -- cubes (MEASURE AND REPLACE) ----------------------------------------------
# (x, y, z) CM, at the GRIPPER TIP, base frame, z from the table.

Z_CUBE_COORD = -4.0

CUBES_INITIAL_POINTS = [          # a row on the pick side, LISTED LEFT -> RIGHT
                                   # -- matches HighH-LowR.py's layout
    (14.0, 10.0, Z_CUBE_COORD),
    (14.0, 15.5, Z_CUBE_COORD),
    (14.0, 23.0, Z_CUBE_COORD),
]
CUBES_TARGET_POINTS = [           # the drop row; cube picked k-th goes to slot k
    (14.0, -10.0, Z_CUBE_COORD),
    (14.0, -15.5, Z_CUBE_COORD),
    (14.0, -23.0, Z_CUBE_COORD),
]

# Gripper orientation (rx, ry, rz DEG) held for EVERY move. CALIBRATION: jog to
# gripper-straight-down, read arm.get_coords()[3:] (current TOOL frame).
PICK_ORIENTATION_DEG = (180.0, 0.0, -45.0)

# -- homing ------------------------------------------------------------------------
HOME_MOVE_S = 2.5             # minimum duration of a joint-space homing (start / abort, or the
                              # final one if a straight leg is refused)
HOME_RETURN_DPS = 35.0        # deg/s -- a long return gets proportionally MORE time
HOME_SETTLE_MOVE_S = 0.4      # minimum duration of the last joint move onto HOME's exact
                              # angles, after the straight-line homing legs

# -- v2 coordinated grab/place -------------------------------------------------
CARRY_HEIGHT_CM = 8.0         # height the end-effector is lifted to / traverses at -- 10.0+ is
                              # unreachable above the y=+-23 cubes; even 8.0 has the arm nearly
                              # stretched there (see MAX_MOVE_JOINT_DPS)
DESCENT_SPEED_CM_S = 8.0      # cm/s for the coordinated lift and grab/place descent
TRAVERSE_SPEED_CM_S = 15.0    # cm/s for the straight carry-height transit between zones
PHASE_DELAY_S = 0.0           # hold at every hand-over between a vertical and a horizontal leg:
                              # lift -> traverse and traverse -> descent on every cube (incl. the
                              # nudge), and between the legs of the final homing (lift / traverse
                              # / up to HOME); 0 = no pause
MOVE_EASE_IN = 1.0            # [0,10] acceleration at the start of EVERY straight lift / traverse /
MOVE_EASE_OUT = 1.0           # descent / homing leg. 0 = no ramp at all (full speed at once /
                              # dead stop), 10 = long gentle S. In between the tip cruises at a
                              # CONSTANT speed -- DESCENT_ / TRAVERSE_SPEED_CM_S are that speed
MOVE_MAX_ACCEL_CM_S2 = 60.0   # how hard the tip may speed up / slow down around a joint-limited
                              # stretch (see MAX_MOVE_JOINT_DPS) -- it eases into the slow part
                              # instead of braking at once
MOVE_IK_STEP_CM = 0.1         # IK is solved this finely along every straight move -- near the
                              # edge of reach (e.g. above the y=23 cube at CARRY_HEIGHT_CM) the
                              # solver only converges in small steps; the setpoints themselves
                              # are one per control tick, independent of this
MAX_MOVE_JOINT_DPS = 160.0    # no straight lift / traverse / descent may need a joint faster than
                              # this (the real J1..J3 servo limit, see config.py). Where a move
                              # would, the tip slows down LOCALLY -- only over that stretch of
                              # the line (near the edge of reach, e.g. above the y=+-23 cubes,
                              # J3 needs up to ~45 deg per cm); everywhere else it cruises

# -- jerk constants -------------------------------------------------------------
# STOP-and-go only (config.STUTTER_TYPE = 0): JERK just ARMS it, its size does
# nothing. Tune with the two counts below + config.JERK_SUBSTEP_DWELL_S (length
# of each stop). JERK = 0 -> clean motion.
JERK = 5.0

config.JERK_STREAM_STOPS = 4            # stops per lift / transit / descent / homing move

# grab order -- deterministic. Arrange CUBES_INITIAL_POINTS left->right, or set
# explicit indices here.
PICK_ORDER = list(range(len(CUBES_INITIAL_POINTS)))

# -- scripted nudge (non-reactive: this profile has no sensing, so the cube
# -- silently relocates but the arm's motion is completely unaffected) ---------
NUDGE_CYCLE = 2                  # EVEN (reach) cycle index whose cube is nudged;
                                 # -1 = off. 2 = cube #2 (the middle cube), same
                                 # one HighH-LowR.py nudges
NUDGE_OFFSET_CM = (2.0, 0.0, 0.0)   # where the nudged cube ends up (narrative/
                                     # RViz-preview only -- see NUDGE_CUBE_PREVIEW_CM)
NUDGE_AT_FRACTION = 0.65          # fraction of the vertical descent onto the cube
                                 # completed before the arm stops (0..1) -- mirrors
                                 # HighH-LowR.py's NUDGE_AT_FRACTION
NUDGE_EASE_OUT = 0.0             # [0,10] ease-out of the partial descent onto the nudged cube
                                 # (the lift / traverse before it use MOVE_EASE_*) -- 0 = it
                                 # stops dead at NUDGE_AT_FRACTION, as if startled
NUDGE_WAIT_S = 2.5               # how long the arm holds there before giving up on
                                 # this cube and moving on to the next one

# where the nudged cube visually ends up -- always fed to RvizBridge as the
# yellow preview cube, independent of whether NUDGE_CYCLE is enabled this run
NUDGE_CUBE_PREVIEW_CM = tuple(
    float(c + o) for c, o in zip(CUBES_INITIAL_POINTS[1], NUDGE_OFFSET_CM)
)

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 120.0          # 0 = closed .. config.MAX_GRIPPER_DEG = full open
GRIP_CLOSED_DEG = 65.0         # tune to the cube width
GRIP_SPEED = 90  #config.GRIPPER_DEFAULT_SPEED
GRIP_SETTLE_S = 0.35          # quiet time after a gripper command: it must LAND and the jaws
                             # start moving. Tunable down to GRIP_MIN_GAP_S, not below.
GRIP_MIN_GAP_S = 1.0         # hard floor -- pymycobot drops a gripper command with no quiet gap
REACH_TOL_CM = 3.0            # has_reached_* tolerance, per axis


# ===========================================================================
# GEOMETRY HELPERS
# ===========================================================================

def _smootherstep(p):
    """Min-jerk progress 0..1 -- the same profile Arm.move_joints uses."""
    p = np.clip(p, 0.0, 1.0)
    return 10 * p ** 3 - 15 * p ** 4 + 6 * p ** 5


def _ease_speed(ease_in, ease_out):
    """The [0,10] ease dials (same as HighH-LowR.py's get_durations) as the
    tip's NORMALISED speed (1 = cruise) against the fraction of the PATH
    covered. Flat (= cruise) in the middle; a dial of exactly 0 means NO ramp
    on that side -- full speed from the first tick / right up to the last one,
    i.e. a dead stop."""
    a = float(np.clip(ease_in, 0.0, 10.0)) / 10.0
    b = float(np.clip(ease_out, 0.0, 10.0)) / 10.0
    r_in = 0.0 if ease_in <= 0.0 else 0.05 + 0.45 * a
    r_out = 0.0 if ease_out <= 0.0 else 0.05 + 0.45 * b
    if r_in + r_out > 1.0:
        k = 1.0 / (r_in + r_out)
        r_in *= k
        r_out *= k

    tau = np.linspace(0.0, 1.0, 2001)
    v = np.ones_like(tau)
    if r_in > 0.0:
        m_in = tau < r_in
        p = tau[m_in] / r_in
        v[m_in] = (1.0 - a) * p + a * _smootherstep(p)
    if r_out > 0.0:
        m_out = tau > 1.0 - r_out
        q = (1.0 - tau[m_out]) / r_out
        v[m_out] = (1.0 - b) * q + b * _smootherstep(q)
    v = np.maximum(v, 1e-6)

    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tau))])
    s_norm = s / s[-1]
    return lambda f: np.interp(np.clip(f, 0.0, 1.0), s_norm, v)


# ===========================================================================
# STATE CHECKS
# ===========================================================================

def current_pos(arm):
    return tuple(float(v) for v in arm.get_coords()[:3])


def _within(xyz, centre, half_extents):
    d = np.abs(np.asarray(xyz, float) - np.asarray(centre, float))
    return bool(np.all(d <= np.asarray(half_extents, float)))


# ===========================================================================
# THE TAPE -- rehearse the whole run first, then play it back
# ===========================================================================
# As in HighH-LowR.py: planning a move (IK along every straight line) takes real
# time on the Jetson, and doing it between moves left the arm standing still.
# So the choreography runs TWICE:
#   1. REHEARSAL on a second, simulated Arm(mock=True): nothing is streamed and
#      nothing sleeps. Every action -- a finished Plan, a gripper command, a
#      deliberate pause, a log line -- is appended to a list, the tape, and the
#      simulated arm is put at the motion's end pose so the next step plans
#      from the right place.
#   2. PLAYBACK on the real arm: the tape is executed in order. No kinematics
#      and no planning happen between motions, so the only gaps left are the
#      deliberate ones (gripper waits, the nudge wait, the stutter-stops).
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


def _do_plan(arm, plan):
    """Execute a finished Plan -- or, in rehearsal, record it and move the
    simulated arm to its end pose. Returns bool."""
    if _TAPE is not None:
        _TAPE.append(("plan", plan, arm.jerk))
        arm.conn.raw.set_angles_directly([float(v) for v in plan.q_waypoints[-1]])
        return True
    ex = arm._execute(plan)
    if not ex.ok:
        print(f"  execution ABORTED -- {ex.error}")
    return bool(ex.ok)


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
            _, plan, jerk = step
            arm.jerk = jerk
            ex = arm._execute(plan)
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


# ===========================================================================
# MOTION
# ===========================================================================

def go_home(arm):
    """Homing -- ONE coordinated joint move to HOME, the duration scaled to the
    joint distance (HOME_MOVE_S minimum, HOME_RETURN_DPS) so a long return is
    not rushed; streamed, so it stutters like everything else. LIVE ONLY (the
    start-of-run and the abort homing -- the end-of-run one is on the tape).
    Returns bool."""
    dq = max(abs(a - b) for a, b in zip(arm.get_angles(), HOME))
    dur = max(HOME_MOVE_S, dq / HOME_RETURN_DPS)
    if dur > HOME_MOVE_S + 0.05:
        print(f"  homing over {dur:.1f}s (joint travel {dq:.0f} deg)")
    if not arm.move_joints(HOME, duration=dur):
        print(f"  homing REFUSED -- {arm.last_error}")
        return False
    return True


def _home_tip():
    """HOME's tip position (x, y, z) in cm -- gripper straight down there, the
    same orientation as PICK_ORIENTATION_DEG."""
    p = pose_coords(HOME)                        # mm/deg (Z_RELATIVE_TO_JOINT1 assumed False)
    return (p[0] / 10.0, p[1] / 10.0, p[2] / 10.0)


def _joint_move(arm, q_target, duration, label):
    """Straight joint-space move over `duration` s, min-jerk -- built exactly
    like Arm.move_joints, but handed to _do_plan so it can go on the tape."""
    q0 = np.asarray(arm.get_angles(), dtype=float)
    qT = np.asarray(q_target, dtype=float)
    problems = check_joint_limits(qT)
    if problems:
        _say(f"  {label} REFUSED -- target violates joint limits: {'; '.join(problems)}")
        return False
    n = max(2, int(round(duration * config.CONTROL_RATE_HZ)) + 1)
    s = _smootherstep(np.linspace(0.0, 1.0, n))
    wps = q0[None, :] + (qT - q0)[None, :] * s[:, None]
    return _do_plan(arm, Plan(ok=True, q_waypoints=wps, duration_s=float(duration)))


def go_home_coordinated(arm):
    """End-of-run homing in straight lines, like the cube moves: vertical lift
    to CARRY_HEIGHT_CM -> horizontal line to above HOME's tip -> vertical move
    to HOME's tip height -> a short joint move that lands exactly on HOME's
    angles. Streamed, so it stutters like the cube moves when jerk is armed
    (config.JERK_STREAM_STOPS). If any straight leg is refused it falls back to
    one coordinated joint move. Returns bool."""
    hx, hy, hz = _home_tip()
    here = current_pos(arm)
    straight = (_coord_move(arm, here[0], here[1], CARRY_HEIGHT_CM, "home: lift")
                and _phase_pause()
                and _traverse(arm, (hx, hy, hz), "home: traverse")
                and _phase_pause()
                and _coord_move(arm, hx, hy, hz, "home: to HOME height"))
    if not straight:
        _say("  straight-line homing refused -- finishing with one joint move")
    dq = max(abs(a - b) for a, b in zip(arm.get_angles(), HOME))
    if straight and dq <= config.SINGLE_JOINT_TOL_DEG:
        return True
    # after the straight legs only an IK residual (~1 deg) is left: settle it
    # quickly instead of over the full HOME_MOVE_S
    dur = max(HOME_SETTLE_MOVE_S if straight else HOME_MOVE_S, dq / HOME_RETURN_DPS)
    if dur > HOME_MOVE_S + 0.05:
        _say(f"  homing over {dur:.1f}s (joint travel {dq:.0f} deg)")
    return _joint_move(arm, HOME, dur, "homing")


def _solve_line(q0, start, goal):
    """Joint angles along the straight line start -> goal (cm), gripper at
    PICK_ORIENTATION_DEG, IK-solved every MOVE_IK_STEP_CM and seeded from the
    previous solution (one IK branch throughout). Returns (fractions, q_rows)
    -- q_rows[0] is q0 -- or (None, reason)."""
    L = float(np.linalg.norm(goal - start))
    m = max(2, int(math.ceil(L / MOVE_IK_STEP_CM)) + 1)
    fr = np.linspace(0.0, 1.0, m)
    mask = np.ones(6, dtype=bool)
    rows, seed = [np.asarray(q0, dtype=float)], np.asarray(q0, dtype=float)
    for f in fr[1:]:
        p = start + f * (goal - start)
        why = check_workspace_bounds(p * 10.0)
        if why:
            return None, f"outside the workspace at {f * 100:.0f}% of the line ({why})"
        desired = np.array([p[0] * 10.0, p[1] * 10.0, p[2] * 10.0, *PICK_ORIENTATION_DEG])
        q = ik.try_solve(desired, mask, seed)
        if q is None:
            return None, f"unreachable along the straight line at {f * 100:.0f}%"
        problems = check_joint_limits(q)
        if problems:
            return None, f"joint limits at {f * 100:.0f}% of the line: {'; '.join(problems)}"
        rows.append(np.asarray(q, dtype=float))
        seed = rows[-1]
    return fr, np.vstack(rows)


def _coord_move(arm, x, y, z, label, speed=None, ease_out=None):
    """One COORDINATED (multi-joint) straight Cartesian move from the tip's
    current position to (x, y, z), end-effector held at PICK_ORIENTATION_DEG,
    handed to _do_plan (recorded in the rehearsal, streamed at playback).

    The tip CRUISES at `speed` cm/s along the line, ramping in / out per
    MOVE_EASE_IN / MOVE_EASE_OUT (ease_out overrides the latter for this move
    -- the nudge's partial descent). The line is IK-solved finely
    (_solve_line), then sampled ONE SETPOINT PER CONTROL TICK along the eased
    progress. Wherever that would need a joint faster than MAX_MOVE_JOINT_DPS
    (near the workspace edge a cm of tip travel costs many degrees) the tip
    slows down over THAT stretch only, easing in and out of it at up to
    MOVE_MAX_ACCEL_CM_S2 -- same straight path, cruise everywhere else."""
    e_out = MOVE_EASE_OUT if ease_out is None else ease_out
    start = np.asarray(current_pos(arm), dtype=float)
    goal = np.array([float(x), float(y), float(z)])
    L = float(np.linalg.norm(goal - start))
    if L < 0.05:
        return True
    fr, rows = _solve_line(arm.get_angles(), start, goal)
    if fr is None:
        _say(f"  {label}: plan REFUSED -- {rows}")
        return False

    cruise = float(speed or DESCENT_SPEED_CM_S)
    # requested speed along the path: constant cruise with the ease ramps ...
    v_req = cruise * _ease_speed(MOVE_EASE_IN, e_out)(fr)
    # ... capped, point by point, at what keeps every joint under the limit
    ds = np.diff(fr) * L
    dq_ds = np.max(np.abs(np.diff(rows, axis=0)), axis=1) / np.maximum(ds, 1e-9)   # deg/cm
    lim_seg = 0.97 * MAX_MOVE_JOINT_DPS / np.maximum(dq_ds, 1e-9)
    v_lim = np.minimum(np.concatenate([lim_seg[:1], lim_seg]), np.concatenate([lim_seg, lim_seg[-1:]]))
    for k in range(1, len(v_lim)):                     # ease into / out of a slow stretch
        v_lim[k] = min(v_lim[k], math.sqrt(v_lim[k - 1] ** 2 + 2 * MOVE_MAX_ACCEL_CM_S2 * ds[k - 1]))
    for k in range(len(v_lim) - 2, -1, -1):
        v_lim[k] = min(v_lim[k], math.sqrt(v_lim[k + 1] ** 2 + 2 * MOVE_MAX_ACCEL_CM_S2 * ds[k]))
    v = np.maximum(np.minimum(v_req, v_lim), 1e-3)
    limited = v_lim < v_req - 1e-6

    t_at = np.concatenate([[0.0], np.cumsum(2.0 * ds / (v[1:] + v[:-1]))])   # time at each IK row
    T = float(t_at[-1])
    for _ in range(4):              # rounding at the ticks -- stretch slightly if still over
        n = max(1, int(math.ceil(T * config.CONTROL_RATE_HZ)))
        t = np.linspace(0.0, T, n + 1)
        u = np.interp(t * (t_at[-1] / T), t_at, fr)
        q = np.column_stack([np.interp(u, fr, rows[:, j]) for j in range(rows.shape[1])])
        q[0], q[-1] = rows[0], rows[-1]
        peak = float(np.max(np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]))
        if peak <= MAX_MOVE_JOINT_DPS + 1e-6:
            break
        T *= peak / (0.99 * MAX_MOVE_JOINT_DPS)
    pl = Plan(ok=True, q_waypoints=q, timestamps=t, duration_s=float(T),
              path_length_cm=L, peak_joint_dps=peak)
    if not _do_plan(arm, pl):
        return False
    slow = (f", slowed near the arm's reach limit over {limited.mean() * 100:.0f}% of the path "
            f"(min {v[limited].min():.1f} cm/s)" if limited.any() else "")
    _say(f"  {label}: {L:.1f} cm, {T:.2f} s, cruise {cruise:.1f} cm/s{slow}, "
         f"peak {peak:.0f} deg/s")
    return True


def _phase_pause():
    """The PHASE_DELAY_S hold between two legs (recorded on the tape).
    Returns True so it can sit in an `and` chain of moves."""
    if PHASE_DELAY_S > 0.0:
        _pause(PHASE_DELAY_S)
    return True


def _traverse(arm, target_xyz, label):
    """Carry-height transit: ONE straight Cartesian line at CARRY_HEIGHT_CM from
    where the tip is to DIRECTLY ABOVE target_xyz, end-effector held at
    PICK_ORIENTATION_DEG (world-fixed, so the gripper heading never changes).
    Ending exactly above the target is what makes the following descent
    vertical."""
    return _coord_move(arm, target_xyz[0], target_xyz[1], CARRY_HEIGHT_CM, label,
                       speed=TRAVERSE_SPEED_CM_S)


def _lift_traverse_descend(arm, target_xyz, label):
    """v2 reach / carry, all straight lines: vertical lift to CARRY_HEIGHT_CM ->
    horizontal line to above target_xyz -> vertical descent onto it,
    end-effector held straight down throughout."""
    here = current_pos(arm)
    if not _coord_move(arm, here[0], here[1], CARRY_HEIGHT_CM, f"{label}: lift"):
        return False
    _phase_pause()                          # hold at the top before the traverse
    if not _traverse(arm, target_xyz, f"{label}: traverse"):
        return False
    _phase_pause()                          # hold above it before descending
    return _coord_move(arm, target_xyz[0], target_xyz[1], target_xyz[2],
                       f"{label}: descend")


def _fire_gripper(arm, deg):
    """Send the gripper command twice with a tiny gap -- pymycobot drops a
    gripper packet that is not followed by a short quiet window."""
    ok = arm.send_gripper(deg, speed=GRIP_SPEED)
    time.sleep(0.06)
    arm.send_gripper(deg, speed=GRIP_SPEED)
    return bool(ok)


def _grip(arm, deg, label):
    """Gripper command with the arm standing still, then the settle wait
    (never less than GRIP_MIN_GAP_S). Recorded in the rehearsal."""
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


def _lift_traverse_descend_nudge(arm, target_xyz, label):
    """Scripted nudge: lift + traverse as normal, then stop the vertical
    descent at NUDGE_AT_FRACTION of the way down, wait NUDGE_WAIT_S (the
    cube supposedly moves during this pause), then leave it -- the
    gripper never closes and this cube's carry is skipped."""
    here = current_pos(arm)
    if not _coord_move(arm, here[0], here[1], CARRY_HEIGHT_CM, f"{label}: lift"):
        return False
    _phase_pause()                          # hold at the top before the traverse
    if not _traverse(arm, target_xyz, f"{label}: traverse"):
        return False
    _phase_pause()                          # hold above it before descending
    start = current_pos(arm)
    mid = (float(target_xyz[0]), float(target_xyz[1]),     # straight down: x, y already there
           start[2] + NUDGE_AT_FRACTION * (float(target_xyz[2]) - start[2]))
    _say(f"  {label}: NUDGE -- stopping the descent {int(NUDGE_AT_FRACTION * 100)}% "
         f"of the way down (z {start[2]:.1f} -> {mid[2]:.1f} of {target_xyz[2]:.1f} cm)")
    if not _coord_move(arm, mid[0], mid[1], mid[2], f"{label}: descend (partial)",
                       ease_out=NUDGE_EASE_OUT):
        return False
    _say(f"  cube moved -- waiting {NUDGE_WAIT_S:.1f}s, then leaving it behind")
    _pause(NUDGE_WAIT_S)
    return True


# ===========================================================================
# PREFLIGHT
# ===========================================================================

def preflight(arm, segments):
    rx, ry, rz = PICK_ORIENTATION_DEG
    print("\n--- preflight: planning every cube point (no motion) ---")
    checks = []
    for i, (s, t) in enumerate(zip(CUBES_INITIAL_POINTS, CUBES_TARGET_POINTS)):
        checks += [(f"init{i+1}", s), (f"init{i+1}^", (s[0], s[1], CARRY_HEIGHT_CM)),
                   (f"tgt{i+1}", t), (f"tgt{i+1}^", (t[0], t[1], CARRY_HEIGHT_CM))]
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


def _choreography(arm, segments):
    """The whole run -- every cube cycle (incl. the nudge) and the final
    straight-line homing -- written against whatever `arm` it is given. main()
    runs it once on a simulated arm with _TAPE recording (the rehearsal) and
    then plays the tape back on the real one. Returns False if anything was
    refused."""
    if not _grip(arm, GRIP_OPEN_DEG, "open before first pick"):
        return False

    skip_k = None
    for ci, seg in enumerate(segments):
        kind = seg["kind"]

        if skip_k is not None and seg["k"] == skip_k:
            _say(f"\n=== cycle {ci}/{len(segments) - 1}  {kind}  cube #{seg['k'] + 1}  "
                 f"-- SKIPPED (left behind after nudge) ===")
            if kind == "carry":
                skip_k = None
            continue

        grip_deg = GRIP_CLOSED_DEG if kind == "reach" else GRIP_OPEN_DEG
        label = "reach & grasp" if kind == "reach" else "carry & place"
        _say(f"\n=== cycle {ci}/{len(segments) - 1}  {label}  cube #{seg['k'] + 1}  "
             f"-> {tuple(round(v, 1) for v in seg['target'])} ===")

        if kind == "reach" and ci == NUDGE_CYCLE:
            if not _lift_traverse_descend_nudge(arm, seg["target"], label):
                return False
            skip_k = seg["k"]
            continue

        if not _lift_traverse_descend(arm, seg["target"], label):
            return False
        if not _grip(arm, grip_deg, "close on cube" if kind == "reach" else "release cube"):
            return False
        # live check at playback: did the tip really get there?
        if _TAPE is not None:
            _TAPE.append(("check", "cube" if kind == "reach" else "target",
                          tuple(seg["target"])))

    _say("\ndone. homing...")
    return go_home_coordinated(arm)


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

    order = list(PICK_ORDER)
    home_tip = _home_tip()

    print(f"pick order (cube indices): {order}")
    segments = _build_segments(order, home_tip)
    if 0 <= NUDGE_CYCLE < len(segments):
        nk = segments[NUDGE_CYCLE]["k"]
        print(f"NUDGE on cycle {NUDGE_CYCLE}: cube #{nk + 1} at {CUBES_INITIAL_POINTS[nk]} "
              f"-- the descent onto it stops at {int(NUDGE_AT_FRACTION * 100)}%, waits "
              f"{NUDGE_WAIT_S:.1f}s, then moves on to the next cube; this one is left behind")

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
    # Jerky motion: v2's cube moves are coordinated / streamed, so with
    # config.JERK_STREAM_STOPS set (see the jerk constants) each lift / transit /
    # descent STOPS and continues at random points -- no tremor; homing stutters
    # the same way (per joint, or across the final coordinated move). The stops
    # are drawn at PLAYBACK (inside arm._execute), so they differ every run.
    # Seed via arm.jerk_seed or config.JERK_SEED.
    arm.jerk = JERK

    bridge = None
    if args.rviz:
        from _rviz_bridge import RvizBridge
        # polyline the EE actually traces: from HOME, then per segment
        # lift straight up to CARRY_HEIGHT_CM -> traverse at that height ->
        # descend onto the target (mirrors _lift_traverse_descend), then the
        # straight-line homing (mirrors go_home_coordinated)
        flat_path = [tuple(map(float, home_tip))]
        for s in segments:
            ox, oy, _oz = s["origin"]
            tx, ty, tz = s["target"]
            flat_path += [(float(ox), float(oy), float(CARRY_HEIGHT_CM)),
                          (float(tx), float(ty), float(CARRY_HEIGHT_CM)),
                          (float(tx), float(ty), float(tz))]
        lx, ly, _lz = flat_path[-1]
        flat_path += [(lx, ly, float(CARRY_HEIGHT_CM)),
                      (float(home_tip[0]), float(home_tip[1]), float(CARRY_HEIGHT_CM)),
                      tuple(map(float, home_tip))]
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

        if PREFLIGHT and not preflight(arm, segments):
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
        sim.jerk = JERK
        _TAPE = []
        try:
            ok = _choreography(sim, segments)
        finally:
            tape, _TAPE = _TAPE, None
            sim.close()
        if not ok:
            for step in tape:                      # show how far the rehearsal got
                if step[0] == "say":
                    print(step[1])
            print("\nrehearsal failed (see above) -- the arm has not moved beyond homing.")
            return 1
        n_moves = sum(1 for step in tape if step[0] == "plan")
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
