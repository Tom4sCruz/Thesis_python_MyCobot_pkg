#!/usr/bin/env python3
"""
MOVEMENT PROFILE: High-human / Low-robot
=======================================

Pick up 4 cubes from one side and place them on the other, with a deliberately
SMOOTH motion: the arm eases into acceleration at the start of every move and
eases out of it at the end. The descent onto a cube is blended into the same
continuous curve as the travel -- there is no "stop above the cube, then go
down" hitch. The only real pauses are AT the grasp and the release, where a
human hand pauses too.

    python3 scripts/Profiles/HighH-LowR.py --mock --yes      # no hardware
    python3 scripts/Profiles/HighH-LowR.py --port /dev/ttyTHS1

HOW THE PROFILE IS PRODUCED
--------------------------
Each move ("leg") is a short poly-line: current pose -> hover point(s) -> cube.
We lay a trapezoidal velocity profile (MAX_SPEED_CM_S, ACCEL_CM_S2, DECEL_CM_S2)
along that poly-line's arc length, optionally rounded toward a jerk-limited
S-curve by PROFILE_SMOOTHING, then hand armik's send_path() a list of waypoints
sampled at equal arc-length together with the per-segment DURATIONS that the
profile implies. send_path() blends velocity through the interior waypoints, so
the hover points are passed through without stopping.

GRIPPER TIMING ("collision box", simple version)
-----------------------------------------------
A pick/place leg ends exactly at the cube, so the cube point IS the trigger:
the gripper fires there. GRIP_TRIGGER_BOX_CM is a per-axis half-extent sanity
gate -- after the leg, the tip must be inside that box around the intended
point or the run aborts (it means the arm did not get where it should).

EVERYTHING you are meant to touch is a CONSTANT near the top of this file.
The cube coordinates and PICK_ORIENTATION_DEG are PLACEHOLDERS -- measure them
on your arm (see the calibration notes) before running on hardware.
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

from armik import Arm, config

# ===========================================================================
# CONSTANTS
# ===========================================================================

# -- run / connection -------------------------------------------------------
HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]   # safe joint-space rest pose
HOME_MOVE_S = 2.5                          # seconds for each homing move
SETTLE_S = 0.3                             # small pause between phases
PREFLIGHT = True                           # plan every cube point before moving

# -- cubes (MEASURE AND REPLACE) ------------------------------------------------
# (x, y, z) in CM, at the GRIPPER TIP, base frame, z from the table.
# Read them by jogging the tip onto each cube's grasp point and calling
# arm.get_coords()[:3]. Source cubes on one side, targets on the other.
SOURCE_CUBES = [
    (15.0,  5.0, 4.0),
    (15.0,  9.0, 4.0),
    (15.0, 13.0, 4.0),
    (15.0, 17.0, 4.0),
]
TARGET_CUBES = [
    (15.0,  -5.0, 4.0),
    (15.0,  -9.0, 4.0),
    (15.0, -13.0, 4.0),
    (15.0, -17.0, 4.0),
]

# -- approach / orientation --------------------------------------------------
APPROACH_HEIGHT_CM = 6.0                   # hover height above a cube

# Gripper orientation (rx, ry, rz, DEGREES) held fixed for EVERY move so the
# gripper stays pointing straight down and a held cube cannot rotate.
# CALIBRATION: jog the arm until the gripper points straight down, then read
# arm.get_coords()[3:]. This value is in the current TOOL frame (see
# config.TOOL_RPY_DEG). The default below is the tip orientation at HOME.
PICK_ORIENTATION_DEG = (180.0, 0.0, -45.0)

# -- gripper ---------------------------------------------------------------------
GRIP_OPEN_DEG = 110.0                      # 0 = closed, config.MAX_GRIPPER_DEG = full open
GRIP_CLOSED_DEG = 25.0                     # tune to the cube width
GRIP_SPEED = config.GRIPPER_DEFAULT_SPEED  # 1..100
GRIP_SETTLE_S = 0.8                        # wait for the jaws (send_gripper is fire-and-forget)
GRIP_TRIGGER_BOX_CM = (2.5, 2.5, 2.5)      # per-axis half-extents of the "trigger box"

# -- smooth velocity profile --------------------------------------------------
MAX_SPEED_CM_S = 12.0                      # cruise / peak tip speed
ACCEL_CM_S2 = 15.0                         # ease-in acceleration at the start of a move
DECEL_CM_S2 = 15.0                         # ease-out deceleration at the end of a move
PROFILE_SMOOTHING = 0.4                    # 0 = trapezoid, 1 = jerk-limited S-curve
PROFILE_WAYPOINTS = 30                     # samples per leg (defines the profile shape)
MIN_SEGMENT_S = 0.02                       # floor on any one segment's duration


# ===========================================================================
# PROFILE MATH
# ===========================================================================

def _min_jerk(u: np.ndarray) -> np.ndarray:
    """Quintic minimum-jerk time scaling: zero velocity AND acceleration at
    both ends (copied from armik.arm._min_jerk)."""
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def _build_profile(L, vmax, accel, decel, smoothing):
    """Return (T, s_of_t) for a trapezoidal (or triangular) velocity profile of
    total arc length L, optionally blended toward a min-jerk S-curve.

    s_of_t is vectorised, monotone increasing, s_of_t(0) == 0, s_of_t(T) == L.
    """
    L = float(L)
    d_a = vmax ** 2 / (2.0 * accel)
    d_d = vmax ** 2 / (2.0 * decel)

    if d_a + d_d <= L:                       # room to reach vmax -> trapezoid
        v_pk = vmax
        d_c = L - d_a - d_d
        t_a, t_c, t_d = v_pk / accel, d_c / v_pk, v_pk / decel
    else:                                    # never reaches vmax -> triangle
        v_pk = math.sqrt(2.0 * L * accel * decel / (accel + decel))
        d_a = v_pk ** 2 / (2.0 * accel)
        d_d = L - d_a
        t_a, t_c, t_d = v_pk / accel, 0.0, v_pk / decel

    T = t_a + t_c + t_d

    def s_of_t(t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, T)
        s = np.empty_like(t)
        m1 = t <= t_a
        s[m1] = 0.5 * accel * t[m1] ** 2
        m2 = (t > t_a) & (t <= t_a + t_c)
        s[m2] = d_a + v_pk * (t[m2] - t_a)
        m3 = t > t_a + t_c
        td = t[m3] - t_a - t_c
        s[m3] = (L - d_d) + v_pk * td - 0.5 * decel * td ** 2
        if smoothing > 0.0:
            mj = L * _min_jerk(t / T if T > 0 else t)
            s = (1.0 - smoothing) * s + smoothing * mj
        return s

    return T, s_of_t


def _polyline_points(verts, s_query):
    """Points at arc-length(s) s_query along the poly-line through verts
    (N x 3). Returns (points, total_length)."""
    verts = np.asarray(verts, dtype=float)
    seg = np.diff(verts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    s = np.clip(np.asarray(s_query, dtype=float), 0.0, total)
    j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seglen) - 1)
    denom = np.where(seglen[j] > 1e-9, seglen[j], 1.0)
    frac = (s - cum[j]) / denom
    return verts[j] + frac[:, None] * seg[j], total


def _profile_waypoints(start_xyz, leg_waypoints):
    """Sample a leg into (xs, ys, zs, durations) following the smooth profile.
    The current pose (start_xyz) is the implicit send_path() start and is NOT
    included in the returned lists."""
    verts = np.vstack([np.asarray(start_xyz, float), np.asarray(leg_waypoints, float)])
    _, L = _polyline_points(verts, [0.0])
    if L < 1e-3:
        return None                                   # already there

    T, s_of_t = _build_profile(
        L, MAX_SPEED_CM_S, ACCEL_CM_S2, DECEL_CM_S2, PROFILE_SMOOTHING
    )

    ss = np.linspace(0.0, L, PROFILE_WAYPOINTS)        # equal arc-length spacing
    # invert s_of_t on a dense grid (S-curve blend is not analytically invertible)
    tg = np.linspace(0.0, T, 4000)
    sg = np.maximum.accumulate(s_of_t(tg))            # guard float non-monotonicity
    ts = np.interp(ss, sg, tg)

    pts, _ = _polyline_points(verts, ss)
    durs = np.maximum(np.diff(ts), MIN_SEGMENT_S)

    return (pts[1:, 0].tolist(), pts[1:, 1].tolist(), pts[1:, 2].tolist(),
            durs.tolist())


# ===========================================================================
# MOTION HELPERS
# ===========================================================================

def _approach(p):
    return (p[0], p[1], p[2] + APPROACH_HEIGHT_CM)


def profiled_move(arm, leg_waypoints, label):
    """One smooth leg through leg_waypoints (list of (x, y, z), cm), holding
    PICK_ORIENTATION_DEG. Returns True on success."""
    rx, ry, rz = PICK_ORIENTATION_DEG
    start = arm.get_coords()[:3]
    sampled = _profile_waypoints(start, leg_waypoints)
    if sampled is None:
        print(f"  {label}: already at target, skipping")
        return True
    xs, ys, zs, durs = sampled

    r = arm.send_path(x=xs, y=ys, z=zs, rx=rx, ry=ry, rz=rz, durations=durs)
    if not r:
        print(f"  {label}: send_path REFUSED -- {arm.last_error}")
        return False

    p = arm.last_plan
    print(f"  {label}: {p.path_length_cm:.1f} cm, {p.duration_s:.2f} s, "
          f"peak {p.peak_joint_dps:.0f} deg/s")
    return True


def within_trigger_box(arm, target_xyz):
    cur = np.asarray(arm.get_coords()[:3], dtype=float)
    off = np.abs(cur - np.asarray(target_xyz, dtype=float))
    return bool(np.all(off <= np.asarray(GRIP_TRIGGER_BOX_CM, dtype=float))), cur


def grip(arm, deg, label):
    print(f"  gripper -> {deg:.0f} deg ({label})")
    if not arm.send_gripper(deg, speed=GRIP_SPEED):
        print(f"  send_gripper REFUSED -- {arm.last_error}")
        return False
    time.sleep(GRIP_SETTLE_S)
    return True


def run_cube(arm, i, prev_target):
    src = SOURCE_CUBES[i]
    tgt = TARGET_CUBES[i]
    print(f"\n=== cube {i + 1}/{len(SOURCE_CUBES)}  "
          f"src={src} -> tgt={tgt} ===")

    # Leg A: reach & grasp. For cubes after the first, lift clear of the
    # previous target first so the sweep does not drag over placed cubes.
    leg_a = []
    if prev_target is not None:
        leg_a.append(_approach(prev_target))
    leg_a += [_approach(src), src]
    if not profiled_move(arm, leg_a, "leg A (reach & grasp)"):
        return False

    ok, cur = within_trigger_box(arm, src)
    if not ok:
        print(f"  !! outside grasp trigger box: at {np.round(cur, 2).tolist()}, "
              f"want {src} +/- {GRIP_TRIGGER_BOX_CM}")
        return False
    if not grip(arm, GRIP_CLOSED_DEG, "close on cube"):
        return False

    # Leg B: lift, carry & place -- one curve through both hover points.
    if not profiled_move(arm, [_approach(src), _approach(tgt), tgt],
                         "leg B (lift, carry & place)"):
        return False

    ok, cur = within_trigger_box(arm, tgt)
    if not ok:
        print(f"  !! outside release trigger box: at {np.round(cur, 2).tolist()}, "
              f"want {tgt} +/- {GRIP_TRIGGER_BOX_CM}")
        return False
    if not grip(arm, GRIP_OPEN_DEG, "release cube"):
        return False

    return True


def preflight(arm):
    """Plan every approach / grasp / release point without moving. Returns True
    if all are reachable at PICK_ORIENTATION_DEG."""
    rx, ry, rz = PICK_ORIENTATION_DEG
    print("\n--- preflight: planning every cube point (no motion) ---")
    checks = []
    for i, (s, t) in enumerate(zip(SOURCE_CUBES, TARGET_CUBES)):
        checks += [(f"src{i+1}", s), (f"src{i+1}^", _approach(s)),
                   (f"tgt{i+1}", t), (f"tgt{i+1}^", _approach(t))]
    bad = 0
    for name, (x, y, z) in checks:
        p = arm.plan_coords(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                            speed=config.DEFAULT_SPEED_CM_S)
        if p.ok:
            print(f"  OK  {name:8s} ({x:5.1f},{y:6.1f},{z:4.1f})  "
                  f"peak {p.peak_joint_dps:.0f} deg/s")
        else:
            print(f"  BAD {name:8s} ({x:5.1f},{y:6.1f},{z:4.1f})  {p.error}")
            bad += 1
    print(f"--- preflight: {len(checks) - bad}/{len(checks)} reachable ---")
    return bad == 0


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")
    args = ap.parse_args()

    if len(SOURCE_CUBES) != len(TARGET_CUBES):
        print("SOURCE_CUBES and TARGET_CUBES must be the same length.")
        return 1

    if not args.mock and not args.yes:
        print("This will move the robot arm and actuate the gripper. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
    try:
        if not arm.conn.is_power_on():
            print("powering on...")
            arm.conn.power_on()
            time.sleep(1.5)

        print("homing...")
        arm.move_joints(HOME, duration=HOME_MOVE_S)
        time.sleep(SETTLE_S)

        print(f"start pose (tip, cm/deg): "
              f"{[round(v, 2) for v in arm.get_coords()]}")

        if PREFLIGHT and not preflight(arm):
            print("\npreflight failed -- fix the cube coordinates or "
                  "PICK_ORIENTATION_DEG. Nothing moved.")
            arm.move_joints(HOME, duration=HOME_MOVE_S)
            return 1

        # open the gripper before the first pick
        if not grip(arm, GRIP_OPEN_DEG, "open before first pick"):
            return 1

        prev_target = None
        for i in range(len(SOURCE_CUBES)):
            if not run_cube(arm, i, prev_target):
                print("\naborting run.")
                arm.move_joints(HOME, duration=HOME_MOVE_S)
                return 1
            prev_target = TARGET_CUBES[i]

        print("\nall cubes placed. homing...")
        arm.move_joints(HOME, duration=HOME_MOVE_S)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    _sys.exit(main())
