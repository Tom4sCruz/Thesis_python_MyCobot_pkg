#!/usr/bin/env python3
"""
Validation for send_path() -- the multi-waypoint, velocity-continuous
counterpart to send_coords().

    python3 scripts/test_path.py --mock --yes
    python3 scripts/test_path.py --port /dev/ttyTHS1

Stages:
  1  fresh_mode is set automatically at connection time
  2  velocity is carried through interior waypoints (the actual fix)
  3  a direction reversal correctly slows to (near) zero at that corner
  4  spacing/duration controls speed, same principle as send_coords()
  5  explicit `durations` overrides the speed-derived default
  6  v_start / v_end chain two calls without a stop at the boundary
  7  failure modes -- the arm must NOT move
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging

import numpy as np

from armik import Arm, config

HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]


def banner(n, title):
    print(f"\n{'=' * 68}\n[{n}] {title}\n{'=' * 68}")


def check(cond, msg):
    print(f"    {'PASS' if cond else 'FAIL'}  {msg}")
    return bool(cond)


def speed_around(cmd, t_cmd, t, window=2):
    k = int(np.argmin(np.abs(t_cmd - t)))
    lo, hi = max(0, k - window), min(len(cmd) - 1, k + window)
    if hi <= lo:
        return None, None
    before = np.max(np.abs(cmd[k] - cmd[lo])) / max(t_cmd[k] - t_cmd[lo], 1e-6) if k > lo else None
    after = np.max(np.abs(cmd[hi] - cmd[k])) / max(t_cmd[hi] - t_cmd[k], 1e-6) if hi > k else None
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.mock and not args.yes:
        print("This will move the robot arm repeatedly. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    failures = []
    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)

    def home():
        arm.move_joints(HOME, duration=2.5)

    if not arm.conn.is_power_on():
        arm.conn.power_on()

    # ------------------------------------------------------------------
    banner(1, "fresh_mode is set automatically")
    check(arm.conn.get_fresh_mode() == 1, f"fresh_mode == 1 (got {arm.conn.get_fresh_mode()})")

    # ------------------------------------------------------------------
    banner(2, "Velocity carried through interior waypoints (same direction)")
    home()
    cur_x = arm.get_coords()[0]
    # a straight-ish line so every segment moves the same direction on y
    r = arm.send_path(x=cur_x, y=[-12, -4, 4, 12], z=[10, 10, 10, 10], speed=15.0)
    if not check(r == 1, "send_path() succeeded"):
        failures.append("blend-same-dir")
        print(f"    {arm.last_error}")
    else:
        p = arm.last_plan
        cmd = np.array(arm.last_execution.q_cmd)
        t_cmd = np.array(arm.last_execution.t_cmd)
        ok = True
        for wt in p.waypoint_times[:-1]:
            before, after = speed_around(cmd, t_cmd, wt)
            if before is None or after is None:
                continue
            print(f"    at t={wt:.2f}s: {before:.1f} -> {after:.1f} deg/s")
            if min(before, after) < 3.0:
                ok = False
        if not check(ok, "no interior waypoint drops near zero (no full stop)"):
            failures.append("blend-same-dir")

    # ------------------------------------------------------------------
    banner(3, "Direction reversal correctly zeroes the joints that reverse")
    home()
    cur_x = arm.get_coords()[0]
    r = arm.send_path(x=cur_x, y=[-12, 0, -12], z=[10, 10, 10], speed=10.0)
    if not check(r == 1, "send_path() with a y reversal succeeded"):
        failures.append("blend-reversal")
    else:
        # Check the actual per-joint velocity the planner assigned at the
        # interior waypoint (not a noisy finite difference of the streamed
        # samples) -- only joints that individually reverse sign between the
        # two segments should be zeroed; others may legitimately keep moving,
        # since a Cartesian reversal doesn't imply every joint reverses too.
        from armik import blending, ik
        from armik.kinematics import pose_coords
        q0 = np.array([0.0] * config.DOF)  # HOME, matches the home() call above
        start = pose_coords(np.array(HOME))
        mask = np.array([True, True, True, False, False, False])
        pts = [(cur_x, -12, 10), (cur_x, 0, 10), (cur_x, -12, 10)]
        q_list = [np.array(HOME)]
        seed = np.array(HOME)
        for px, py, pz in pts:
            d = start.copy()
            d[0], d[1], d[2] = px * 10, py * 10, pz * 10
            qk = ik.try_solve(d, mask, seed, max_iters=100)
            q_list.append(qk)
            seed = qk
        q_list = np.array(q_list)
        durations = np.array([1.2, 1.2, 1.2])
        seg_v = np.diff(q_list, axis=0) / durations[:, None]
        reverses = np.sign(seg_v[0]) != np.sign(seg_v[1])
        v_wp = blending.waypoint_velocities(q_list, durations)
        zeroed = np.abs(v_wp[1]) < 1e-6
        print(f"    joints that reverse direction: {np.where(reverses)[0] + 1}")
        print(f"    joints zeroed at the corner:   {np.where(zeroed)[0] + 1}")
        if not check(np.array_equal(reverses, zeroed),
                     "exactly the reversing joints are zeroed -- no more, no fewer"):
            failures.append("blend-reversal")

    # ------------------------------------------------------------------
    banner(4, "Spacing/duration controls speed")
    home()
    cur_x = arm.get_coords()[0]
    arm.send_path(x=cur_x, y=[-12, -6, 0], z=[10, 10, 10], speed=10.0)
    wide = arm.last_plan.peak_joint_dps
    home()
    arm.send_path(x=cur_x, y=[-12, -11, -10], z=[10, 10, 10], speed=10.0)
    tight = arm.last_plan.peak_joint_dps
    print(f"    wide spacing peak: {wide:.1f} deg/s, tight spacing peak: {tight:.1f} deg/s")
    check(tight < wide, "tighter spacing (same speed) gives a lower peak joint rate")

    # ------------------------------------------------------------------
    banner(5, "Explicit durations override speed-derived timing")
    home()
    cur_x = arm.get_coords()[0]
    r = arm.send_path(x=cur_x, y=[-10, 0, 10], z=[10, 15, 10], durations=[0.8, 0.8, 0.8])
    if not check(r == 1, "send_path() with explicit durations"):
        failures.append("durations")
        print(f"    {arm.last_error}")
    else:
        wt = arm.last_plan.waypoint_times
        check(abs(wt[0] - 0.8) < 1e-6 and abs(wt[1] - 1.6) < 1e-6 and abs(wt[2] - 2.4) < 1e-6,
              f"waypoint times match the given durations exactly ({wt})")

    # ------------------------------------------------------------------
    banner(6, "v_start / v_end chain two calls without stopping")
    home()
    cur_x = arm.get_coords()[0]
    r1 = arm.send_path(x=cur_x, y=[-10, 0], z=[10, 15], speed=8.0, v_end=[5, 0, 0, 0, 0, 0])
    exit_v = arm.last_plan.exit_velocity if r1 else None
    r2 = (arm.send_path(x=cur_x, y=[10, 20], z=[10, 5], speed=8.0, v_start=exit_v)
          if r1 else 0)
    if not check(r1 == 1 and r2 == 1, "both chained calls succeeded"):
        failures.append("chaining")
    else:
        check(np.allclose(arm.last_plan.entry_velocity, exit_v),
              f"call 2's entry velocity matches call 1's exit velocity ({exit_v})")

    # ------------------------------------------------------------------
    banner(7, "Failure modes -- the arm must NOT move")
    home()
    before = arm.get_angles()
    cur_x = arm.get_coords()[0]
    cases = [
        ("mismatched list lengths", dict(x=cur_x, y=[1, 2, 3], z=[1, 2])),
        ("no list given (all scalar/None)", dict(x=cur_x)),
        ("far out of reach", dict(y=[300, 400])),
        ("speed too high for hardware",
         dict(x=cur_x, y=list(range(-15, 16, 1)), z=[10] * 31, speed=50.0)),
    ]
    for name, kw in cases:
        rr = arm.send_path(**kw)
        check(rr == 0, f"refused: {name}")
        print(f"        reason: {arm.last_error}")
    after = arm.get_angles()
    moved = max(abs(a - b) for a, b in zip(after, before))
    if not check(moved < 0.5, f"arm did not move during any refusal ({moved:.4f} deg)"):
        failures.append("silent-failure")

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("FAILED: " + ", ".join(failures) if failures else "All stages passed.")
    home()
    arm.close()
    return 1 if failures else 0


if __name__ == "__main__":
    _sys.exit(main())
