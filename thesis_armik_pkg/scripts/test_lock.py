#!/usr/bin/env python3
"""
Validation for lock()/unlock() -- single-joint and multi-joint motion.

    python3 scripts/test_lock.py --mock --yes
    python3 scripts/test_lock.py --port /dev/ttyTHS1

Stages:
  1  lock() holds a joint at exactly 0.000 degrees of travel
  2  true single-joint mode (5 of 6 locked) -- only the free joint moves
  3  genuine multi-joint mode (partial lock) -- coordinated, not sequential
  4  locking makes an otherwise-reachable target correctly unreachable
  5  unlock() / locked_joint() context manager / apply_locks()
  6  send_path() respects locks too
  7  failure modes -- bad joint id, out-of-limit angle, all 6 locked
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
    print(f"\n{'=' * 66}\n[{n}] {title}\n{'=' * 66}")


def check(cond, msg):
    print(f"    {'PASS' if cond else 'FAIL'}  {msg}")
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.mock and not args.yes:
        print("This will move the robot arm. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    failures = []
    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)

    def home():
        arm.unlock()
        arm.move_joints(HOME, duration=2.5)

    # ------------------------------------------------------------------
    banner(1, "lock() holds the joint exactly")
    home()
    arm.lock(1)
    before = arm.get_angles()
    r = arm.send_coords(x=arm.get_coords()[0], z=arm.get_coords()[2] - 2.0, speed=3.0)
    if not check(r == 1, "send_coords() with J1 locked succeeded"):
        failures.append("lock-holds")
        print(f"    {arm.last_error}")
    else:
        after = arm.get_angles()
        check(abs(after[0] - before[0]) < 1e-6, f"J1 moved {abs(after[0]-before[0]):.6f} deg (want 0)")

    # ------------------------------------------------------------------
    banner(2, "True single-joint mode -- only the free joint moves")
    home()
    for j in (1, 3, 4, 5, 6):
        arm.lock(j)
    before = arm.get_angles()
    r = arm.send_coords(z=25.0, speed=3.0)
    if not check(r == 1, "send_coords() with only J2 free succeeded"):
        failures.append("single-joint")
        print(f"    {arm.last_error}")
    else:
        after = arm.get_angles()
        moved = [abs(after[i] - before[i]) for i in range(6)]
        print(f"    per-joint displacement: {[round(m,3) for m in moved]}")
        check(moved[1] > 1.0 and all(m < 1e-6 for i, m in enumerate(moved) if i != 1),
              "only J2 moved")

    # ------------------------------------------------------------------
    banner(3, "Genuine multi-joint mode -- coordinated, not sequential")
    home()
    for j in (4, 5, 6):
        arm.lock(j)
    before = arm.get_angles()
    r = arm.send_coords(x=20, y=5, z=20, speed=3.0)
    if not check(r == 1, "send_coords() with J1-J3 free succeeded"):
        failures.append("multi-joint")
    else:
        after = arm.get_angles()
        moved = [abs(after[i] - before[i]) for i in range(6)]
        print(f"    per-joint displacement: {[round(m,3) for m in moved]}")
        check(all(m < 1e-6 for m in moved[3:]), "J4-J6 stayed exactly put")
        check(any(m > 1.0 for m in moved[:3]), "J1-J3 moved to satisfy the target")

    # ------------------------------------------------------------------
    banner(4, "Locking correctly makes some targets unreachable")
    home()
    start_coords = arm.get_coords()
    r_free = arm.send_coords(y=start_coords[1] + 10.0, speed=3.0)
    home()
    arm.lock(1)   # J1 primarily controls y -- this should now fail
    r_locked = arm.send_coords(y=start_coords[1] + 10.0, speed=3.0)
    check(r_free == 1 and r_locked == 0,
          f"reachable unlocked ({r_free}), correctly unreachable locked ({r_locked})")
    if r_locked == 0:
        print(f"    reason: {arm.last_error}")

    # ------------------------------------------------------------------
    banner(5, "unlock() / locked_joint() / apply_locks()")
    home()
    arm.lock(1)
    arm.lock(2)
    check(arm.locked == {1: 0.0, 2: 0.0}, f"both locked: {arm.locked}")
    arm.unlock(1)
    check(arm.locked == {2: 0.0}, f"unlock(1) leaves only J2: {arm.locked}")
    arm.unlock()
    check(arm.locked == {}, f"unlock() clears everything: {arm.locked}")

    with arm.locked_joint(3, 0.0):
        inside = dict(arm.locked)
    check(inside == {3: 0.0} and arm.locked == {}, "context manager locks, then releases")

    home()
    arm.conn.send_angles([10.0, 10.0, -80.0, 0.0, 0.0, 0.0], 20.0)
    arm.lock(1, angle=0.0)
    r = arm.apply_locks(duration=1.0)
    check(r == 1 and abs(arm.get_angles()[0]) < 0.5,
          f"apply_locks() physically moved J1 to 0.0 (now {arm.get_angles()[0]:.2f})")

    # ------------------------------------------------------------------
    banner(6, "send_path() respects locks too")
    home()
    arm.lock(1)
    before = arm.get_angles()
    r = arm.send_path(z=[15.0, 25.0, 15.0], speed=3.0)
    if not check(r == 1, "send_path() with J1 locked succeeded"):
        failures.append("path-lock")
        print(f"    {arm.last_error}")
    else:
        after = arm.get_angles()
        check(abs(after[0] - before[0]) < 1e-6, f"J1 moved {abs(after[0]-before[0]):.6f} deg across the path")

    # ------------------------------------------------------------------
    banner(7, "Failure modes")
    home()
    try:
        arm.lock(7)
        check(False, "out-of-range joint_id should have raised")
    except ValueError as exc:
        check(True, f"out-of-range joint_id raised: {exc}")

    try:
        arm.lock(1, angle=999.0)
        check(False, "out-of-limit angle should have raised")
    except ValueError as exc:
        check(True, f"out-of-limit angle raised: {exc}")

    home()
    for j in range(1, 7):
        arm.lock(j)
    r = arm.send_coords(x=20.0, speed=3.0)
    check(r == 0, f"all 6 joints locked -> refused ({arm.last_error})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("FAILED: " + ", ".join(failures) if failures else "All stages passed.")
    home()
    arm.close()
    return 1 if failures else 0


if __name__ == "__main__":
    _sys.exit(main())
