#!/usr/bin/env python3
"""
Interactive joint-zero calibration.

Parks the arm straight (every joint -> 0 deg), then walks J1..J6 one at a time:

  * press Enter (blank line)      -> this joint is fine, move to the next one
  * type a number (2, -1.5, 4.2)  -> nudge THIS joint by that many degrees
                                     (relative), then ask again -- repeat until
                                     you press Enter

At the end it prints each joint's cumulative nudge: those are the zero offsets
you dialled in.

    python3 scripts/calibrate_zero.py --mock --yes      # no hardware
    python3 scripts/calibrate_zero.py --port /dev/ttyTHS1
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time

from armik import Arm, ArmError, config


def _wait(read_now, target, tol=1.5, timeout=45.0):
    """Poll read_now() (-> float) until it is within `tol` deg of `target`, or
    `timeout` seconds pass. Returns the last reading."""
    t0 = time.perf_counter()
    cur = read_now()
    while abs(cur - target) > tol and time.perf_counter() - t0 < timeout:
        time.sleep(0.1)
        cur = read_now()
    return cur


def _ask(prompt):
    """input().strip(), but an empty/closed stdin counts as 'press Enter'."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser(description="Interactive joint-zero calibration.")
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="deg/s for every move (default 2 -- slow, for calibration)")
    args = ap.parse_args()
    spd = args.speed

    if not args.mock and not args.yes:
        print("This will move the arm to all-zero, then nudge the joints you pick. "
              "Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
    try:
        if not arm.conn.is_power_on():
            print("powering on...")
            arm.conn.power_on()
            time.sleep(1.5)

        print(f"\nmoving all joints to 0 deg at {spd} deg/s (slow)...")
        arm.conn.send_angles([0.0] * config.DOF, spd)
        for j in range(1, config.DOF + 1):
            _wait(lambda: arm.get_angles()[j - 1], 0.0, timeout=90.0)
        print(f"at zero: {[round(a, 2) for a in arm.get_angles()]}\n")

        offsets = [0.0] * config.DOF
        for j in range(1, config.DOF + 1):
            while True:
                raw = _ask(f"J{j}  [Enter = OK, or degrees to nudge, e.g. 2 or -1.5]: ")
                if raw == "":
                    break
                try:
                    delta = float(raw)
                except ValueError:
                    print("  not a number -- press Enter to move on, or type a value like 2 or -1.5")
                    continue
                offsets[j - 1] += delta
                try:
                    arm.conn.send_angle(j, offsets[j - 1], spd)
                except (ArmError, ValueError) as exc:
                    print(f"  refused: {exc}")
                    offsets[j - 1] -= delta
                    continue
                reached = _wait(lambda: arm.get_angles()[j - 1], offsets[j - 1])
                print(f"  J{j} -> {reached:+.2f} deg   (cumulative nudge {offsets[j - 1]:+.2f})")

        print("\n=== calibration offsets (deg from the arm's reported zero) ===")
        for j, off in enumerate(offsets, 1):
            print(f"  J{j}: {off:+.2f}")
        print(f"final angles: {[round(a, 2) for a in arm.get_angles()]}")
        return 0

    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping the arm.")
        arm.conn.stop()
        return 1
    finally:
        arm.close()


if __name__ == "__main__":
    _sys.exit(main())
