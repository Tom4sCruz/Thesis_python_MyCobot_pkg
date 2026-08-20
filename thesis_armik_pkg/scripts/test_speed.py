#!/usr/bin/env python3
"""
Move back and forth along the y-axis (fixed x and z), increasing speed by 1
cm/s every iteration. Every 10th speed value, ask before continuing. Ctrl+C
stops the arm immediately.

    python3 scripts/test_speed.py --mock --y-left -10 --y-right 10
    python3 scripts/test_speed.py --port /dev/ttyTHS1 --y-left -10 --y-right 10 --x 17 --z 15

If --x/--z are omitted, the arm's CURRENT x/z (read at startup) are held fixed
and only y is swept.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time

from armik import Arm, config

HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the initial safety prompt")
    ap.add_argument("--x", type=float, default=None, help="cm; default = current x")
    ap.add_argument("--z", type=float, default=None, help="cm; default = current z")
    ap.add_argument("--y-left", type=float, required=True, help="cm")
    ap.add_argument("--y-right", type=float, required=True, help="cm")
    ap.add_argument("--confirm-every", type=int, default=5,
                     help="ask to continue every N cm/s of speed")
    args = ap.parse_args()

    if not args.mock and not args.yes:
        print("This will move the robot arm repeatedly, back and forth. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            print("aborted")
            return 1

    with Arm(port=args.port, baudrate=args.baud, mock=args.mock) as arm:
        if not arm.conn.is_power_on():
            arm.conn.power_on()
            time.sleep(1.5)

        x = args.x
        z = args.z
        if x is None and z is None:
            # Neither given -> don't just adopt whatever the arm happens to be
            # sitting at (leftover from a previous run, a crash, manual jogging,
            # etc). Home first so the fixed x/z for this sweep is deterministic
            # and reproducible run to run.
            print("no --x/--z given -- homing first for a repeatable starting position")
            arm.move_joints(HOME, duration=2.5)
            cur = arm.get_coords()
            x, z = cur[0], cur[2]
        elif x is None or z is None:
            # Only one given -- fill in the other from the CURRENT pose, not HOME,
            # since forcing a home here would silently move the arm away from
            # wherever the caller put it before running with an explicit x or z.
            cur = arm.get_coords()
            if x is None:
                x = cur[0]
            if z is None:
                z = cur[2]
        print(f"holding x={x:.2f} cm, z={z:.2f} cm; "
              f"sweeping y between {args.y_left} cm (left) and {args.y_right} cm (right)")

        init_speed = 20
        speed = init_speed
        try:
            while True:
                going_left = (speed % 2 == 0)
                target_y = args.y_left if going_left else args.y_right
                label = "left" if going_left else "right"

                print(f"[speed={speed} cm/s] moving {label} to y={target_y} cm ...", end=" ")
                ok = arm.send_coords(x=x, y=target_y, z=z, speed=float(speed))

                if not ok:
                    print("FAILED")
                    print(f"  reason: {arm.last_error}")
                    print("  stopping the sweep -- a higher speed will not fix this.")
                    break
                print("done")

                if args.confirm_every > 0 and (speed - init_speed) % args.confirm_every == 0:
                    ans = input(
                        f"  reached speed={speed} cm/s. Continue? [y/N]: "
                    ).strip().lower()
                    if ans != "y":
                        print("stopping at user's request")
                        break

                speed += 1

        except KeyboardInterrupt:
            # send_coords() streams setpoints through a loop that sleeps between
            # them (see Arm._execute); Ctrl+C interrupts that sleep and unwinds
            # straight out of send_coords() without going through its normal
            # error handling, since KeyboardInterrupt isn't caught there. So the
            # arm is NOT stopped automatically -- this call is what actually
            # halts it, immediately, at whatever position it was mid-move to.
            print("\nCtrl+C -- stopping the arm now.")
            arm.stop()

        print(f"\nfinal speed reached: {speed} cm/s")
    return 0


if __name__ == "__main__":
    _sys.exit(main())