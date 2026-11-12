#!/usr/bin/env python3
"""Enable (focus) or disable (release) the myCobot 280's servos.

Standalone: needs only pymycobot, so it can be copied anywhere (e.g.
~/Scripts/ on the Jetson) and driven from a shell alias:

    alias servos='python3 $HOME/Scripts/toggle_servos.py'
    servos --release              # all servos off (hold the arm!)
    servos --focus --id 236       # servos 2, 3 and 6 on

--id takes the servo ids as one string of digits 1-6, no separators.
"""

import argparse
import sys
import time

DEFAULT_PORT = "/dev/ttyTHS1"   # same as armik/config.py
DEFAULT_BAUD = 1000000
COUNTDOWN_S = 3
GAP_S = 0.05                    # firmware drops a command right behind another
SERVO_IDS = "123456"


def parse_ids(text):
    if not text:
        raise argparse.ArgumentTypeError("give at least one servo id (digits 1-6)")
    bad = sorted({c for c in text if c not in SERVO_IDS})
    if bad:
        raise argparse.ArgumentTypeError(
            f"invalid servo id(s) {', '.join(repr(c) for c in bad)} in {text!r}: "
            f"use digits 1-6 only, no separators (e.g. 135)")
    return [int(c) for c in dict.fromkeys(text)]   # dedupe, keep order


def main():
    p = argparse.ArgumentParser(
        description="Enable (--focus) or disable (--release) myCobot 280 servos.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--release", action="store_true", help="disable servos (arm goes limp)")
    mode.add_argument("--focus", action="store_true", help="enable servos (arm holds its pose)")
    p.add_argument("--id", type=parse_ids, metavar="IDS",
                   help="only these servos, as digits 1-6 (e.g. 135); default: all")
    p.add_argument("--port", default=DEFAULT_PORT, help=f"serial port (default {DEFAULT_PORT})")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                   help=f"baud rate (default {DEFAULT_BAUD})")
    args = p.parse_args()

    verb = "Releasing" if args.release else "Focusing"
    which = "all servos" if args.ids is None else \
        "servo" + ("s " if len(args.ids) > 1 else " ") + ", ".join(map(str, args.ids))
    print(f"{verb} {which} on {args.port}")
    if args.release:
        print("WARNING: released joints go limp -- hold the arm so it doesn't drop.")

    try:
        from pymycobot import MyCobot280
    except ImportError:
        try:
            from pymycobot.mycobot import MyCobot as MyCobot280  # older versions
        except ImportError:
            sys.exit("pymycobot is not installed (`pip install pymycobot`).")
    try:
        mc = MyCobot280(args.port, args.baud)
    except Exception as exc:
        sys.exit(f"could not open {args.port}: {exc}")
    time.sleep(0.1)

    try:
        for n in range(COUNTDOWN_S, 0, -1):
            print(f"  {n}...", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\naborted -- nothing sent.")
        return 1

    if args.ids is None:
        mc.release_all_servos() if args.release else mc.focus_all_servos()
    else:
        for i, sid in enumerate(args.ids):
            if i:
                time.sleep(GAP_S)
            mc.release_servo(sid) if args.release else mc.focus_servo(sid)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
