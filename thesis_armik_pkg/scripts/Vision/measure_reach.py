#!/usr/bin/env python3
"""
VISION: measure how far the arm really reaches, by hand -> reach_limit.json
==========================================================================

reach_cube.py refuses a cube or target further from the J1 axis than a reach
limit (straight-down grasp), so a hopeless press is turned down at once
instead of after a slow IK search. Without a measurement that limit is
reach_cube.MAX_DOWN_REACH_CM (~28 cm, from an IK scan). This script measures
it on the real arm: the servos are released, you stretch the gripper out as
far as it goes -- pointing straight down, at grasp height -- and the arm's
own joint angles (armik FK) say how far that is.

    python3 scripts/Vision/measure_reach.py --port /dev/ttyTHS1   # take_arm.sh first
    python3 scripts/Vision/measure_reach.py --mock                # flow only

PROCEDURE
    1. The servos are released at start -- SUPPORT THE ARM, it goes limp.
    2. Guide the gripper outward, pointing straight down, with its tip at
       grasp height (about where it closes on a cube resting on the table:
       halfway up the cube). The live line shows r (distance from the J1
       axis), the tilt and whether z is in the grasp band.
    3. At the furthest point, hold still and press SPACE (one sample). It is
       refused if the gripper is tilted more than MAX_TILT_DEG or z is off.
    4. Repeat in a few directions (ahead, left, right -- wherever cubes and
       the target go). U drops the last sample.
    5. S saves: max_down_reach_cm = the SMALLEST of the samples (each one is
       "as far as it goes" in its direction, so the smallest is safe in all
       of them). reach_cube.py subtracts REACH_MARGIN_CM and also uses the
       sampled poses to seed its IK near the limit.
    Q / Esc quits (servos are re-engaged on the way out).

The limit only gates; the IK must still solve every waypoint. If a sample is
beyond what armik's model solves straight down, reach_cube says "no IK
solution" there: then the arm model (DH table, tool offset, joint limits in
armik/config.py) disagrees with the real arm, not the measurement. Each
sample is checked against armik's joint limits when it is taken.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import argparse
import json
import math
import select
import time

import numpy as np

from armik import Arm, ArmConnection, config, ik
from armik.kinematics import check_joint_limits
import _vision as V
import reach_cube as RC
from calibrate_base import gripper_tilt


MAX_TILT_DEG = 5.0               # a sample must be this close to straight down ...
GRASP_Z_BAND_CM = 1.5            # ... with the tip this close to the grasp height
GRASP_Z_CM = V.BASE_MARKER_POSE[2] + V.CUBE_SIDE_CM / 2.0   # table + half a cube (nominal frame)
POLL_S = 0.2                     # live readout period


def read_keys(timeout):
    """Characters typed since the last call (non-blocking, cbreak mode)."""
    out = ""
    while select.select([_sys.stdin], [], [], timeout)[0]:
        ch = _os.read(_sys.stdin.fileno(), 1).decode(errors="ignore")
        if not ch:
            break
        out += ch
        timeout = 0.0
    return out


def solves_down(coords, q):
    """Does armik's IK find this tip position with the gripper exactly down?"""
    orient = RC._down(RC.grip_yaw_of(coords[3:6]))
    sol = ik.try_solve(RC._ik_target(coords[0], coords[1], coords[2], *orient),
                       np.ones(6, dtype=bool), np.asarray(q, dtype=float))
    return sol is not None and not check_joint_limits(sol)


def main():
    ap = argparse.ArgumentParser(description="Measure the arm's straight-down reach by hand.")
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=RC.REACH_LIMIT_PATH)
    args = ap.parse_args()
    args.out = V.check_out_path(args.out, "--out")

    arm = Arm(connection=ArmConnection(args.port, mock=args.mock))
    print("Releasing the servos -- SUPPORT THE ARM.")
    time.sleep(1.0)
    arm.conn.release_all_servos()
    print(f"Guide the gripper as far out as it goes, straight down, tip at z ~ {GRASP_Z_CM:.1f} cm.\n"
          "Keys: SPACE sample   U undo   S save   Q quit\n")

    samples = []                                # dicts: q, coords, r
    coords, q = None, None
    tty_state = None
    if _sys.stdin.isatty():
        import termios
        import tty
        tty_state = termios.tcgetattr(_sys.stdin)
        tty.setcbreak(_sys.stdin.fileno())

    def say(msg):
        print("\r\033[K" + msg)

    try:
        while True:
            keys = read_keys(POLL_S)
            try:
                coords, q = arm.get_coords(), arm.get_angles()
            except Exception as exc:                # a missed serial read: keep the last one
                say(f"read failed: {exc}")
            if coords is not None:
                r = math.hypot(coords[0], coords[1])
                tilt = gripper_tilt(coords)
                z_ok = abs(coords[2] - GRASP_Z_CM) <= GRASP_Z_BAND_CM
                best = min(s["r"] for s in samples) if samples else None
                print(f"\r\033[Ktip ({coords[0]:6.1f}, {coords[1]:6.1f}, {coords[2]:6.1f}) cm  "
                      f"r {r:5.1f} cm  tilt {tilt:4.1f} deg {'ok' if tilt <= MAX_TILT_DEG else '--'}  "
                      f"z {'ok' if z_ok else '--'}  samples {len(samples)}"
                      + (f"  limit {best:.1f} cm" if best is not None else ""), end="", flush=True)

            for ch in keys.lower():
                if ch in ("q", "\x1b"):
                    return 0
                if ch == " " and coords is not None:
                    if tilt > MAX_TILT_DEG:
                        say(f"refused: gripper tilted {tilt:.1f} deg (max {MAX_TILT_DEG:.0f}) -- "
                            "hold it straight down")
                    elif not z_ok:
                        say(f"refused: tip z {coords[2]:.1f} cm, grasp height is {GRASP_Z_CM:.1f} "
                            f"+- {GRASP_Z_BAND_CM:.1f} cm")
                    else:
                        samples.append({"q": [float(v) for v in q],
                                        "coords": [float(v) for v in coords], "r": r})
                        bearing = math.degrees(math.atan2(coords[1], coords[0]))
                        note = ""
                        problems = check_joint_limits(q)
                        if problems:
                            note = ("  !! outside armik's joint limits (" + "; ".join(problems)
                                    + ") -- IK will not use this pose")
                        elif not solves_down(coords, q):
                            note = "  !! armik's IK does not solve this point straight down"
                        say(f"sample {len(samples)}: r {r:.1f} cm, bearing {bearing:.0f} deg{note}")
                elif ch == "u" and samples:
                    samples.pop()
                    say(f"undone, {len(samples)} samples")
                elif ch == "s":
                    if not samples:
                        say("S: take a sample first (SPACE)")
                        continue
                    limit = min(s["r"] for s in samples)
                    data = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "max_down_reach_cm": limit, "grasp_z_cm": GRASP_Z_CM,
                            "samples": samples}
                    with open(args.out, "w") as fh:
                        json.dump(data, fh, indent=2)
                    say(f"Saved {args.out}: reach {limit:.1f} cm from {len(samples)} samples "
                        f"(reach_cube.py uses {limit - RC.REACH_MARGIN_CM:.1f} cm)")
    except KeyboardInterrupt:
        return 0
    finally:
        if tty_state is not None:
            import termios
            termios.tcsetattr(_sys.stdin, termios.TCSADRAIN, tty_state)
        print("\nRe-engaging the servos.")
        arm.conn.focus_all_servos()
        arm.close()


if __name__ == "__main__":
    _sys.exit(main())
