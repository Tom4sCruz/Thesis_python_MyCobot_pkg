#!/usr/bin/env python3
"""
Template for your own scripts. Copy this; don't import it.

    python3 scripts/example.py --mock
    python3 scripts/example.py --port /dev/ttyTHS1
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time

from armik import Arm, config

HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    # `with` guarantees the port is closed and the arm told to stop, even if
    # your script raises partway through.
    with Arm(port=args.port, mock=args.mock) as arm:

        if not arm.conn.is_power_on():
            arm.conn.power_on()
            time.sleep(1.5)

        # 1. Always start from a known configuration.
        arm.move_joints(HOME, duration=3.0)

        # 2. Where is the tip? cm and degrees.
        x, y, z, rx, ry, rz = arm.get_coords()
        print(f"tip at x={x:.2f} y={y:.2f} z={z:.2f} cm")
        print(f"       rx={rx:.1f} ry={ry:.1f} rz={rz:.1f} deg")

        # 3. Full 6-DOF target: everything constrained.
        if arm.send_coords(x, y + 3.0, z - 2.0, rx, ry, rz, speed=4.0) == 1:
            print("full-pose move done")
        else:
            print("failed:", arm.last_error)

        # 4. Position only. Orientation is free, so the wrist will settle
        #    wherever is convenient -- but only just enough.
        arm.move_joints(HOME, duration=2.5)
        x, y, z, *_ = arm.get_coords()
        if arm.send_coords(x=x, y=y + 4.0, z=z - 3.0, speed=4.0) == 1:
            print("xyz move done; free-coord drift:", arm.last_plan.free_drift)

        # 5. A single axis. Everything else is held as close to put as possible.
        arm.move_joints(HOME, duration=2.5)
        _, _, z, *_ = arm.get_coords()
        if arm.send_coords(z=z - 3.0, speed=2.0) == 1:
            print("z-only move done; drift:", arm.last_plan.free_drift)

        # 6. Check reachability WITHOUT moving.
        plan = arm.plan_coords(x=25.0, speed=4.0)
        if plan.ok:
            print(f"x=25cm reachable: {plan.path_length_cm} cm path, "
                  f"peak {plan.peak_joint_dps:.0f} deg/s")
        else:
            print("x=25cm not reachable:", plan.error)

        # 7. Anything this package doesn't wrap is still reachable through the
        #    live pymycobot object, under the same lock so it can't collide
        #    with a running move.
        with arm.conn.lock:
            print("gripper moving:", arm.conn.raw.is_gripper_moving())

        # 8. Log a trajectory for later analysis.
        arm.move_joints(HOME, duration=2.5)
        _, y, z, *_ = arm.get_coords()
        if arm.send_coords(y=y + 5.0, speed=3.0) == 1:
            arm.last_execution.to_csv("trajectory.csv")
            print("wrote trajectory.csv")

        arm.move_joints(HOME, duration=3.0)


if __name__ == "__main__":
    main()
