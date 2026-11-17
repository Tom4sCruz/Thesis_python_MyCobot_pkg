#!/usr/bin/env python3
"""
DEMO (no profiles): the robot and a participant exchange every cube in turn
===========================================================================

The study's task with plain linear moves (as reach_cube.py, no movement
profiles). Cubes are found the way reach_cube.py --cube-z table finds them:
each cube is assumed to rest on the table, the depth stream is not used for
them.

    python3 scripts/Vision/demo_no-profiles.py --mock                  # real camera, simulated arm
    python3 scripts/Vision/demo_no-profiles.py --port /dev/ttyTHS1     # real arm (take_arm.sh first)
    python3 scripts/Vision/demo_no-profiles.py --port /dev/ttyTHS1 --on-hand   # hand it over

THE EXCHANGE
    The turns come from ROBOT_CUBE_IDS / HUMAN_CUBE_IDS / ROBOT_FIRST: the two
    sides alternate (ROBOT_FIRST picks who starts); once one side has no cubes
    left the other takes its remaining turns. Add an ID to either tuple and
    the demo expects one more exchange.

    robot turn        the robot picks its next cube (ascending ID) wherever it
                      is in reach, and leaves it in the centre of the Target
                      Area -- or, with --on-hand, carries it to DROP_POS_CM
                      (reach_cube.py) and opens once a hand is held still
                      under it. Before placing on the table it waits for the
                      Target Area to be clear of other cubes.
    participant turn  the participant takes the robot's cube away and puts one
                      of their own (any of HUMAN_CUBE_IDS, any order) in the
                      Target Area. Once it has been steady inside the area for
                      PICKUP_DELAY_S, the robot picks it and puts it at its
                      HUMAN_PUTAWAY_CM slot.

    Every run starts and ends at HOME. After the last turn: FINISHED.

KEYS
    SPACE   idle            -> start the exchange
            waiting         -> pause the exchange (nothing moving)
            moving          -> STOP the arm at once (paused)
            paused / failed -> resume the current turn (re-planned from the
                               arm's pose; if a cube is held it is carried on)
    N       skip the current turn (not while moving)
    G       toggle "holding a cube" by hand
    O       open the gripper (and clear "holding")
    H       go HOME (pauses the exchange)
    B       forget the locked base pose and re-acquire it
    Q / Esc stop and quit

The exchange itself (ROBOT_CUBE_IDS / HUMAN_CUBE_IDS / ROBOT_FIRST, the
Target Area, put-away slots, speeds, heights), the keys and the camera loop
live in _demo.py, shared with the movement profiles demo_human.py and
demo_robot.py -- change them there. The grasp geometry, gripper handling,
HOME and DROP_POS_CM / hand watching come from reach_cube.py.

Units: cm and degrees (armik's public API), robot frame.
"""

from __future__ import annotations

import sys as _sys

import _demo


if __name__ == "__main__":
    _sys.exit(_demo.run_demo(_demo.DemoRunner, "demo_no-profiles", "linear moves"))
