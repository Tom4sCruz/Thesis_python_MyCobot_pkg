#!/usr/bin/env python3
"""
DEMO, ROBOTLIKE profile: the cube exchange with industrial-robot motion
=======================================================================

The study's exchange (_demo.py: turns, Target Area, keys, --on-hand), moved
like LowH-HighR_v2.py moves:

  * every move is a STRAIGHT LINE, and only ever vertical or horizontal:
    up -> across (at the current height) -> straight down onto the cube /
    the spot. PHASE_DELAY_S is held at each hand-over between legs;
  * the gripper stays straight down the whole time (yaw turned to meet the
    cube square-on, so the grasp works) -- it never looks at anything;
  * deliberate JERK (JERK_ENABLED): every streamed move -- the legs and the
    joint move HOME -- stops dead and continues at JERK_STREAM_STOPS random
    points (armik's stop-and-go stream stutter, config.STUTTER_TYPE = 0; each
    stop lasts config.JERK_SUBSTEP_DWELL_S). JERK_ENABLED = False -> clean;
  * between turns, and while the participant is due, the arm waits at HOME --
    except when the next turn is the robot's own (SKIP_HOME_BEFORE_ROBOT_TURN):
    after putting a cube away it stays at the transition height and swings
    straight across to above its next cube.

    python3 scripts/Vision/demo_robot.py --mock                  # real camera, simulated arm
    python3 scripts/Vision/demo_robot.py --port /dev/ttyTHS1     # real arm (take_arm.sh first)
    python3 scripts/Vision/demo_robot.py --port /dev/ttyTHS1 --on-hand
    ./scripts/laptop_demo.sh robot [--on-hand]                   # laptop + Jetson bridge

Keys and the exchange: see _demo.py. Units: cm and degrees, robot frame.
"""

from __future__ import annotations

import sys as _sys

import _demo
import reach_cube as RC
from armik import config


# =============================================================================
# CONSTANTS
# =============================================================================

# -- jerk ---------------------------------------------------------------------------
JERK_ENABLED = True              # False = perfectly smooth (no stops at all)
JERK = 5.0                       # arms the stutter; with stop-and-go its size does nothing
JERK_STREAM_STOPS = 4            # stops per streamed move (each leg, the move HOME)
                                 # (stop length: config.JERK_SUBSTEP_DWELL_S)

# -- motion -------------------------------------------------------------------------
TRAVEL_SPEED_CM_S = 6.0          # up / across / down legs
APPROACH_SPEED_CM_S = 3.0        # the last descent onto a cube / lowering it onto the table
PHASE_DELAY_S = 0.3              # hold at each hand-over between a vertical and a horizontal leg
SKIP_HOME_BEFORE_ROBOT_TURN = True   # True: when the next turn is the robot's, don't go HOME
                                 # after this transfer -- the next pick swings across from here,
                                 # at the transition (carry) height


# =============================================================================
# MOTION
# =============================================================================

def setup_arm(arm):
    """Arm the stop-and-go stutter on every streamed move (or not)."""
    if JERK_ENABLED:
        if config.STUTTER_TYPE != 0:
            print(f"WARNING: config.STUTTER_TYPE = {config.STUTTER_TYPE}: the streamed "
                  f"stop-and-go needs 0 -- the moves will tremble instead")
        config.JERK_STREAM_STOPS = int(JERK_STREAM_STOPS)
        arm.jerk = float(JERK)
    else:
        config.JERK_STREAM_STOPS = 0
        arm.jerk = 0.0
    print(f"jerk: {'ON (' + str(JERK_STREAM_STOPS) + ' stops per move)' if JERK_ENABLED else 'off'}")


class RobotRunner(_demo.DemoRunner):
    """Vertical / horizontal straight legs, gripper always down, rest at HOME."""

    REST_WHILE_WAITING = True        # _rest() is the inherited HOME

    def _move(self, x, y, z, orient, speed=TRAVEL_SPEED_CM_S):
        return super()._move(x, y, z, orient, speed)

    def _hold(self):
        return self._wait(PHASE_DELAY_S) if PHASE_DELAY_S > 0.0 else True

    def _up(self, z, orient):
        """Vertical leg up to z (skipped if already there)."""
        tip = self.arm.get_coords()
        if tip[2] >= z - 0.3:
            return True
        return self._move(tip[0], tip[1], z, orient) and self._hold()

    def _across(self, x, y, z_min, orient):
        """Horizontal leg to above (x, y), at the current height (never below
        z_min). If that line is out of reach (e.g. from HOME's high tip to a far
        cube), a straight line to (x, y, z_min) instead."""
        tip = self.arm.get_coords()
        z = max(float(tip[2]), z_min)
        if not self._move(x, y, z, orient):
            if self.conn.abort.is_set():
                return False
            print(f"[run]   level traverse at z {z:.1f} refused ({self.arm.last_error}) "
                  f"-- straight line to z {z_min:.1f}")
            if not self._move(x, y, z_min, orient):
                return False
        return self._hold()

    def _down(self, x, y, z, orient, speed=TRAVEL_SPEED_CM_S):
        """Vertical leg down to z (skipped if already there)."""
        tip = self.arm.get_coords()
        if abs(tip[2] - z) < 0.3:
            return True
        return self._move(x, y, z, orient, speed)

    def _end(self):
        """The last step: HOME -- or none, when the robot's next pick follows."""
        if SKIP_HOME_BEFORE_ROBOT_TURN and self.next_turn_is_robot:
            return []
        return [self._end_step()]

    def _steps(self, t):
        steps = []
        carry_z, place_z, po = t["carry_z"], t["place_z"], t["po"]
        if not t["holding"]:
            o, gx, gy = t["o"], t["gx"], t["gy"]
            hover_z, grasp_z = t["hover_z"], t["grasp_z"]
            steps += [("rise", lambda: self._up(hover_z, o)),
                      ("open gripper", lambda: self._grip(RC.GRIP_OPEN_DEG)),
                      ("above cube", lambda: self._across(gx, gy, hover_z, o)
                       and self._down(gx, gy, hover_z, o) and self._hold()),
                      ("descend", lambda: self._down(gx, gy, grasp_z, o, APPROACH_SPEED_CM_S)),
                      ("grasp", self._grasp),
                      ("lift", lambda: self._up(carry_z, o))]
        else:
            steps.append(("rise", lambda: self._up(carry_z, po)))
        if t["dest"][0] == "hand":
            dpx, dpy, dpz = t["dpx"], t["dpy"], t["dpz"]
            # step names match RC.HandWatch.WATCH_STEPS
            steps += [("drop pos", lambda: self._across(dpx, dpy, dpz, po)
                       and self._down(dpx, dpy, dpz, po)),
                      ("wait for hand", self._wait_hand),
                      ("release", self._release)] + self._end()
        else:
            px, py = t["px"], t["py"]
            steps += [("above spot", lambda: self._across(px, py, carry_z, po)
                       and self._down(px, py, carry_z, po) and self._hold()),
                      ("lower", lambda: self._down(px, py, place_z, po, APPROACH_SPEED_CM_S)),
                      ("release", self._release),
                      ("retreat", lambda: self._up(carry_z, po))] + self._end()
        return steps


if __name__ == "__main__":
    _sys.exit(_demo.run_demo(RobotRunner, "demo_robot", "Robotlike: straight legs, jerk",
                             setup_arm=setup_arm))
