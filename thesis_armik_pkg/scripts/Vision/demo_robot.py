#!/usr/bin/env python3
"""
DEMO, ROBOTLIKE profile: the cube exchange with industrial-robot motion
=======================================================================

The study's exchange (_demo.py: turns, Target Area, keys, --on-hand), moved
like LowH-HighR_v2.py moves:

  * every move is a STRAIGHT LINE along one axis at a time: up -> along x
    and along y (at the current height; the order whose corner is
    reachable, the corner farther from the base first) -> straight down
    onto the cube / the spot. PHASE_DELAY_S is held at each hand-over
    between legs. If neither order can be followed: a U of legs (below),
    and as a last resort one diagonal line;
  * the gripper stays straight down the whole time (yaw turned to meet the
    cube square-on, so the grasp works) -- it never looks at anything;
  * deliberate JERK (JERK_ENABLED): every streamed move -- the legs and the
    joint move HOME -- stops dead and continues at JERK_STREAM_STOPS random
    points (armik's stop-and-go stream stutter, config.STUTTER_TYPE = 0; each
    stop lasts config.JERK_SUBSTEP_DWELL_S). JERK_ENABLED = False -> clean;
  * between turns, and while the participant is due, the arm waits at HOME --
    except when the next turn is the robot's own (SKIP_HOME_BEFORE_ROBOT_TURN):
    after putting a cube away it stays at the transition height and goes on
    straight to its next cube (x / y legs, or the U: along x to
    U_TRANSIT_X_CM, along y, along x), clear of the base;
  * the return to HOME is straight legs too (HOME_LINEAR_PATH): along x to
    HOME's x, along y to HOME's y, then up to HOME -- on every return, or
    (False) only after the exchange's last turn.

    python3 scripts/Vision/demo_robot.py --mock                  # real camera, simulated arm
    python3 scripts/Vision/demo_robot.py --port /dev/ttyTHS1     # real arm (take_arm.sh first)
    python3 scripts/Vision/demo_robot.py --port /dev/ttyTHS1 --on-hand
    ./scripts/laptop_demo.sh robot [--on-hand]                   # laptop + Jetson bridge

Keys and the exchange: see _demo.py. Units: cm and degrees, robot frame.
"""

from __future__ import annotations

import sys as _sys

import math

import numpy as np

import _demo
import reach_cube as RC
from armik import config
from armik.arm import Execution
from armik.kinematics import pose_coords


# =============================================================================
# CONSTANTS
# =============================================================================

# -- jerk ---------------------------------------------------------------------------
JERK_ENABLED = False              # False = perfectly smooth (no stops at all)
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
U_TRANSIT_X_CM = 16.0            # when neither x-then-y nor y-then-x can be followed (a corner too
                                 # near the base or out of reach): along x to this x (then -/+ 2 cm),
                                 # along y, along x to the target. Too large and the corners (e.g.
                                 # y -20 at the low carry height) are out of reach
HOME_LINEAR_PATH = True          # True: every return to HOME is straight legs x -> y -> z;
                                 # False: only after the last turn (else one joint move)
HOME_TOL_CM = 2.0                # "at HOME" = the tip within this of HOME's tip: legs shorter than
                                 # this are skipped and there is no final settling move

# -- planning ahead -------------------------------------------------------------------
PREPLAN_ENABLED = True           # rehearse every move of a transfer on a simulated arm before the
                                 # first one, so no leg is planned / read live in between
PREPLAN_TOL_DEG = 2.0            # a rehearsed step is used only if every joint is within this of
                                 # its planned start; otherwise it runs live from where the arm is


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
    MOVE_STEPS = ("rise", "above cube", "descend", "lift", "above spot", "drop pos", "lower",
                  "retreat", "rest")             # rehearsed ahead (PREPLAN_ENABLED)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._tape, self._tape_step = {}, None

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
            self._say(f"level traverse at z {z:.1f} refused ({self.arm.last_error}) "
                      f"-- straight line to z {z_min:.1f}")
            if not self._move(x, y, z_min, orient):
                return False
        return self._hold()

    def _leg(self, x, y, z, orient, what, min_cm=0.3):
        """One straight leg (skipped if shorter than min_cm), then the hold."""
        tip = self.arm.get_coords()
        if math.dist(tip[:3], (x, y, z)) < min_cm:
            return True
        self._say(f"leg: {what}")
        return self._move(x, y, z, orient) and self._hold()

    def _say(self, text):
        if self._tape_step is not None:                   # rehearsing
            self._tape[self._tape_step].append(("say", text))
        else:
            print(f"[run]   {text}")

    @staticmethod
    def _home_tip():
        return [v / 10.0 for v in pose_coords(RC.HOME)[:3]] + list(pose_coords(RC.HOME)[3:6])

    def _leg_q(self, q, start_xyz, end, orient):
        """Joint pose at the end of a straight leg start_xyz -> end, planned
        from q (None = the live pose) without moving; None if it can't be
        followed. A leg shorter than 0.3 cm is free."""
        if math.dist(start_xyz, end) < 0.3:
            return q if q is not None else self.arm.get_angles()
        if q is None:
            pl = self.arm.plan_coords(*end, *orient)
        else:
            pl = self.plan_from(q, lambda a: a.plan_coords(*end, *orient))
        return pl.q_waypoints[-1] if pl.ok and pl.q_waypoints is not None else None

    def _axis_to(self, x, y, z_min, orient):
        """To above (x, y) along x and y legs at the current height (never
        below z_min). Both orders are checked before anything moves -- the one
        whose corner is farther from the base first -- and the first that can
        be followed is driven. Neither -> the U (_u_legs) -> one diagonal line
        (_across)."""
        tip = self.arm.get_coords()
        z = max(float(tip[2]), z_min)
        start, end = (float(tip[0]), float(tip[1]), z), (float(x), float(y), z)
        if float(tip[2]) < z - 0.3 and not self._up(z, orient):
            return False
        corners = [((x, tip[1], z), "x"), ((tip[0], y, z), "y")]
        corners.sort(key=lambda c: -math.hypot(c[0][0], c[0][1]))
        for corner, first in corners:
            q1 = self._leg_q(None, start, corner, orient)
            if q1 is None or self._leg_q(q1, corner, end, orient) is None:
                continue
            second = "y" if first == "x" else "x"
            return (self._leg(*corner, orient, f"along {first} to "
                              f"{corner[0] if first == 'x' else corner[1]:.1f}")
                    and self._leg(*end, orient, f"along {second} to "
                                  f"{x if second == 'x' else y:.1f}"))
        self._say("no x/y order can be followed -- U path")
        return self._u_legs(x, y, z_min, orient)

    def _u_legs(self, x, y, z_min, orient):
        """Three straight legs at the current height -- along x to a transit x
        (U_TRANSIT_X_CM, then U_TRANSIT_X_CM -/+ 2 cm), along y, along x to the
        target -- all checked before anything moves. None can be followed ->
        one diagonal line (_across)."""
        tip = self.arm.get_coords()
        z = max(float(tip[2]), z_min)
        start, end = (float(tip[0]), float(tip[1]), z), (float(x), float(y), z)
        for xt in (U_TRANSIT_X_CM, U_TRANSIT_X_CM - 2.0, U_TRANSIT_X_CM + 2.0):
            c1, c2 = (xt, start[1], z), (xt, end[1], z)
            q1 = self._leg_q(None, start, c1, orient)
            q2 = q1 is not None and self._leg_q(q1, c1, c2, orient)
            if q2 is None or q2 is False or self._leg_q(q2, c2, end, orient) is None:
                continue
            return (self._leg(*c1, orient, f"along x to {xt:.1f}")
                    and self._leg(*c2, orient, f"along y to {y:.1f}")
                    and self._leg(*end, orient, f"along x to {x:.1f}"))
        self._say("no U path can be followed either -- one straight line")
        return self._across(x, y, z_min, orient)

    def _home_legs(self):
        """Near HOME along straight legs: along x to HOME's x, along y to
        HOME's y, up to HOME's height (turning to HOME's orientation). Legs
        shorter than HOME_TOL_CM are skipped and nothing settles onto the exact
        HOME angles afterwards -- close enough is the point. A refused leg ->
        plain _home()."""
        hx, hy, hz, *ho = self._home_tip()
        tip = self.arm.get_coords()
        o = tuple(tip[3:6])
        tol = HOME_TOL_CM
        ok = (self._leg(hx, tip[1], tip[2], o, f"HOME: along x to {hx:.1f}", tol)
              and self._leg(hx, hy, tip[2], o, f"HOME: along y to {hy:.1f}", tol)
              and self._leg(hx, hy, hz, tuple(ho), f"HOME: up to z {hz:.1f}", tol))
        if ok:
            return True
        if self.conn.abort.is_set():
            return False
        self._say(f"straight legs to HOME refused ({self.arm.last_error}) -- joint move instead")
        return self._home()

    def _near_home(self):
        return math.dist(self.arm.get_coords()[:3], self._home_tip()[:3]) < HOME_TOL_CM

    def _rest(self):
        if self._near_home():
            return True
        if HOME_LINEAR_PATH or self.is_last_turn:
            return self._home_legs()
        return self._home()

    def _down(self, x, y, z, orient, speed=TRAVEL_SPEED_CM_S, hold=False):
        """Vertical leg down to z (skipped if already there); hold=True: the
        hold after it (only if it moved)."""
        tip = self.arm.get_coords()
        if abs(tip[2] - z) < 0.3:
            return True
        return self._move(x, y, z, orient, speed) and (not hold or self._hold())

    def _end(self):
        """The last step: HOME -- or none, when the robot's next pick follows."""
        if SKIP_HOME_BEFORE_ROBOT_TURN and self.next_turn_is_robot:
            return []
        return [self._end_step()]

    # -- planning ahead: rehearse, then replay -------------------------------------------

    def _steps(self, t):
        raw = self._live_steps(t)
        return ([("plan ahead", lambda: self._rehearse(raw))]
                + [(n, self._replayed(n, fn)) if n in self.MOVE_STEPS else (n, fn)
                   for n, fn in raw])

    def _rehearse(self, steps):
        """Run the transfer's movement steps on the planner arm (from where the
        real arm is now), recording each step's plans and holds -- the same leg
        choices the live code would make, with nothing moving."""
        self._tape = {}
        if not PREPLAN_ENABLED:
            return True
        a = self.planner()
        a.conn.raw.set_angles_directly([float(v) for v in self.arm.get_angles()])

        def record(plan):
            self._tape[self._tape_step].append(("plan", plan))
            a.conn.raw.set_angles_directly([float(v) for v in plan.q_waypoints[-1]])
            return Execution(ok=True)

        def hold():
            self._tape[self._tape_step].append(("hold",))
            return True

        real, saved_exec, at_rest = self.arm, a._execute, self.at_rest
        self.arm, a._execute, self._hold = a, record, hold
        done = []
        try:
            for name, fn in steps:
                if name not in self.MOVE_STEPS:
                    continue
                self._tape_step = name
                self._tape[name] = []
                if not fn():
                    del self._tape[name]
                    print(f"[plan]  {name}: could not be planned ahead -- it runs live")
                    break
                done.append(name)
        finally:
            self.arm, a._execute, self._tape_step, self.at_rest = real, saved_exec, None, at_rest
            del self._hold                                 # back to the class's _hold
        print(f"[plan]  planned ahead: {', '.join(done) or 'nothing'}")
        return True

    def _replayed(self, name, live):
        """The step `name`: its rehearsed plans if the arm is where they start
        (one read each), else the live code from where the arm is."""
        def run():
            entries = self._tape.get(name) if PREPLAN_ENABLED else None
            if entries is None:
                return live()
            for e in entries:
                if e[0] == "say":
                    print(f"[run]   {e[1]}")
                elif e[0] == "hold":
                    if not self._hold():
                        return False
                else:
                    plan = e[1]
                    q = np.asarray(self.arm.get_angles(), dtype=float)
                    dev = float(np.max(np.abs(q - np.asarray(plan.q_waypoints[0], dtype=float))))
                    if dev > PREPLAN_TOL_DEG:
                        print(f"[run]   {name}: the arm is {dev:.1f} deg from the planned start "
                              f"-- running it live from here")
                        return live()
                    ex = self.arm._execute(plan)
                    self.arm.last_plan, self.arm.last_execution = plan, ex
                    if not ex.ok:
                        self.arm.last_error = ex.error
                        return False
            if name == "rest":
                self.at_rest = not self.conn.abort.is_set()
            return True
        return run

    def _live_steps(self, t):
        steps = []
        carry_z, place_z, po = t["carry_z"], t["place_z"], t["po"]
        if not t["holding"]:
            o, gx, gy = t["o"], t["gx"], t["gy"]
            hover_z, grasp_z = t["hover_z"], t["grasp_z"]
            steps += [("rise", lambda: self._up(hover_z, o)),
                      ("open gripper", lambda: self._grip(RC.GRIP_OPEN_DEG)),
                      ("above cube", lambda: self._axis_to(gx, gy, hover_z, o)
                       and self._down(gx, gy, hover_z, o, hold=True)),
                      ("descend", lambda: self._down(gx, gy, grasp_z, o, APPROACH_SPEED_CM_S)),
                      ("grasp", self._grasp),
                      ("lift", lambda: self._up(carry_z, o))]
        else:
            steps.append(("rise", lambda: self._up(carry_z, po)))
        if t["dest"][0] == "hand":
            dpx, dpy, dpz = t["dpx"], t["dpy"], t["dpz"]
            # step names match RC.HandWatch.WATCH_STEPS
            steps += [("drop pos", lambda: self._axis_to(dpx, dpy, dpz, po)
                       and self._down(dpx, dpy, dpz, po)),
                      ("wait for hand", self._wait_hand),
                      ("release", self._release)] + self._end()
        else:
            px, py = t["px"], t["py"]
            steps += [("above spot", lambda: self._axis_to(px, py, carry_z, po)
                       and self._down(px, py, carry_z, po, hold=True)),
                      ("lower", lambda: self._down(px, py, place_z, po, APPROACH_SPEED_CM_S)),
                      ("release", self._release),
                      ("retreat", lambda: self._up(carry_z, po))] + self._end()
        return steps


if __name__ == "__main__":
    _sys.exit(_demo.run_demo(RobotRunner, "demo_robot", "Robotlike: straight legs, jerk",
                             setup_arm=setup_arm))
