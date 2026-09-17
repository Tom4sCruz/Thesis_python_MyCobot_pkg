#!/usr/bin/env python3
"""
Interactive keyboard jog tool -- find real-world cube coordinates by driving
the arm around by hand, for CUBES_INITIAL_POINTS / CUBES_TARGET_POINTS in
scripts/Profiles/*.py.

    python3 scripts/jog_arm.py --mock --yes      # no hardware
    python3 scripts/jog_arm.py --port /dev/ttyTHS1
    python3 scripts/jog_arm.py --port /dev/ttyUSB-mycobot   # via the socat bridge

CONTROLS
--------
  UP / DOWN     hold: x moves continuously (+/-) until released; tap: a brief nudge
  LEFT / RIGHT  hold/tap, same idea, on y
  W / S         hold/tap, same idea, on z
  F             toggle gripper open <-> closed
  ENTER         read the arm's ACTUAL current coords and log a waypoint
  [ / ]         decrease / increase the jog reach-ahead distance (see below)
  q             quit

Orientation (rx, ry, rz) is read once at startup and held fixed for the
whole session (no move happens on launch) unless --rx/--ry/--rz are given,
in which case one explicit, logged move normalizes orientation first.

Every ENTER press appends the measured coordinates to jog_waypoints.log as
a ready-to-paste Python tuple.

HOW "HOLD TO MOVE" ACTUALLY WORKS
----------------------------------
The myCobot firmware only ever accepts position+speed commands, never a
velocity command -- so Arm.send_coords()/send_path() (which STREAM a whole
smoothed trajectory from Python, blocking the caller until it's done) can't
give real "move while held, stop instantly on release" behavior: there's no
way to watch the keyboard while a blocking call is running, and calling
stop() from another thread wouldn't reliably stick anyway -- the streaming
loop has no cancel flag and would just re-issue its next scheduled command
moments later, overriding the stop.

Instead: on keydown, plan_coords() validates (IK + safety-checked, exactly
like send_coords() would, but WITHOUT moving) a target JOG_REACH_CM ahead in
the pressed direction, then that single joint-space target is sent with ONE
raw, non-blocking Arm.send_angles() call. The firmware then drives the
entire smooth motion toward it on its own hardware-side controller --
Python is free immediately and just watches the keyboard. While the key
keeps repeating, the far target is refreshed every ~200ms so the arm never
actually arrives; the moment no repeat shows up, arm.stop() is called, and
since nothing is left running to override it, the arm actually stays
stopped. A brief tap just means the refresh loop times out on its first
check -- so it moves for about one poll interval, not to the full
reach-ahead distance. If a target JOG_REACH_CM out isn't reachable (e.g.
near a workspace edge), the distance is halved and retried down to a small
floor before giving up and reporting the refusal.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import curses
import datetime

import numpy as np

from armik import Arm, config

GRIP_OPEN_DEG = 120.0
GRIP_CLOSED_DEG = 65.0
GRIP_SPEED = config.GRIPPER_DEFAULT_SPEED

WAYPOINT_LOG = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "jog_waypoints.log")

REACH_MIN_CM = 1.0
REACH_MAX_CM = 30.0
REACH_INCREMENT = 1.0
REACH_BACKOFF_FLOOR_CM = 0.5  # give up (and report refusal) once a halved retry drops below this

JOG_POLL_MS = 200  # no repeat of the held key within this long -> released; also the
                   # refresh cadence for re-aiming the far target while still held

MAX_LOG_LINES = 12


def _clamp_reach(reach):
    return max(REACH_MIN_CM, min(REACH_MAX_CM, round(reach, 1)))


def run(stdscr, arm, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    x, y, z, rx, ry, rz = arm.get_coords()

    if args.rx is not None or args.ry is not None or args.rz is not None:
        rx = args.rx if args.rx is not None else rx
        ry = args.ry if args.ry is not None else ry
        rz = args.rz if args.rz is not None else rz
        arm.send_coords(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, speed=args.speed)

    gv = arm.get_gripper_value()
    gripper_closed = (
        gv is not None and gv / 100.0 * config.MAX_GRIPPER_DEG < (GRIP_OPEN_DEG + GRIP_CLOSED_DEG) / 2.0
    )

    reach = args.step
    log = []

    def status(msg):
        log.append(msg)
        del log[:-MAX_LOG_LINES]

    def draw():
        stdscr.erase()
        stdscr.addstr(0, 0, "JOG ARM  --  UP/DOWN=x  LEFT/RIGHT=y  W/S=z  F=gripper  "
                             "ENTER=log waypoint  [ ]=reach dist  q=quit")
        stdscr.addstr(2, 0, f"x={x:7.2f}  y={y:7.2f}  z={z:7.2f}  "
                            f"rx={rx:7.2f}  ry={ry:7.2f}  rz={rz:7.2f}  cm/deg")
        stdscr.addstr(3, 0, f"reach={reach:.1f} cm   speed={args.speed:.1f} cm/s   "
                            f"gripper={'CLOSED' if gripper_closed else 'OPEN'}")
        stdscr.addstr(5, 0, "-" * 70)
        for i, line in enumerate(log):
            stdscr.addstr(6 + i, 0, line[:70])
        stdscr.refresh()

    def _direction_for(key):
        if key == curses.KEY_UP:
            return (1.0, 0.0, 0.0)
        if key == curses.KEY_DOWN:
            return (-1.0, 0.0, 0.0)
        if key == curses.KEY_LEFT:
            return (0.0, 1.0, 0.0)
        if key == curses.KEY_RIGHT:
            return (0.0, -1.0, 0.0)
        if key in (ord("w"), ord("W")):
            return (0.0, 0.0, 1.0)
        if key in (ord("s"), ord("S")):
            return (0.0, 0.0, -1.0)
        return None

    def jog_toward(ux, uy, uz):
        """Validate (IK + safety-checked, like send_coords() but without
        moving) a target `reach` cm out along the unit direction (ux,uy,uz)
        from wherever the arm actually is right now, backing off to a
        smaller distance if that far isn't reachable; on success, send it as
        ONE raw, non-blocking joint-space command so the firmware drives the
        whole motion itself. Returns True on success.

        Position is re-read fresh each call (it must be, to aim from wherever
        the arm currently is), but orientation is NOT re-read -- it stays
        pinned to this session's fixed rx/ry/rz. Re-reading and re-targeting
        orientation every refresh would feed each call's small IK residual
        back in as the next call's target, letting drift compound over a
        long hold instead of cancelling out."""
        cx, cy, cz = arm.get_coords()[:3]
        dist = reach
        while dist >= REACH_BACKOFF_FLOOR_CM:
            plan = arm.plan_coords(x=cx + ux * dist, y=cy + uy * dist, z=cz + uz * dist,
                                   rx=rx, ry=ry, rz=rz, speed=args.speed)
            if plan.ok:
                q0, q1 = plan.q_waypoints[0], plan.q_waypoints[-1]
                dps = float(np.max(np.abs(q1 - q0))) / plan.duration_s * config.STREAM_SPEED_GAIN
                if not arm.send_angles(q1, dps):
                    status(f"REFUSED -- {arm.last_error}")
                    return False
                return True
            dist /= 2.0
        status(f"REFUSED -- {plan.error}")
        return False

    def jog_hold(key):
        """Keep the arm moving in one direction for as long as `key` keeps
        repeating within JOG_POLL_MS; see the module docstring for how this
        approximates "hold to move" on hardware that has no velocity mode."""
        nonlocal x, y, z
        ux, uy, uz = _direction_for(key)
        if not jog_toward(ux, uy, uz):
            draw()
            return
        while True:
            stdscr.timeout(JOG_POLL_MS)
            nxt = stdscr.getch()
            stdscr.nodelay(True)
            if nxt == -1:
                arm.stop()
                x, y, z = arm.get_coords()[:3]
                status("stopped")
                draw()
                return
            # drain any further backlog -- only whether SOMETHING repeated
            # matters here, not how many, so a burst can't skip ahead
            while True:
                more = stdscr.getch()
                if more == -1:
                    break
                nxt = more
            if nxt != key:
                arm.stop()
                x, y, z = arm.get_coords()[:3]
                curses.ungetch(nxt)
                draw()
                return
            if not jog_toward(ux, uy, uz):
                arm.stop()
                x, y, z = arm.get_coords()[:3]
                draw()
                return

    status(f"start: x={x:.2f} y={y:.2f} z={z:.2f} rx={rx:.2f} ry={ry:.2f} rz={rz:.2f}")
    draw()

    while True:
        key = stdscr.getch()
        if key == -1:
            curses.napms(10)
            continue

        if key in (ord("q"), ord("Q")):
            break

        if _direction_for(key) is not None:
            jog_hold(key)
            continue

        if key in (ord("["),):
            reach = _clamp_reach(reach - REACH_INCREMENT)
            draw()
        elif key in (ord("]"),):
            reach = _clamp_reach(reach + REACH_INCREMENT)
            draw()
        elif key in (ord("f"), ord("F")):
            target = GRIP_OPEN_DEG if gripper_closed else GRIP_CLOSED_DEG
            arm.send_gripper(target, speed=GRIP_SPEED)
            gripper_closed = not gripper_closed
            status(f"gripper -> {'CLOSED' if gripper_closed else 'OPEN'} ({target:.0f} deg)")
            draw()
        elif key in (curses.KEY_ENTER, 10, 13):
            cx, cy, cz, crx, cry, crz = arm.get_coords()
            line = f"({cx:.2f}, {cy:.2f}, {cz:.2f}),  # {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
            with open(WAYPOINT_LOG, "a") as f:
                f.write(line + "\n")
            status(f"logged: {line}")
            draw()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--step", type=float, default=8.0, dest="step",
                     help="jog reach-ahead distance, cm -- how far past the current "
                          "position the arm aims while a direction key is held")
    ap.add_argument("--speed", type=float, default=10.0, help="cm/s")
    ap.add_argument("--rx", type=float, default=None, help="normalize orientation on start (deg)")
    ap.add_argument("--ry", type=float, default=None, help="normalize orientation on start (deg)")
    ap.add_argument("--rz", type=float, default=None, help="normalize orientation on start (deg)")
    args = ap.parse_args()

    if not args.mock and not args.yes:
        print("This will move the robot arm. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
    try:
        curses.wrapper(run, arm, args)
    finally:
        arm.close()
    print(f"waypoints logged to {WAYPOINT_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
