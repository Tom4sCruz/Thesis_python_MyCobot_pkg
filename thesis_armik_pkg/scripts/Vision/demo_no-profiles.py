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

The grasp geometry, gripper handling, HOME and DROP_POS_CM / hand watching
come from reach_cube.py (import reach_cube as RC), so calibrating those there
(GRASP_OFFSET_MM, GRIP_*, ...) applies here too. This file owns the exchange,
the Target Area, speeds and heights.

Units: cm and degrees (armik's public API), robot frame.
"""

from __future__ import annotations

import os as _os, sys as _sys
# scripts/Vision/ is two levels below the package root -> three dirname() calls
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import argparse
import math
import threading
import time
from collections import deque

import cv2
import numpy as np

from armik import Arm, config
import _vision as V
import _hands as H
import reach_cube as RC


# =============================================================================
# CONSTANTS
# =============================================================================

# -- the exchange ---------------------------------------------------------------
ROBOT_CUBE_IDS = (0, 1)          # the robot's cubes (ArUco ID = cube); picked in ascending order
HUMAN_CUBE_IDS = (2,)            # the participant's cubes; they may hand them over in any order
ROBOT_FIRST = True               # True: the robot exchanges first; False: the participant does

# Target Area: two opposite corners (x, y) of a rectangle on the table, robot
# frame cm (axes-aligned). The robot leaves its cubes in its centre and picks
# the participant's cubes from it.
TARGET_AREA_CM = [(16.0, -6.0), (24.0, 6.0)]

# Where the robot puts each participant's cube it took: one table-level (x, y)
# robot-frame cm per HUMAN_CUBE_IDS entry, in the same order.
HUMAN_PUTAWAY_CM = [(8.0, -20.0)]

PICKUP_DELAY_S = 2.0             # participant's cube steady in the area this long -> robot reaches

# -- overlay ----------------------------------------------------------------------
SHOW_REACH_CIRCLE = True         # draw the straight-down reach circle (RC.MAX_DOWN_REACH_CM)

# -- motion -----------------------------------------------------------------------
TRAVEL_SPEED_CM_S = 6.0
APPROACH_SPEED_CM_S = 3.0        # descend onto a cube / lower it onto the table
HOVER_ABOVE_CM = 6.0             # approach / carry height above the grasp / place height
RELEASE_CLEARANCE_CM = 0.5       # cube bottom this far above the table when it's let go

WINDOW = "demo_no-profiles  (SPACE start/pause/resume, N skip, G holding, O open, H home, Q quit)"

ALL_CUBE_IDS = tuple(ROBOT_CUBE_IDS) + tuple(HUMAN_CUBE_IDS)

# colours (BGR)
COL_ROBOT_CUBE = (0, 200, 255)
COL_HUMAN_CUBE = (200, 120, 255)
COL_DONE_CUBE = (140, 140, 140)
COL_ACTIVE_CUBE = (255, 255, 255)
COL_IN = (255, 0, 0)             # blue: cube inside the circle / area
COL_OUT = (0, 0, 255)            # red


def owner(cid):
    return "robot" if cid in ROBOT_CUBE_IDS else "participant"


def area_bounds():
    (x0, y0), (x1, y1) = TARGET_AREA_CM
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def area_center():
    x0, x1, y0, y1 = area_bounds()
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def in_area(xy):
    x0, x1, y0, y1 = area_bounds()
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def putaway_of(cid):
    return HUMAN_PUTAWAY_CM[list(HUMAN_CUBE_IDS).index(cid)]


def place_heights(table_z):
    """(place tip z, carry tip z) for a cube set down on the table."""
    place_z = table_z + V.CUBE_SIDE_CM / 2.0 + RELEASE_CLEARANCE_CM + RC.GRASP_Z_OFFSET_CM
    return place_z, place_z + HOVER_ABOVE_CM


def build_turns(robot_ids=ROBOT_CUBE_IDS, human_ids=HUMAN_CUBE_IDS, robot_first=ROBOT_FIRST):
    """[("robot", cube_id) | ("human", None), ...]: alternate, then the rest.
    A participant's turn has no fixed cube -- any of theirs not yet exchanged."""
    queues = {"robot": [("robot", c) for c in sorted(robot_ids)],
              "human": [("human", None) for _ in human_ids]}
    order = ("robot", "human") if robot_first else ("human", "robot")
    turns = []
    while queues["robot"] or queues["human"]:
        for side in order:
            if queues[side]:
                turns.append(queues[side].pop(0))
    return turns


def check_config(table_z, on_hand):
    """(errors, warnings) for the constants above; errors stop the script."""
    errors, warnings = [], []
    if not ROBOT_CUBE_IDS and not HUMAN_CUBE_IDS:
        errors.append("no cubes: ROBOT_CUBE_IDS and HUMAN_CUBE_IDS are both empty")
    both = set(ROBOT_CUBE_IDS) & set(HUMAN_CUBE_IDS)
    if both:
        errors.append(f"cube IDs {sorted(both)} are in both ROBOT_CUBE_IDS and HUMAN_CUBE_IDS")
    if len(set(ALL_CUBE_IDS)) != len(ALL_CUBE_IDS):
        errors.append("a cube ID is listed twice")
    for cid in ALL_CUBE_IDS:
        if cid in (V.BASE_ID, V.TARGET_ID):
            errors.append(f"cube ID {cid} clashes with the BASE / TARGET marker IDs")
    if len(HUMAN_PUTAWAY_CM) != len(HUMAN_CUBE_IDS):
        errors.append(f"HUMAN_PUTAWAY_CM has {len(HUMAN_PUTAWAY_CM)} spots for "
                      f"{len(HUMAN_CUBE_IDS)} participant cubes (need one each, same order)")
    if len(TARGET_AREA_CM) != 2:
        errors.append("TARGET_AREA_CM needs exactly two corner points")
    else:
        x0, x1, y0, y1 = area_bounds()
        if x1 - x0 < V.CUBE_SIDE_CM or y1 - y0 < V.CUBE_SIDE_CM:
            errors.append(f"TARGET_AREA_CM is {x1 - x0:.1f} x {y1 - y0:.1f} cm -- smaller "
                          f"than a cube")
        far = max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1))
        if far > RC.MAX_TILTED_REACH_CM:
            warnings.append(f"a Target Area corner is {far:.1f} cm from the base -- a cube "
                            f"there is out of reach (~{RC.MAX_TILTED_REACH_CM:.0f} cm)")
    if errors:
        return errors, warnings

    # every fixed place the robot sets a cube down must solve (from HOME)
    place_z, carry_z = place_heights(table_z)
    spots = [("the Target Area centre", area_center())] if not on_hand else []
    spots += [(f"HUMAN_PUTAWAY_CM for cube {cid}", putaway_of(cid)) for cid in HUMAN_CUBE_IDS]
    for what, (x, y) in spots:
        def pts(orient, x=x, y=y):
            dx, dy = RC.grasp_offset_xy(orient)
            return [("above", x + dx, y + dy, carry_z), ("place", x + dx, y + dy, place_z)]
        _, why = RC.plan_orientation(pts, RC.PICK_ORIENTATION_DEG[2],
                                     [np.array(RC.HOME, dtype=float)], (x, y), what)
        if why:
            errors.append(f"{what} ({x:.1f}, {y:.1f}): {why}")
    for cid in HUMAN_CUBE_IDS:
        if in_area(putaway_of(cid)):
            warnings.append(f"the put-away spot of cube {cid} is inside the Target Area")
    if on_hand:
        dpx, dpy, dpz = RC.DROP_POS_CM
        _, why = RC.plan_orientation(lambda o: [("drop", dpx, dpy, dpz)],
                                     RC.PICK_ORIENTATION_DEG[2],
                                     [np.array(RC.HOME, dtype=float)], (dpx, dpy), "DROP_POS_CM")
        if why:
            errors.append(f"DROP_POS_CM: {why}")
    return errors, warnings


# =============================================================================
# PERCEPTION
# =============================================================================

class DemoScene(RC.Scene):
    """RC.Scene in table mode, tracking every cube of the exchange at once."""

    def __init__(self, *a, on_hand=False, **kw):
        super().__init__(*a, **kw)
        self.on_hand = on_hand
        self.hist = {cid: deque(maxlen=RC.STABLE_FRAMES) for cid in ALL_CUBE_IDS}

    def update(self, frame, frozen):
        super().update(frame, frozen)
        if (self.on_hand and not frozen and self.T_robot_cam is not None
                and V.BASE_ID in self.markers):
            self._update_table_plane(self.markers)      # only for the hand's distance
        if self.T_robot_cam is None:
            self.cubes = {}
        else:
            # every ID of the exchange (V.CUBE_IDS may not list newly added cubes)
            self.cubes = V.cubes_from_faces(
                self.all_markers, self.K, self.dist, self.T_robot_cam, cube_ids=ALL_CUBE_IDS,
                side=V.CUBE_SIDE_CM, marker_len=self.args.cube_marker_cm, table_z=self.table_z)
        for cid, h in self.hist.items():
            c = self.cubes.get(cid)
            if c is None:
                h.clear()
            else:
                h.append((c["center"].copy(), c["yaw"]))

    def stable(self, cid):
        """(center, yaw) median of the last STABLE_FRAMES if steady, else None."""
        h = self.hist.get(cid)
        if h is None or len(h) < RC.STABLE_FRAMES:
            return None
        centers = np.array([c for c, _ in h])
        med = np.median(centers, axis=0)
        if np.max(np.linalg.norm(centers - med, axis=1)) > RC.STABLE_TOL_CM:
            return None
        a = np.radians(4.0 * np.array([y for _, y in h]))
        yaw = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 4.0
        return med, V.wrap90(yaw)


# =============================================================================
# MOTION
# =============================================================================

def _timed(name, fn):
    """fn, printing how long it took (each step's own moves include their planning)."""
    def run():
        t0 = time.monotonic()
        ok = fn()
        print(f"[run]   {name} took {time.monotonic() - t0:.2f} s")
        return ok
    return run


class DemoRunner(RC.Runner):
    """RC.Runner with one planner: pick a cube (unless held) and set it down at
    a table spot or hand it over at DROP_POS_CM."""

    def __init__(self, arm, conn, on_hand=False):
        super().__init__(arm, conn, no_place=False, on_hand=on_hand)
        self.message = "press SPACE to start the exchange"

    def _move(self, x, y, z, orient, speed=TRAVEL_SPEED_CM_S):
        return super()._move(x, y, z, orient, speed)

    def start_transfer(self, cube_center, cube_yaw, dest, table_z):
        """dest = ("table", x, y) or ("hand",). Plans from the current pose and
        starts the run; returns an error string (nothing moves) or None."""
        t_plan = time.monotonic()
        q0 = np.array(self.arm.get_angles(), dtype=float)
        tip = self.arm.get_coords()
        down = RC._pointing_down(tip)
        home_q = np.array(RC.HOME, dtype=float)
        seeds = [q0, home_q] if down else [home_q]

        place_z, carry_z = place_heights(table_z)
        cube_plan, seed = None, None
        if not self.holding:
            if cube_center is None:
                return "cube not seen"
            cx, cy = float(cube_center[0]), float(cube_center[1])
            grasp_z = float(cube_center[2]) + RC.GRASP_Z_OFFSET_CM
            hover_z = grasp_z + HOVER_ABOVE_CM
            carry_z = max(carry_z, hover_z)

            def cube_points(orient):
                dx, dy = RC.grasp_offset_xy(orient)
                return [("above cube", cx + dx, cy + dy, hover_z),
                        ("grasp", cx + dx, cy + dy, grasp_z),
                        ("lift", cx + dx, cy + dy, carry_z)]

            cube_plan, why = RC.plan_orientation(cube_points, RC.grip_rz(cube_yaw), seeds,
                                                 (cx, cy), "the cube")
            if why:
                return why
            _, gx, gy, _ = cube_points(cube_plan["orient"])[0]
            seed, yaw = cube_plan["seed"], cube_plan["yaw"]
        else:
            yaw = RC.grip_yaw_of(tip[3:6]) if down else RC.PICK_ORIENTATION_DEG[2]
        next_seeds = [cube_plan["q"]] if cube_plan else seeds

        if dest[0] == "hand":
            dpx, dpy, dpz = (float(v) for v in RC.DROP_POS_CM)
            dest_plan, why = RC.plan_orientation(lambda o: [("drop", dpx, dpy, dpz)], yaw,
                                                 next_seeds, (dpx, dpy), "DROP_POS_CM")
            if why:
                return why
            ox, oy = RC.grasp_offset_xy(dest_plan["orient"])
            self.drop_cube = (dpx - ox, dpy - oy,
                              dpz - RC.GRASP_Z_OFFSET_CM - V.CUBE_SIDE_CM / 2.0)
        else:
            _, tx, ty = dest

            def dest_points(orient):
                dx, dy = RC.grasp_offset_xy(orient)   # the CUBE, not the tip, lands on the spot
                return [("above spot", tx + dx, ty + dy, carry_z),
                        ("place", tx + dx, ty + dy, place_z)]

            dest_plan, why = RC.plan_orientation(dest_points, yaw, next_seeds, (tx, ty),
                                                 f"the spot ({tx:.1f}, {ty:.1f})")
            if why:
                return why
            _, px, py, _ = dest_points(dest_plan["orient"])[0]
        if seed is None:
            seed = dest_plan["seed"]

        steps = []
        if not self.holding:
            o = cube_plan["orient"]
            steps += [("rise", lambda: self._rise(hover_z, o)),
                      ("open gripper", lambda: self._grip(RC.GRIP_OPEN_DEG)),
                      ("above cube", lambda: self._move(gx, gy, hover_z, o)),
                      ("descend", lambda: self._move(gx, gy, grasp_z, o, APPROACH_SPEED_CM_S)),
                      ("grasp", self._grasp),
                      ("lift", lambda: self._move(gx, gy, carry_z, o))]
        else:
            steps.append(("rise", lambda: self._rise(carry_z, dest_plan["orient"])))
        po = dest_plan["orient"]
        if dest[0] == "hand":
            # step names match RC.HandWatch.WATCH_STEPS
            steps += [("drop pos", lambda: self._move(dpx, dpy, dpz, po)),
                      ("wait for hand", self._wait_hand),
                      ("release", self._release),
                      ("home", self._home)]
        else:
            steps += [("above spot", lambda: self._move(px, py, carry_z, po)),
                      ("lower", lambda: self._move(px, py, place_z, po, APPROACH_SPEED_CM_S)),
                      ("release", self._release),
                      ("retreat", lambda: self._move(px, py, carry_z, po)),
                      ("home", self._home)]
        if seed is home_q or (seed is None and not down):
            print("[run] starting from HOME ("
                  + ("gripper not pointing down" if not down else "no plan from the current pose")
                  + ")")
            steps.insert(0, ("home", self._home))

        steps = [(n, _timed(n, fn)) for n, fn in steps]
        self.conn.abort.clear()
        self.status, self.step = "RUNNING", ""
        print(f"[run] {'carry' if self.holding else 'pick'} -> {dest}; planned in "
              f"{time.monotonic() - t_plan:.2f} s; steps: {', '.join(n for n, _ in steps)}")
        self._thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
        self._thread.start()
        return None


# =============================================================================
# THE EXCHANGE
# =============================================================================

class Exchange:
    """Turn-taking state machine, ticked once per camera frame by the main loop.

    phase: IDLE -> WAIT (-> COUNTDOWN, participant turns) -> RUNNING -> WAIT ...
           -> FINISHED; PAUSED / FAILED until SPACE."""

    def __init__(self, runner: DemoRunner, on_hand: bool):
        self.runner = runner
        self.on_hand = on_hand
        self.turns = build_turns()
        self.i = 0
        self.done = set()
        self.phase = "IDLE"
        self.active = None               # cube of the current run / countdown
        self.countdown = None            # (cid, t0, center) on a participant turn
        self.info = "press SPACE to start the exchange"

    @property
    def turn(self):
        return self.turns[self.i] if self.i < len(self.turns) else None

    def turn_label(self):
        if self.turn is None:
            return f"all {len(self.turns)} turns done"
        side, cid = self.turn
        who = "ROBOT" if side == "robot" else "PARTICIPANT"
        return f"turn {self.i + 1}/{len(self.turns)}: {who}" + (f" (cube {cid})" if cid is not None
                                                                else "")

    def pending_human(self):
        return [c for c in HUMAN_CUBE_IDS if c not in self.done]

    def dest_for(self, side, cid):
        if side == "robot":
            return ("hand",) if self.on_hand else ("table",) + area_center()
        return ("table",) + tuple(putaway_of(cid))

    # -- transitions ----------------------------------------------------------

    def start(self):
        self.phase = "WAIT" if self.turn is not None else "FINISHED"
        self.countdown = None

    def pause(self):
        if self.runner.running:
            self.runner.stop()              # tick() turns RUNNING into PAUSED
        elif self.phase in ("WAIT", "COUNTDOWN"):
            self.phase, self.countdown = "PAUSED", None
            self.info = "paused -- SPACE to resume"

    def resume(self, scene):
        """Re-plan the current turn. Holding a cube -> carry it on at once."""
        if self.runner.holding and self.active is not None:
            side = "robot" if self.active in ROBOT_CUBE_IDS else "human"
            self._go(None, 0.0, self.dest_for(side, self.active), scene)
        else:
            self.start()

    def skip(self):
        if self.turn is None:
            return
        side, cid = self.turn
        print(f"[demo] turn {self.i + 1} skipped by hand")
        if cid is not None:
            self.done.add(cid)
        self._advance()

    def _advance(self):
        self.i += 1
        self.active, self.countdown = None, None
        self.phase = "WAIT" if self.turn is not None else "FINISHED"
        if self.phase == "FINISHED":
            self.info = "exchange finished"
            print("[demo] FINISHED")

    def _go(self, center, yaw, dest, scene):
        err = self.runner.start_transfer(center, yaw, dest, scene.table_z)
        if err:
            self.phase = "FAILED"
            self.runner.status, self.runner.message = "FAILED", f"refused: {err}"
            self.info = "fix it, then SPACE to retry (N skips this turn)"
            print(f"[demo] refused: {err}")
        else:
            self.phase = "RUNNING"
            self.info = (f"moving cube {self.active} -> "
                         + ("the hand" if dest[0] == "hand" else f"({dest[1]:.1f}, {dest[2]:.1f})"))

    # -- per frame --------------------------------------------------------------

    def tick(self, scene, now):
        r = self.runner
        if self.phase == "RUNNING" and not r.running:
            if r.status == "DONE":
                print(f"[demo] {self.turn_label()} done (cube {self.active})")
                self.done.add(self.active)
                self._advance()
            elif r.status == "STOPPED":
                self.phase, self.info = "PAUSED", "stopped -- SPACE to resume"
            else:
                self.phase = "FAILED"
                self.info = "SPACE to retry this turn (N skips it)"
        if self.phase not in ("WAIT", "COUNTDOWN") or r.running:
            return
        if scene.T_robot_cam is None:
            self.info = "BASE marker not seen"
            return
        side, cid = self.turn
        if side == "robot":
            self._robot_wait(scene, cid)
        else:
            self._human_wait(scene, now)

    def _robot_wait(self, scene, cid):
        self.active = cid
        s = scene.stable(cid)
        if s is None:
            self.info = f"waiting to see robot cube {cid} steady"
            return
        if not self.on_hand:
            blocking = [c for c, e in scene.cubes.items() if c != cid and in_area(e["center"])]
            if blocking:
                self.info = ("waiting for the Target Area to be cleared (cube "
                             + ", ".join(map(str, sorted(blocking))) + ")")
                return
        self._go(s[0], s[1], self.dest_for("robot", cid), scene)

    def _human_wait(self, scene, now):
        cd = self.countdown
        if cd is not None:
            cid, t0, c0 = cd
            s = scene.stable(cid)
            if (s is None or not in_area(s[0])
                    or np.linalg.norm(s[0] - c0) > RC.STABLE_TOL_CM):
                self.countdown, self.phase = None, "WAIT"
                return
            left = PICKUP_DELAY_S - (now - t0)
            if left > 0.0:
                self.info = f"picking cube {cid} in {left:.1f} s"
                return
            self.active = cid
            self._go(s[0], s[1], self.dest_for("human", cid), scene)
            return
        for cid in self.pending_human():
            s = scene.stable(cid)
            if s is not None and in_area(s[0]):
                self.countdown = (cid, now, s[0].copy())
                self.phase = "COUNTDOWN"
                self.active = cid
                self.info = f"cube {cid} in the Target Area"
                return
        self.active = None
        self.info = ("waiting for a participant's cube ("
                     + ", ".join(map(str, self.pending_human())) + ") in the Target Area")


# =============================================================================
# OVERLAY
# =============================================================================

PHASE_COLORS = {"IDLE": (255, 255, 255), "WAIT": (255, 255, 255), "COUNTDOWN": (255, 200, 0),
                "RUNNING": (0, 220, 255), "PAUSED": (0, 140, 255), "FAILED": (0, 0, 255),
                "FINISHED": (0, 220, 0)}


def draw_area(frame, T, K, dist, z, color):
    x0, x1, y0, y1 = area_bounds()
    rect = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    px = V.project_robot_points(rect, T, K, dist)
    cv2.polylines(frame, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], True, color, 2,
                  cv2.LINE_AA)
    V.put_text(frame, "target area", V._pt(px[0] + [6, -6]), color)


def draw_overlay(frame, scene: DemoScene, ex: Exchange, runner: DemoRunner, tip, fps,
                 hands=None):
    K, dist = scene.K, scene.dist
    if scene.all_markers:
        sightings = [(i, c) for i, cs in scene.all_markers.items() for c in cs]
        ids = np.array([i for i, _ in sightings], dtype=np.int32).reshape(-1, 1)
        corners = [c.reshape(1, 4, 2).astype(np.float32) for _, c in sightings]
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    y = 25
    def line(text, color=(255, 255, 255)):
        nonlocal y
        V.put_text(frame, text, (10, y), color)
        y += 24

    T = scene.T_robot_cam
    if T is None:
        line(f"BASE marker (id {V.BASE_ID}) not seen -- no robot frame yet", (0, 0, 255))
    else:
        V.draw_robot_axes(frame, T, K, dist)
        z = scene.table_z
        if SHOW_REACH_CIRCLE:
            outside = any(math.hypot(c["center"][0], c["center"][1]) > RC.MAX_DOWN_REACH_CM
                          for c in scene.cubes.values())
            RC.draw_reach_circle(frame, T, K, dist, z, RC.MAX_DOWN_REACH_CM,
                                 COL_OUT if outside else COL_IN)
        inside = any(cid in scene.cubes and in_area(scene.cubes[cid]["center"])
                     for cid in ex.pending_human())
        draw_area(frame, T, K, dist, z, COL_IN if inside else COL_OUT)
        for cid in HUMAN_CUBE_IDS:
            x, yy = putaway_of(cid)
            V.draw_point(frame, [x, yy, z], T, K, dist, COL_HUMAN_CUBE, f"slot {cid}", radius=3)
        for cid, c in sorted(scene.cubes.items()):
            if cid == ex.active:
                col = COL_ACTIVE_CUBE
            elif cid in ex.done:
                col = COL_DONE_CUBE
            else:
                col = COL_ROBOT_CUBE if cid in ROBOT_CUBE_IDS else COL_HUMAN_CUBE
            V.draw_cube_box(frame, c["center"], c["yaw"], V.CUBE_SIDE_CM, T, K, dist, col)
            V.draw_point(frame, c["center"], T, K, dist, col, f"{cid} ({owner(cid)})", radius=2)
        if tip is not None:
            V.draw_point(frame, tip[:3], T, K, dist, (255, 0, 200), "tip (FK)", radius=5)

    col = PHASE_COLORS.get(ex.phase, (255, 255, 255))
    line(f"{ex.phase}   {ex.turn_label()}", col)
    line(ex.info, col)
    left_r = [c for c in sorted(ROBOT_CUBE_IDS) if c not in ex.done]
    line(f"exchanged {len(ex.done)}/{len(ALL_CUBE_IDS)}   robot left: {left_r or '-'}   "
         f"participant left: {ex.pending_human() or '-'}")
    rcol = RC.STATUS_COLORS.get(runner.status, (255, 255, 255))
    step = f" [{runner.step}]" if runner.running and runner.step else ""
    line(f"arm: {runner.status}{step}   holding: {'YES' if runner.holding else 'no'}", rcol)
    if runner.status in ("FAILED", "STOPPED"):
        line(runner.message, rcol)
    if hands is not None and runner.running and runner.step in RC.HandWatch.WATCH_STEPS:
        hands.draw(frame, scene, runner.drop_cube)
        line(hands.info, (0, 220, 0) if hands.ready else (0, 140, 255))
    V.put_text(frame, f"{fps:4.1f} fps", (10, frame.shape[0] - 12))


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Robot <-> participant cube exchange (linear moves).")
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--mock", action="store_true", help="simulated arm (the camera is still real)")
    V.add_camera_args(ap)
    ap.add_argument("--calib", default=None,
                    help="camera .npz (calibrate_camera.py); default: RealSense factory intrinsics")
    ap.add_argument("--base-calib", default=V.BASE_CALIB_PATH,
                    help="teach-pairs correction from calibrate_base.py (skipped if missing)")
    ap.add_argument("--cube-marker-cm", type=float, default=V.CUBE_MARKER_CM,
                    help="measured black square of a cube-face marker")
    ap.add_argument("--big-marker-cm", type=float, default=V.BIG_MARKER_CM,
                    help="measured black square of the BASE marker")
    ap.add_argument("--on-hand", action="store_true",
                    help="robot turns: hand the cube over at DROP_POS_CM instead of the Target Area")
    ap.add_argument("--hand-model", default=None,
                    help="MediaPipe Tasks hand_landmarker.task (only if mp.solutions is missing)")
    args = ap.parse_args()
    args.cube_z = "table"            # cubes rest on the table; depth is only for the hand
    args.cube_id = None              # RC.Scene's single tracked cube: unused here
    if args.on_hand and args.source != "realsense":
        ap.error("--on-hand needs the RealSense depth (--source realsense)")

    calib = V.load_base_calib(args.base_calib)
    print(f"base correction: {calib}"
          + ("" if _os.path.exists(args.base_calib or "") else "  (none -- run calibrate_base.py)"))
    errors, warnings = check_config(V.table_z(calib), args.on_hand)
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"CONFIG ERROR: {e}", file=_sys.stderr)
        return 2
    turns = build_turns()
    print("turns: " + "  ".join(f"R{cid}" if side == "robot" else "P" for side, cid in turns))

    cam = V.open_camera(args.source, args.camera, args.width, args.height, depth=args.on_hand)
    conn = RC.AbortableConnection(args.port, mock=args.mock)
    arm = Arm(connection=conn)
    arm.set_single_joint(0)                   # streamed moves only: those are what SPACE can stop
    runner = DemoRunner(arm, conn, on_hand=args.on_hand)
    ex = Exchange(runner, args.on_hand)
    hands = (RC.HandWatch(H.HandTracker(args.hand_model or H.HAND_MODEL_PATH))
             if args.on_hand else None)

    scene = None
    tip, t_tip = None, 0.0
    fps, t_prev = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Camera frame grab failed", file=_sys.stderr)
                return 1
            if scene is None:
                h, w = frame.shape[:2]
                K, dist, how = V.camera_model(cam, w, h, args.calib)
                print(f"camera {cam.name}: {w}x{h}, {how}")
                scene = DemoScene(K, dist, calib, args, point_fn=cam.point_at,
                                  on_hand=args.on_hand)

            moving = runner.running
            scene.update(frame, frozen=moving)
            now = time.monotonic()
            ex.tick(scene, now)

            if runner.running:
                tip = None
            elif now - t_tip > RC.TIP_POLL_S:
                try:
                    tip = arm.get_coords()
                except Exception as exc:            # a missed serial read: just skip the dot
                    print(f"tip read failed: {exc}")
                    tip = None
                t_tip = now

            if hands is not None:
                if runner.running and runner.step in RC.HandWatch.WATCH_STEPS and runner.drop_cube:
                    ready = hands.update(frame, scene, runner.drop_cube, now)
                else:
                    hands.reset()
                    ready = False
                (runner.hand_ready.set if ready else runner.hand_ready.clear)()

            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            draw_overlay(frame, scene, ex, runner, tip, fps, hands)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if runner.running or ex.phase in ("WAIT", "COUNTDOWN"):
                    print("[key] SPACE -> pause")
                    ex.pause()
                elif ex.phase == "IDLE":
                    print("[key] SPACE -> start")
                    ex.start()
                elif ex.phase in ("PAUSED", "FAILED"):
                    print("[key] SPACE -> resume")
                    ex.resume(scene)
            elif runner.running:
                continue                              # the keys below only while the arm is still
            elif key == ord("n"):
                ex.skip()
            elif key == ord("g"):
                runner.holding = not runner.holding
                ex.info = f"holding set to {runner.holding} by hand"
            elif key == ord("o"):
                arm.send_gripper(RC.GRIP_OPEN_DEG)
                runner.holding = False
                ex.info = "gripper opened"
            elif key == ord("h"):
                if ex.phase in ("WAIT", "COUNTDOWN", "FAILED"):
                    ex.phase, ex.countdown = "PAUSED", None
                runner.start_home()
                ex.info = "going home -- SPACE to resume" if ex.phase == "PAUSED" else "going home"
            elif key == ord("b"):
                scene.relock()
                ex.info = "re-acquiring the base"
    finally:
        runner.stop()
        runner.join(timeout=3.0)
        if hands is not None:
            hands.tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        arm.close()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
