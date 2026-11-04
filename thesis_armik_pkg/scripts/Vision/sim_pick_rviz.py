#!/usr/bin/env python3
"""
VISION, NO ROBOT: replay the pick-and-place in RViz from a captured scene
=========================================================================

Loads the scene.json written by capture_scene.py and runs reach_cube.py's
own planner and state machine (Runner) on a SIMULATED arm, streaming it to
RViz through scripts/Profiles/_rviz_bridge.py: the robot model, the planned
path, the target zone (magenta disc), the ~28 cm reach ring and the cube
(orange), which rides along in the gripper between grasp and release.

Run inside the Study-docker container (needs rclpy), with RViz already up:

    ros2 launch /thesis/thesis_armik_pkg/ros/launch/mock_display.launch.py
    # second shell:
    cd /thesis/thesis_armik_pkg && python3 scripts/Vision/sim_pick_rviz.py
    python3 scripts/Vision/sim_pick_rviz.py --scene lab1.json --no-place
    python3 scripts/Vision/sim_pick_rviz.py --no-rviz      # console only (also on the host)

KEYS (typed in this terminal -- no Enter needed)
    SPACE   start / stop / resume, exactly as in reach_cube.py
    R       reset: cube back to its captured pose, arm at HOME, gripper open
    W / S   move the cube +1 / -1 cm along robot x  (idle, not holding)
    A / D   move the cube +1 / -1 cm along robot y
    Z / X   rotate the cube +5 / -5 deg
    H       home        O  open gripper        G  toggle "holding"
    Q       quit

The simulated arm moves in real time (the mock tracks every setpoint
instantly, the stream keeps the real schedule), so a run takes as long as it
would on the robot.
"""

from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE)))          # armik
_sys.path.insert(0, _os.path.join(_os.path.dirname(_HERE), "Profiles"))  # _rviz_bridge

import argparse
import json
import math
import select
import time

import numpy as np

import _vision as V
import reach_cube as RC
from armik import Arm


# =============================================================================
# CONSTANTS
# =============================================================================

SCENE_PATH = _os.path.join(_HERE, "scene.json")
MOVE_STEP_CM = 1.0               # W/A/S/D cube nudge
ROTATE_STEP_DEG = 5.0            # Z/X cube rotation
ATTACH_SNAP_CM = 3.0             # G with the cube further than this from the tip: snap it under the tip
TICK_S = 0.02                    # main loop period (keys, cube follow, status)


# =============================================================================
# THE SIMULATED CUBE
# =============================================================================

class SimCube:
    """Cube pose in the robot frame (cm, deg). Rests where it is; while the
    Runner is holding it, it keeps the offset to the tip it had at the grasp."""

    def __init__(self, center, yaw, table_z):
        self.home_center = np.array(center, dtype=float)
        self.home_yaw = float(yaw)
        self.table_z = float(table_z)
        self.reset()

    def reset(self):
        self.center = self.home_center.copy()
        self.yaw = self.home_yaw
        self._attached = None            # (offset xyz, yaw offset) while held

    def update(self, holding, tip):
        """tip = arm.get_coords(). Call often (every tick)."""
        if holding and tip is not None:
            t = np.array(tip[:3], dtype=float)
            if self._attached is None:
                if np.linalg.norm(self.center - t) > ATTACH_SNAP_CM:
                    # "holding" set by hand away from the cube: put it in the gripper
                    dx, dy = RC.grasp_offset_xy(tip[3:6])
                    self.center = t - np.array([dx, dy, RC.GRASP_Z_OFFSET_CM])
                self._attached = (self.center - t, self.yaw - float(tip[5]))
            off, yaw_off = self._attached
            self.center = t + off
            self.yaw = float(tip[5]) + yaw_off
        elif self._attached is not None:
            self._attached = None        # let go: it lands on the table where it is
            self.center[2] = self.table_z + V.CUBE_SIDE_CM / 2.0

    def pose(self):
        return (float(self.center[0]), float(self.center[1]), float(self.center[2]), self.yaw)


# =============================================================================
# SIMULATION
# =============================================================================

class Sim:
    def __init__(self, scene, no_place=False, bridge_factory=None):
        self.scene = scene
        self.table_z = float(scene["table_z"])
        self.target_xy = np.array(scene["target"]["xy"], dtype=float)
        self.cube = SimCube(scene["cube"]["center"], scene["cube"]["yaw"], self.table_z)
        self.conn = RC.AbortableConnection("mock", mock=True, quiet_mock=True)
        self.arm = Arm(connection=self.conn)
        self.arm.set_single_joint(0)
        self.arm.send_gripper(RC.GRIP_OPEN_DEG)
        self.runner = RC.Runner(self.arm, self.conn, no_place)
        self.bridge = bridge_factory(self) if bridge_factory else None
        self._last_status = None

    # -- per tick -------------------------------------------------------------

    def tick(self):
        try:
            tip = self.arm.get_coords()
        except Exception:                # noqa: BLE001 -- a mock read can't really fail
            tip = None
        self.cube.update(self.runner.holding, tip)
        r = self.runner
        status = (r.status, r.step if r.running else "", r.holding)
        if status != self._last_status:
            self._last_status = status
            step = f" [{status[1]}]" if status[1] else ""
            print(f"  {r.status}{step}  holding={'YES' if r.holding else 'no'}  -- {r.message}")

    # -- keys -------------------------------------------------------------------

    def key(self, ch):
        """Handle one key; returns False to quit."""
        r = self.runner
        ch = ch.lower()
        if ch == "q":
            return False
        if ch == " ":
            if r.running:
                print("[key] SPACE -> stop")
                r.stop()
                return True
            x, y, z, yaw = self.cube.pose()
            err = r.start_pick(np.array([x, y, z]), yaw, self.target_xy, self.table_z)
            if err:
                r.status, r.message = "FAILED", f"refused: {err}"
                print(f"[run] refused: {err}")
            elif self.bridge is not None:
                self.bridge.update_path(r.planned_points)
            return True
        if r.running:
            print("  (busy -- SPACE stops the run first)")
            return True
        if ch == "r":
            self.cube.reset()
            r.holding = False
            r.status, r.message = "IDLE", "reset -- press SPACE to pick the cube"
            self.conn._mc.set_angles_directly(RC.HOME)
            self.arm.send_gripper(RC.GRIP_OPEN_DEG)
            if self.bridge is not None:
                self.bridge.update_path([])
            self._report_cube("reset")
        elif ch in "wsadzx":
            if r.holding:
                print("  (holding the cube -- G or R first)")
                return True
            dx = {"w": MOVE_STEP_CM, "s": -MOVE_STEP_CM}.get(ch, 0.0)
            dy = {"a": MOVE_STEP_CM, "d": -MOVE_STEP_CM}.get(ch, 0.0)
            dyaw = {"z": ROTATE_STEP_DEG, "x": -ROTATE_STEP_DEG}.get(ch, 0.0)
            self.cube.center += [dx, dy, 0.0]
            self.cube.yaw = V.wrap90(self.cube.yaw + dyaw)
            self._report_cube("moved")
        elif ch == "h":
            r.start_home()
        elif ch == "o":
            self.arm.send_gripper(RC.GRIP_OPEN_DEG)
            r.holding = False
            r.message = "gripper opened"
        elif ch == "g":
            r.holding = not r.holding
            r.message = f"holding set to {r.holding} by hand"
        return True

    def _report_cube(self, what):
        x, y, z, yaw = self.cube.pose()
        d = math.hypot(x, y)
        ok = ("OK" if d <= RC.MAX_DOWN_REACH_CM else "OK, tilted grasp" if d <= RC.MAX_TILTED_REACH_CM
              else f"out of reach (> ~{RC.MAX_TILTED_REACH_CM:.0f} cm)")
        print(f"  cube {what}: ({x:.1f}, {y:.1f}, {z:.1f}) cm  yaw {yaw:.0f}  "
              f"-> {d:.1f} cm from the base axis, {ok}")

    def close(self):
        self.runner.stop()
        self.runner.join(timeout=3.0)
        if self.bridge is not None:
            self.bridge.stop()
        self.arm.close()


def make_bridge(sim):
    """RvizBridge wired to the simulation; None (console only) if rclpy is missing."""
    from _rviz_bridge import RvizBridge
    tx, ty = sim.target_xy
    bridge = RvizBridge(
        joint_source=sim.arm.get_angles,
        tip_source=sim.arm.get_coords,
        gripper_source=sim.arm.get_gripper_value,
        target_zone=(tx, ty, sim.table_z, sim.scene.get("marker_cm", {}).get("big", V.BIG_MARKER_CM)),
        reach_ring=(RC.MAX_DOWN_REACH_CM, sim.table_z),
        live_cube_source=sim.cube.pose,
        node_name="thesis_vision_sim",
    )
    try:
        bridge.start()
    except RuntimeError as exc:
        print(f"RViz disabled: {exc}")
        return None
    return bridge


# =============================================================================
# MAIN
# =============================================================================

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


def main():
    ap = argparse.ArgumentParser(description="Replay the vision pick-and-place in RViz from scene.json")
    ap.add_argument("--scene", default=SCENE_PATH, help="JSON from capture_scene.py")
    ap.add_argument("--no-place", action="store_true", help="stop after grasping and lifting")
    ap.add_argument("--no-rviz", action="store_true", help="console only (no ROS needed)")
    args = ap.parse_args()

    with open(args.scene) as fh:
        scene = json.load(fh)
    print(f"scene {args.scene} (captured {scene.get('created', '?')}, intrinsics: "
          f"{scene.get('intrinsics', '?')})")
    sim = Sim(scene, args.no_place, None if args.no_rviz else make_bridge)
    sim._report_cube("captured")
    tx, ty = sim.target_xy
    print(f"  target ({tx:.1f}, {ty:.1f}) cm -> {math.hypot(tx, ty):.1f} cm from the base axis")
    print(__doc__.split("KEYS (typed in this terminal -- no Enter needed)")[1].split("The simulated")[0])

    tty_state = None
    if _sys.stdin.isatty():
        import termios
        import tty
        tty_state = termios.tcgetattr(_sys.stdin)
        tty.setcbreak(_sys.stdin.fileno())
    try:
        running = True
        while running:
            for ch in read_keys(TICK_S):
                if not sim.key(ch):
                    running = False
                    break
            sim.tick()
    except KeyboardInterrupt:
        pass
    finally:
        if tty_state is not None:
            import termios
            termios.tcsetattr(_sys.stdin, termios.TCSADRAIN, tty_state)
        sim.close()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
