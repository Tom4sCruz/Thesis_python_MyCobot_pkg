"""
A stand-in for pymycobot's MyCobot280, so the whole package can be developed,
unit-tested and demonstrated with no hardware attached.

It models perfect, instantaneous tracking: whatever you send becomes the
current state immediately. That is deliberately optimistic -- it validates the
maths and the control flow, NOT the dynamics. A trajectory that passes in mock
can still be too fast for the real servos, which is why the speed precheck in
arm.py exists and runs regardless of backend.
"""

from __future__ import annotations

import time


class MockMyCobot:
    """Minimal API-compatible subset of pymycobot's MyCobot280."""

    def __init__(self, angles=None):
        self._angles = list(angles) if angles else [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
        self._powered = True
        self._fresh_mode = 0   # pymycobot's documented default: queue mode
        # (timestamp, angles, speed) for every command received
        self.commands: list[tuple[float, list[float], int]] = []
        # (timestamp, value, speed) for every gripper command received
        self.gripper_commands: list[tuple[float, int, int]] = []
        self._gripper_value = 0

    # -- state --------------------------------------------------------------

    def power_on(self):
        self._powered = True
        return 1

    def power_off(self):
        self._powered = False
        return 1

    def is_power_on(self):
        return 1 if self._powered else 0

    def set_fresh_mode(self, mode):
        self._fresh_mode = mode
        return 1

    def get_fresh_mode(self):
        return self._fresh_mode

    def get_angles(self):
        return list(self._angles)

    def get_coords(self):
        """Mimics the firmware's FK by using ours -- fine for flow testing,
        useless for validating the DH table (it would be circular). Uses the
        bare FLANGE, like the real firmware, which knows nothing about
        config.TOOL_OFFSET_MM."""
        from .kinematics import flange_pose_coords
        return [float(v) for v in flange_pose_coords(self._angles)]

    # -- motion -------------------------------------------------------------

    def send_angles(self, angles, speed):
        self.commands.append((time.perf_counter(), list(angles), speed))
        self._angles = [float(a) for a in angles]
        return 1

    def send_angle(self, joint_id, angle, speed):
        self.commands.append((time.perf_counter(), [joint_id, angle], speed))
        self._angles[joint_id - 1] = float(angle)
        return 1

    def set_gripper_value(self, value, speed, gripper_type=None, is_torque=None):
        self.gripper_commands.append((time.perf_counter(), int(value), int(speed)))
        self._gripper_value = int(value)
        return 1

    def get_gripper_value(self, gripper_type=None):
        return self._gripper_value

    def stop(self):
        return 1

    def is_moving(self):
        return 0

    def is_gripper_moving(self):
        return 0

    def release_all_servos(self):
        return 1

    # -- test helpers -------------------------------------------------------

    def set_angles_directly(self, angles):
        """Teleport the mock arm, bypassing the command log."""
        self._angles = [float(a) for a in angles]

    def reset_log(self):
        self.commands.clear()
        self.gripper_commands.clear()
