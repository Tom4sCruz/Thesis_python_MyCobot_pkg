"""
armik -- Cartesian control for the myCobot 280, built on pymycobot.

pymycobot is a serial protocol codec: it formats byte packets and decodes
replies. It does no kinematics. Every FK/IK operation it appears to offer
(get_coords, send_coords, solve_inv_kinematics) is forwarded to firmware you
cannot inspect, constrain, or debug.

This package puts the kinematics back in Python, which buys three things the
firmware path cannot give you:

  * PARTIAL CONSTRAINTS -- constrain any subset of x, y, z, rx, ry, rz. The
    rest are free, but held as close to where they started as possible.
  * STRAIGHT-LINE MOTION -- IK is re-solved at every setpoint along the line.
  * PLAN-THEN-EXECUTE -- the entire path is validated before the first byte is
    sent, so an unreachable target moves the arm zero millimetres instead of
    stranding it mid-path.

Motion is issued only through pymycobot's send_angles()/send_angle().

Quick start
-----------
    from armik import Arm

    with Arm(port="/dev/ttyTHS1") as arm:
        if not arm.conn.is_power_on():
            arm.conn.power_on()

        print(arm.get_coords())                     # cm and degrees
        arm.send_coords(x=20, y=-6, z=15, speed=4)  # 1 on success, 0 on failure
        arm.send_coords(z=12)                        # z only, rest held steady

Offline, with no hardware:
    with Arm(mock=True) as arm:
        ...
"""

from . import config, ik, kinematics
from .arm import Arm, Execution, Plan, FAILURE, SUCCESS
from .config import JOINT_1_HEIGHT_CM
from .connection import ArmConnection, ArmError, dps_to_firmware_speed
from .ik import IKFailure
from .kinematics import (
    forward_kinematics,
    geometric_jacobian,
    manipulability,
    matrix_to_rpy,
    pose_coords,
    rpy_to_matrix,
    task_jacobian,
)

__version__ = "1.0.0"

__all__ = [
    "Arm",
    "ArmConnection",
    "ArmError",
    "IKFailure",
    "Plan",
    "Execution",
    "SUCCESS",
    "FAILURE",
    "JOINT_1_HEIGHT_CM",
    "config",
    "ik",
    "kinematics",
    "forward_kinematics",
    "geometric_jacobian",
    "task_jacobian",
    "pose_coords",
    "manipulability",
    "rpy_to_matrix",
    "matrix_to_rpy",
    "dps_to_firmware_speed",
]
