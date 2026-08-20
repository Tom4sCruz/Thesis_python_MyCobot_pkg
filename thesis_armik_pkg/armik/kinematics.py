"""
Forward kinematics and the task-space Jacobian.

pymycobot does no kinematics of its own -- it is a serial protocol codec, and
all FK/IK normally happens on firmware you cannot inspect or constrain. This
module is the replacement: it computes, in Python, where the tip is for a given
set of joint angles, and how the six task coordinates respond to joint motion.

UNITS: joint angles in DEGREES, positions in MILLIMETRES, orientations in
DEGREES. No radians escape this module.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def _dh_matrix(theta_rad: float, d: float, a: float, alpha_rad: float) -> np.ndarray:
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def frame_chain(q_deg: Sequence[float]) -> list[np.ndarray]:
    """[T_0_0, T_0_1, ..., T_0_6] -- seven 4x4 transforms."""
    q = np.asarray(q_deg, dtype=float)
    if q.shape != (config.DOF,):
        raise ValueError(f"expected {config.DOF} joint angles, got {q.shape}")
    frames = [np.eye(4)]
    T = np.eye(4)
    for i, (offset, d, a, alpha) in enumerate(config.DH_TABLE):
        T = T @ _dh_matrix(np.deg2rad(q[i] + offset), d, a, np.deg2rad(alpha))
        frames.append(T.copy())
    return frames


def forward_kinematics(q_deg: Sequence[float]) -> np.ndarray:
    """End-effector pose as a 4x4 homogeneous transform (translation in mm)."""
    return frame_chain(q_deg)[-1]


def geometric_jacobian(q_deg: Sequence[float]) -> np.ndarray:
    """
    Classic geometric Jacobian, (6, 6): linear velocity (mm/rad) stacked on
    angular velocity (rad/rad). Used for the singularity check, not for IK --
    IK uses the task Jacobian below so that individual Euler components can be
    constrained independently.
    """
    frames = frame_chain(q_deg)
    p_e = frames[-1][:3, 3]
    J = np.zeros((6, config.DOF))
    for i in range(config.DOF):
        z = frames[i][:3, 2]
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def manipulability(q_deg: Sequence[float]) -> float:
    """Yoshikawa index. Near zero means a singularity is nearby."""
    J = geometric_jacobian(q_deg)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


# ---------------------------------------------------------------------------
# Rotation <-> Euler
# ---------------------------------------------------------------------------

def rpy_to_matrix(rx_deg, ry_deg, rz_deg, convention: str | None = None) -> np.ndarray:
    convention = convention or config.EULER_CONVENTION
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    if convention == "xyz_fixed":
        return Rz @ Ry @ Rx
    if convention == "zyx_fixed":
        return Rx @ Ry @ Rz
    raise ValueError(f"unknown Euler convention: {convention}")


def matrix_to_rpy(R: np.ndarray, convention: str | None = None) -> np.ndarray:
    convention = convention or config.EULER_CONVENTION
    if convention == "xyz_fixed":
        sy = np.hypot(R[0, 0], R[1, 0])
        if sy < 1e-8:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0
        else:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
    elif convention == "zyx_fixed":
        sy = np.hypot(R[2, 2], R[1, 2])
        if sy < 1e-8:
            rx = 0.0
            ry = np.arctan2(R[0, 2], sy)
            rz = np.arctan2(-R[1, 0], R[1, 1])
        else:
            rz = np.arctan2(-R[0, 1], R[0, 0])
            ry = np.arctan2(R[0, 2], sy)
            rx = np.arctan2(-R[1, 2], R[2, 2])
    else:
        raise ValueError(f"unknown Euler convention: {convention}")
    return np.rad2deg([rx, ry, rz])


def pose_coords(q_deg: Sequence[float]) -> np.ndarray:
    """Joint angles -> [x, y, z, rx, ry, rz] in mm and degrees."""
    T = forward_kinematics(q_deg)
    return np.concatenate([T[:3, 3], matrix_to_rpy(T[:3, :3])])


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def wrap180(a):
    """Wrap degrees to [-180, 180). Scalar or array."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def coord_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """
    Error in the six task coordinates, mm and degrees.
    Rotation rows are wrapped, so 179 -> -179 is a 2 degree error, not 358.
    """
    e = np.empty(6)
    e[:3] = desired[:3] - current[:3]
    e[3:] = wrap180(desired[3:] - current[3:])
    return e


# ---------------------------------------------------------------------------
# Task Jacobian
# ---------------------------------------------------------------------------

def task_jacobian(q_deg: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """
    Numerical Jacobian of the TASK coordinates:
        d[x, y, z, rx, ry, rz] / d[q1..q6]   in mm/deg and deg/deg.

    We differentiate the Euler representation directly rather than using the
    geometric Jacobian, because that is what allows an arbitrary SUBSET of the
    six coordinates to be constrained independently -- constraining only rz,
    say, has no clean expression in terms of an angular-velocity twist.

    The price is that this representation degenerates near gimbal lock; see
    config.GIMBAL_WARN_DEG and the warning raised in arm.py.
    """
    base = pose_coords(q_deg)
    J = np.zeros((6, config.DOF))
    for i in range(config.DOF):
        qp = np.array(q_deg, dtype=float)
        qp[i] += h
        pert = pose_coords(qp)
        d = np.empty(6)
        d[:3] = pert[:3] - base[:3]
        d[3:] = wrap180(pert[3:] - base[3:])
        J[:, i] = d / h
    return J


def check_joint_limits(q_deg) -> list[str]:
    """Human-readable list of soft-limit violations (empty if fine)."""
    limits = config.joint_limits_array()
    out = []
    for i, a in enumerate(q_deg):
        lo, hi = limits[i]
        if not lo <= a <= hi:
            out.append(f"J{i+1}={a:.2f} outside [{lo:.1f}, {hi:.1f}]")
    return out
