"""
Inverse kinematics with PARTIAL pose constraints.

This is what firmware IK fundamentally cannot do. send_coords() on the firmware
demands a full 6-DOF pose: all of x, y, z, rx, ry, rz, every time. If you only
care about three of those numbers, you still have to invent the other three,
and any pose you invent may be unreachable -- which is a large part of why
send_coords() feels finicky.

Here, each of the six task coordinates is independently either:

  CONSTRAINED -- you gave a value; it is driven to that value at full weight.
  FREE        -- you passed None; it is SOFT-ANCHORED to the value it had when
                 the motion started, at config.FREE_ANCHOR_WEIGHT.

The soft anchor is the whole mechanism behind "unmentioned coordinates should
change as little as possible". A free coordinate is not "don't care" -- it is
"prefer to stay put, but yield if the constrained coordinates need you to". A
weight of 0 would make it truly don't-care and let it wander.

Method: damped least squares (Levenberg-Marquardt) on the task Jacobian, with
per-row weights, joint-limit clamping, and a per-iteration step clamp. Damping
keeps it stable near singularities where an undamped pseudo-inverse would
demand enormous joint velocities.
"""

from __future__ import annotations

import numpy as np

from . import config
from .kinematics import coord_error, pose_coords, task_jacobian


class IKFailure(RuntimeError):
    """The constrained coordinates could not be satisfied."""


def solve(
    desired: np.ndarray,
    mask: np.ndarray,
    q_seed: np.ndarray,
    max_iters: int | None = None,
) -> np.ndarray:
    """
    Solve for joint angles satisfying the constrained task coordinates.

    Parameters
    ----------
    desired : (6,) target task coordinates [x, y, z, rx, ry, rz] in mm/deg.
        Entries where mask is False should hold the ANCHOR value (normally the
        coordinate's value at the start of the motion), not a target.
    mask : (6,) bool. True = constrained, False = free/soft-anchored.
    q_seed : (6,) starting joint angles in degrees. Seeding from the arm's
        current configuration is what keeps the solver in the same IK branch
        (elbow up/down, wrist flipped or not) instead of returning a
        mathematically valid solution that requires swinging halfway across
        the workspace.

    Returns
    -------
    (6,) joint angles in degrees.

    Raises
    ------
    IKFailure if the constrained coordinates cannot be reached.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise IKFailure("no coordinates constrained; nothing to solve for")

    limits = config.joint_limits_array()
    q = np.array(q_seed, dtype=float)

    # Full weight on what the caller asked for, soft anchor on the rest.
    w = np.where(mask, 1.0, config.FREE_ANCHOR_WEIGHT)
    w[3:] = w[3:] * config.ROT_WEIGHT_MM_PER_DEG

    lam2 = config.IK_DAMPING ** 2
    iters = max_iters or config.IK_MAX_ITERS

    best_q = q.copy()
    best_cost = np.inf

    for _ in range(iters):
        current = pose_coords(q)
        e = coord_error(current, desired)

        # Convergence is judged ONLY on the constrained coordinates. A free
        # coordinate sitting 4 mm from its anchor is not a failure.
        pos_ok = rot_ok = True
        cost = 0.0
        if mask[:3].any():
            pe = np.max(np.abs(e[:3][mask[:3]]))
            pos_ok = pe < config.POS_TOL_MM
            cost += pe
        if mask[3:].any():
            re = np.max(np.abs(e[3:][mask[3:]]))
            rot_ok = re < config.ROT_TOL_DEG
            cost += re
        if cost < best_cost:
            best_cost, best_q = cost, q.copy()
        if pos_ok and rot_ok:
            return q

        J = task_jacobian(q)
        Jw = J * w[:, None]
        ew = e * w

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam2 * np.eye(6), ew)

        biggest = np.max(np.abs(dq))
        if biggest > config.IK_STEP_CLAMP_DEG:
            dq *= config.IK_STEP_CLAMP_DEG / biggest

        q = np.clip(q + dq, limits[:, 0], limits[:, 1])

    names = [n for n, m in zip(config.COORD_NAMES, mask) if m]
    raise IKFailure(
        f"did not converge on {', '.join(names)} "
        f"(best residual {best_cost:.2f} in mm/deg)"
    )


def try_solve(desired, mask, q_seed, max_iters=None):
    """solve() but returns None instead of raising. Convenient in path loops."""
    try:
        return solve(desired, mask, q_seed, max_iters)
    except IKFailure:
        return None
