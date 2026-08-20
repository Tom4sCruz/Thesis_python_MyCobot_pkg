"""
Multi-waypoint trajectory blending.

`Arm.send_coords()` is one complete, self-contained trajectory per call --
shaped by a minimum-jerk profile with ZERO velocity at both ends, every time.
That is exactly right for a single free-standing move, and exactly wrong for a
list of points you want the tip to sweep through continuously: calling
send_coords() once per point makes the arm decelerate to rest at every single
one, no matter how fast the points are sent or what pymycobot's fresh_mode is
set to (see the docstring in connection.py -- fresh_mode preempts, it doesn't
blend).

This module is what `Arm.send_path()` uses instead: given a list of joint-space
waypoints and a duration for each segment, it produces ONE continuous stream
where velocity is carried THROUGH interior waypoints rather than reset to zero
at each one. Segment speed is exactly (distance)/(duration) for that segment;
the *change* in speed from one segment to the next is the acceleration, and
it's controlled purely by how far apart the waypoints are and how much time
each segment is given.

Mechanism: cubic Hermite interpolation between consecutive waypoints, with the
tangent (velocity) AT each interior waypoint chosen by the standard via-point
heuristic -- average the two adjacent segment velocities when they point the
same way on a given joint, force zero when the path reverses direction on that
joint (otherwise the blend would overshoot past the waypoint before turning
back). This is the textbook technique for continuous-path (multipoint)
trajectories in robotics -- see e.g. Craig, "Introduction to Robotics", the
trajectory-generation chapter.

What this does NOT give you: acceleration continuity. The Hermite spline
matches position and velocity at every waypoint (C1) but not acceleration --
it can step there. In practice the tangent heuristic already smooths adjacent
segment speeds, so the step is small, but if a condition genuinely needs
bounded jerk through every waypoint, the next step up is a quintic blend per
segment (one more boundary condition per side) -- a direct extension of
hermite_sample below.
"""

from __future__ import annotations

import numpy as np


def waypoint_velocities(
    q_list: np.ndarray,
    durations: np.ndarray,
    v_start: np.ndarray | None = None,
    v_end: np.ndarray | None = None,
) -> np.ndarray:
    """
    q_list : (m+1, DOF) joint-space waypoints; q_list[0] is the starting pose.
    durations : (m,) seconds for each of the m segments.
    v_start / v_end : optional (DOF,) deg/s to carry into/out of this stretch,
        e.g. to chain two send_path() calls without a stop between them (pass
        the previous call's plan.exit_velocity). Default zero -- start and end
        at rest.

    Returns (m+1, DOF) deg/s: the velocity AT each waypoint, chosen so the
    blended path never has to instantaneously jump joint speed.
    """
    q_list = np.asarray(q_list, dtype=float)
    durations = np.asarray(durations, dtype=float)
    m, dof = durations.shape[0], q_list.shape[1]
    if q_list.shape[0] != m + 1:
        raise ValueError("need exactly one more waypoint than the number of durations")
    if np.any(durations <= 0):
        raise ValueError("all segment durations must be positive")

    seg_v = np.diff(q_list, axis=0) / durations[:, None]      # (m, DOF)
    v = np.zeros((m + 1, dof))
    v[0] = v_start if v_start is not None else 0.0
    v[-1] = v_end if v_end is not None else 0.0

    for k in range(1, m):
        before, after = seg_v[k - 1], seg_v[k]
        same_dir = np.sign(before) == np.sign(after)
        weighted = (durations[k] * before + durations[k - 1] * after) / (
            durations[k - 1] + durations[k]
        )
        # Clamp so the blend can't run faster than either neighbouring segment
        # was already moving -- keeps the Hermite curve from bulging past the
        # waypoints it's supposed to pass through.
        cap = np.minimum(np.abs(before), np.abs(after))
        weighted = np.clip(weighted, -cap, cap)
        v[k] = np.where(same_dir, weighted, 0.0)
    return v


def _hermite_basis(tau: np.ndarray):
    t2, t3 = tau ** 2, tau ** 3
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + tau
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00, h10, h01, h11


def hermite_sample(q0, v0, q1, v1, duration: float, n_samples: int) -> np.ndarray:
    """
    One segment, position and velocity matched at both ends.
    Returns (n_samples, DOF), tau sampled uniformly over [0, 1].
    """
    tau = np.linspace(0.0, 1.0, n_samples)
    h00, h10, h01, h11 = _hermite_basis(tau)
    m0, m1 = np.asarray(v0) * duration, np.asarray(v1) * duration
    return (
        h00[:, None] * q0 + h10[:, None] * m0
        + h01[:, None] * q1 + h11[:, None] * m1
    )


def hermite_velocity(q0, v0, q1, v1, duration: float, n_samples: int) -> np.ndarray:
    """Analytic derivative of hermite_sample, in deg/s -- exact, not finite-differenced."""
    tau = np.linspace(0.0, 1.0, n_samples)
    t2 = tau ** 2
    dh00 = 6 * t2 - 6 * tau
    dh10 = 3 * t2 - 4 * tau + 1
    dh01 = -6 * t2 + 6 * tau
    dh11 = 3 * t2 - 2 * tau
    m0, m1 = np.asarray(v0) * duration, np.asarray(v1) * duration
    d_dtau = (
        dh00[:, None] * q0 + dh10[:, None] * m0
        + dh01[:, None] * q1 + dh11[:, None] * m1
    )
    return d_dtau / duration    # chain rule: d/dt = (d/dtau) * (dtau/dt) = (d/dtau) / duration
