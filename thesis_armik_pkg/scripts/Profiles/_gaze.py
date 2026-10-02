"""
GAZE (look-at) orientation helper, shared by HighH-LowR.py / HighH-HighR.py.
=============================================================================

Lets the gripper tip's pointing axis track a 3D target point (a cube while
reaching for it, a drop point while carrying it there) instead of holding a
fixed PICK_ORIENTATION_DEG. Pure numpy; no new dependency.

GEOMETRY
--------
The tool tip's local +z axis is the pointing axis in world frame, independent
of config.TOOL_RPY_DEG (a pure z-roll, which doesn't move the z axis) -- see
the comment above TOOL_OFFSET_MM in armik/config.py. So "look at target T from
tip position P" means: build a rotation matrix whose 3rd column is
normalize(T - P), with the other two columns (right/up) derived from a
world-up reference, then convert to (rx, ry, rz) via kinematics.matrix_to_rpy.
This supersedes ORIENT_LOCK/_yaw() on a gazed arc -- the jaw heading is
whatever a natural "look at it" pose implies, not independently controlled.

EASE-IN
-------
A target SWITCH (not tip motion) should not snap the orientation instantly.
GazeEaser exponentially eases the tracked orientation toward the live
look-at pose each waypoint: alpha = 1 - exp(-dt / ease_in_s). ease_in_s = 0
means no smoothing (every waypoint's orientation IS that waypoint's pure
look-at pose); larger values lock on more slowly. The easing is done as a
rotation-matrix slerp (via quaternions), not per-axis Euler lerp, which would
reintroduce wraparound/gimbal artifacts from matrix_to_rpy's own branch cuts.

EASE-OUT
--------
Gazing all the way to the last waypoint would arrive at the cube/drop point
still tilted toward it, not gripper-straight-down. gaze_then_level_waypoints
blends the live gaze pose toward a fixed "level" (straight-down) pose over
the final ease_out_s seconds of the arc -- framed in TIME-TO-ARRIVAL, not
time-elapsed, since what matters is "how long before touchdown does it start
leveling out", independent of how long the arc as a whole takes:
beta = exp(-remaining_s / ease_out_s), so beta ~= 0 (pure gaze) while
remaining_s >> ease_out_s and beta -> 1 (pure level) as remaining_s -> 0,
landing exactly on the level pose at the arc's last waypoint. Blended via the
same quaternion slerp as ease-in, between the (already ease-in-eased) live
gaze matrix and the level matrix -- not a second independent easer.
"""

from __future__ import annotations

import math

import numpy as np

from armik import kinematics

_STRAIGHT_DOWN = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])


def _look_at_matrix(point_xyz, target_xyz, prev_matrix=None, up_hint=(0.0, 0.0, 1.0)):
    """3x3 rotation whose 3rd column is normalize(target - point). Falls back
    to prev_matrix (or a straight-down default, if there is none yet) when
    point == target -- the end of every reach/carry arc IS its own gaze
    target by construction, where the look-at direction is undefined --
    rather than normalizing a zero vector.

    The "up" reference used to derive roll is prev_matrix's OWN up axis when
    given, not a constant world vector -- a plain world-up look-at recomputes
    roll from scratch every call, which can swing it by 100+ deg even when the
    gaze direction itself barely tilts (whenever forward crosses near the
    world-up meridian), and that roll discontinuity is what the IK actually
    struggles to track, not the (much milder) change in gaze direction. Using
    the previous call's up keeps roll continuous/minimal-twist, at the cost of
    some roll "drift" over a long run of continuously changing directions --
    a non-issue for the short, few-second arcs this is used on. Falls back to
    world-up (then (1,0,0)) whenever a candidate is nearly parallel to
    forward (gaze near-vertical) or there is no previous frame yet."""
    point = np.asarray(point_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    diff = target - point
    norm = float(np.linalg.norm(diff))
    if norm < 1e-6:
        if prev_matrix is not None:
            return prev_matrix
        return _STRAIGHT_DOWN.copy()
    forward = diff / norm

    candidates = []
    if prev_matrix is not None:
        candidates.append(np.asarray(prev_matrix)[:, 1])
    candidates.append(np.asarray(up_hint, dtype=float))
    candidates.append(np.array([1.0, 0.0, 0.0]))
    candidates.append(np.array([0.0, 1.0, 0.0]))

    right = None
    for up in candidates:
        cand = np.cross(up, forward)
        rn = float(np.linalg.norm(cand))
        if rn > 1e-6:
            right = cand / rn
            break
    true_up = np.cross(forward, right)
    return np.column_stack([right, true_up, forward])


def look_at_rpy(point_xyz, target_xyz, prev_matrix=None) -> tuple[float, float, float]:
    """One-shot (rx, ry, rz) deg looking from point_xyz at target_xyz. No
    easing -- for static reachability probes (preflight), not a live path."""
    R = _look_at_matrix(point_xyz, target_xyz, prev_matrix=prev_matrix)
    rx, ry, rz = kinematics.matrix_to_rpy(R)
    return float(rx), float(ry), float(rz)


# ---------------------------------------------------------------------------
# Rotation-matrix slerp (via quaternions)
# ---------------------------------------------------------------------------

def _mat_to_quat(R):
    """3x3 rotation -> (w, x, y, z), Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _slerp_quat(q0, q1, t):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = math.acos(dot)
    theta = theta0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def _slerp_matrix(R0, R1, alpha):
    if alpha >= 1.0:
        return R1
    if alpha <= 0.0:
        return R0
    q = _slerp_quat(_mat_to_quat(R0), _mat_to_quat(R1), alpha)
    return _quat_to_mat(q)


# ---------------------------------------------------------------------------
# Stateful per-arc tracker
# ---------------------------------------------------------------------------

class GazeEaser:
    """Exponentially eases the tracked orientation toward the live look-at
    pose, so a target SWITCH (not tip motion) produces a smooth re-aim
    instead of a snap.

    ease_in_s: time constant (seconds). 0 -> no smoothing, every waypoint's
    orientation IS that waypoint's pure look-at pose. Larger -> slower
    lock-on onto a new target -- the dial the user controls.
    """

    def __init__(self, ease_in_s: float):
        self.tau = max(float(ease_in_s), 0.0)
        self._R = None

    def seed(self, rx: float, ry: float, rz: float) -> None:
        """Call once, from arm.get_coords()[3:], before the first step of an arc."""
        self._R = kinematics.rpy_to_matrix(rx, ry, rz)

    def step_matrix(self, point_xyz, target_xyz, dt: float) -> np.ndarray:
        R_goal = _look_at_matrix(point_xyz, target_xyz, prev_matrix=self._R)
        if self._R is None or self.tau <= 1e-9:
            self._R = R_goal
        else:
            alpha = 1.0 - math.exp(-max(dt, 0.0) / self.tau)
            self._R = _slerp_matrix(self._R, R_goal, alpha)
        return self._R

    def step(self, point_xyz, target_xyz, dt: float) -> tuple[float, float, float]:
        rx, ry, rz = kinematics.matrix_to_rpy(self.step_matrix(point_xyz, target_xyz, dt))
        return float(rx), float(ry), float(rz)


def gaze_waypoints(points_xyz, target_xyz, start_rpy, durations, ease_in_s):
    """Seed a GazeEaser from start_rpy (the arm's current orientation, e.g.
    arm.get_coords()[3:]) and step it once per point in points_xyz against
    the matching entry of durations. Returns three parallel lists
    (rx, ry, rz) ready for arm.send_path(rx=, ry=, rz=)."""
    easer = GazeEaser(ease_in_s)
    easer.seed(*start_rpy)
    rx_list, ry_list, rz_list = [], [], []
    last_dt = 0.0
    for i, p in enumerate(points_xyz):
        dt = float(durations[i]) if i < len(durations) else last_dt
        last_dt = dt
        rx, ry, rz = easer.step(p, target_xyz, dt)
        rx_list.append(rx)
        ry_list.append(ry)
        rz_list.append(rz)
    return rx_list, ry_list, rz_list


def gaze_then_level_waypoints(points_xyz, target_xyz, level_rpy_sequence,
                               start_rpy, durations, ease_in_s, ease_out_s):
    """Like gaze_waypoints, but blends back to level_rpy_sequence (the fixed
    straight-down pose for each of the same waypoints, e.g. _fixed_sequence())
    over the final ease_out_s seconds of the arc, so the LAST waypoint lands
    on level_rpy_sequence[-1] instead of the raw look-at pose. See the
    EASE-OUT note at the top of this file."""
    easer = GazeEaser(ease_in_s)
    easer.seed(*start_rpy)
    n = len(points_xyz)
    durs = [float(durations[i]) if i < len(durations)
            else (float(durations[-1]) if len(durations) else 0.0)
            for i in range(n)]
    remaining_s = [0.0] * n
    running = 0.0
    for i in range(n - 1, -1, -1):
        remaining_s[i] = running
        running += durs[i]
    tau_out = max(float(ease_out_s), 0.0)

    rx_list, ry_list, rz_list = [], [], []
    for i, p in enumerate(points_xyz):
        R_gaze = easer.step_matrix(p, target_xyz, durs[i])
        R_level = kinematics.rpy_to_matrix(*level_rpy_sequence[i])
        if tau_out <= 1e-9:
            beta = 1.0 if remaining_s[i] <= 1e-9 else 0.0
        else:
            beta = math.exp(-remaining_s[i] / tau_out)
        R = _slerp_matrix(R_gaze, R_level, beta)
        rx, ry, rz = kinematics.matrix_to_rpy(R)
        rx_list.append(float(rx))
        ry_list.append(float(ry))
        rz_list.append(float(rz))
    return rx_list, ry_list, rz_list


def ease_to_rpy(target_rpy_sequence, start_rpy, durations, ease_in_s):
    """Like gaze_waypoints, but the per-waypoint target orientation is already
    known (e.g. the fixed PICK_ORIENTATION_DEG / ORIENT_LOCK schedule) rather
    than computed live from a look-at point.

    Needed because, after a gazed arc, the arm's ACTUAL orientation can be far
    from PICK_ORIENTATION_DEG -- commanding that fixed pose as a hard constant
    from waypoint 1 (the old, gaze-less code's assumption, which always held
    because every arc commanded the same constant) then demands an instant
    snap on the very first waypoint. Used both for a gaze-disabled/no-target
    arc and for the gaze-unreachable fallback, so reverting to the fixed pose
    always eases there smoothly instead. A no-op (returns target_rpy_sequence
    verbatim) when start_rpy already matches it, e.g. GAZE_ENABLED=False for
    the whole run -- every arc has always held the same fixed pose then, so
    there is nothing to ease from."""
    tau = max(float(ease_in_s), 0.0)
    R = kinematics.rpy_to_matrix(*start_rpy)
    rx_list, ry_list, rz_list = [], [], []
    last_dt = 0.0
    for i, target_rpy in enumerate(target_rpy_sequence):
        dt = float(durations[i]) if i < len(durations) else last_dt
        last_dt = dt
        R_goal = kinematics.rpy_to_matrix(*target_rpy)
        if tau <= 1e-9:
            R = R_goal
        else:
            alpha = 1.0 - math.exp(-max(dt, 0.0) / tau)
            R = _slerp_matrix(R, R_goal, alpha)
        rx, ry, rz = kinematics.matrix_to_rpy(R)
        rx_list.append(float(rx))
        ry_list.append(float(ry))
        rz_list.append(float(rz))
    return rx_list, ry_list, rz_list
