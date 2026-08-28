"""
The public API: get_coords() and send_coords().

    from armik import Arm

    arm = Arm(port="/dev/ttyTHS1")
    arm.get_coords()                              # [x,y,z (cm), rx,ry,rz (deg)]
    arm.send_coords(x=20, y=-6, z=15, speed=4.0)  # -> 1 or 0
    arm.send_coords(z=12)                          # z only; everything else free

UNITS AT THIS BOUNDARY: positions in CENTIMETRES, orientations in DEGREES,
speed in CM/S of end-effector travel. Everything inside the package works in
mm/deg; the conversion happens only in _cm_to_mm / _mm_to_cm below, so there is
exactly one place to look when a number is off by a factor of ten.

HOW A MOVE IS EXECUTED
----------------------
1. Read the current joint angles, run FK to get the current task coordinates.
2. Build a target vector: constrained coordinates take your value, free ones
   are anchored to their current value.
3. Sample a STRAIGHT LINE from current to target and solve IK at every
   waypoint, each seeded from the previous solution.
4. Check the whole path against joint limits and servo speed ceilings.
5. Only if all of that succeeded, stream the joint setpoints to pymycobot's
   send_angles() on an absolute-deadline schedule.

Steps 1-4 happen before anything is sent. If any of them fails, the arm does
not move at all -- never partially, never stranded mid-path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from . import blending, config, ik
from .connection import ArmConnection, ArmError
from .kinematics import (
    check_joint_limits,
    check_workspace_bounds,
    forward_kinematics,
    manipulability,
    pose_coords,
    wrap180,
)

log = logging.getLogger(__name__)

SUCCESS = 1
FAILURE = 0


def _cm_to_mm(v: float) -> float:
    return float(v) * 10.0


def _mm_to_cm(v: float) -> float:
    return float(v) / 10.0


@dataclass
class Plan:
    """What was worked out before anything moved. Inspect after a failure."""
    ok: bool = False
    error: str | None = None
    q_waypoints: np.ndarray | None = None
    timestamps: np.ndarray | None = None    # explicit per-sample times; None = uniform
    duration_s: float = 0.0
    path_length_cm: float = 0.0
    constrained: list[str] = field(default_factory=list)
    free_drift: dict[str, float] = field(default_factory=dict)
    peak_joint_dps: float = 0.0
    start_coords: list[float] = field(default_factory=list)
    target_coords: list[float] = field(default_factory=list)
    segment_q: np.ndarray | None = None
    segment_duration: np.ndarray | None = None
    # populated only by send_path()/plan_path()
    waypoint_times: list[float] = field(default_factory=list)
    entry_velocity: np.ndarray | None = None
    exit_velocity: np.ndarray | None = None


@dataclass
class Execution:
    """What actually happened on the wire."""
    ok: bool = False
    error: str | None = None
    duration_s: float = 0.0
    setpoints: int = 0
    late_deadlines: int = 0
    t_cmd: list[float] = field(default_factory=list)
    q_cmd: list[list[float]] = field(default_factory=list)

    def to_csv(self, path: str) -> None:
        import csv
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + [f"J{i+1}" for i in range(config.DOF)])
            for t, q in zip(self.t_cmd, self.q_cmd):
                w.writerow([f"{t:.6f}"] + [f"{v:.4f}" for v in q])


class Arm:
    """
    Cartesian control of the myCobot 280 built directly on pymycobot.

    Motion is issued exclusively through pymycobot's send_angles(); the
    firmware's own send_coords()/send_coord() are never used, because they hide
    the IK, demand a full 6-DOF pose, and fail silently when the pose is
    unreachable.
    """

    def __init__(
        self,
        port: str = config.DEFAULT_PORT,
        baudrate: int = config.DEFAULT_BAUDRATE,
        mock: bool = False,
        control_rate_hz: float | None = None,
        connection: ArmConnection | None = None,
    ):
        self.conn = connection or ArmConnection(port, baudrate, mock=mock)
        self.control_rate_hz = float(control_rate_hz or config.CONTROL_RATE_HZ)
        self.last_plan: Plan | None = None
        self.last_execution: Execution | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coords(self) -> list[float]:
        """
        Current end-effector pose: [x, y, z, rx, ry, rz].
        Position in CM, orientation in DEGREES.

        Computed with this package's forward kinematics, not the firmware's, so
        it is consistent with whatever send_coords() will aim at. (Firmware FK
        is available via conn.get_coords_firmware() -- see verify_fk.py.)
        """
        q = np.array(self.conn.get_angles(), dtype=float)
        c = pose_coords(q)
        out = [_mm_to_cm(c[0]), _mm_to_cm(c[1]), _mm_to_cm(c[2]),
               float(c[3]), float(c[4]), float(c[5])]
        if config.Z_RELATIVE_TO_JOINT1:
            out[2] -= config.JOINT_1_HEIGHT_CM
        return [round(float(v), 4) for v in out]

    def send_coords(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        rx: float | None = None,
        ry: float | None = None,
        rz: float | None = None,
        speed: float = config.DEFAULT_SPEED_CM_S,
    ) -> int:
        """
        Move the end effector in a straight line to the given coordinates.

        Every coordinate is optional. Pass only what you want held; anything
        left as None is free to change, but will change as little as the solver
        can manage.

            arm.send_coords(x=20, y=-6, z=15)         # position, any orientation
            arm.send_coords(z=12)                      # drop z, hold the rest
            arm.send_coords(x=18, rz=45, speed=2.0)    # mixed, slowly

        Parameters
        ----------
        x, y, z    : target position in CM (base frame). None = unconstrained.
        rx, ry, rz : target orientation in DEGREES. None = unconstrained.
        speed      : end-effector linear speed in CM/S. If only orientation is
                     constrained, config.DEFAULT_ROT_SPEED_DEG_S is used.

        Returns
        -------
        1 on success, 0 on failure.

        On failure the arm does not move at all. The reason is available in
        `self.last_error` (and `self.last_plan.error`) when you want it --
        silent to the caller, not silent to you.
        """
        self.last_error = None
        self.last_execution = None

        plan = self.plan_coords(x, y, z, rx, ry, rz, speed)
        self.last_plan = plan
        if not plan.ok:
            self.last_error = plan.error
            log.info("send_coords refused: %s", plan.error)
            return FAILURE

        #execution = self._execute(plan)
        if self.get_single_joint() == 1:
            execution = self._execute_single_joint(plan, speed=speed)
        else:
            execution = self._execute(plan)
        
        self.last_execution = execution
        if not execution.ok:
            self.last_error = execution.error
            log.info("send_coords aborted: %s", execution.error)
            return FAILURE
        return SUCCESS

    def plan_coords(
        self,
        x=None, y=None, z=None, rx=None, ry=None, rz=None,
        speed: float = config.DEFAULT_SPEED_CM_S,
    ) -> Plan:
        """
        Work out the whole motion WITHOUT moving. Same arguments as
        send_coords(). Useful for checking reachability, inspecting the drift
        on free coordinates, or previewing peak joint speed before committing.
        """
        return self._plan([x, y, z, rx, ry, rz], speed)

    def send_path(
        self,
        x=None, y=None, z=None, rx=None, ry=None, rz=None,
        speed: float = config.DEFAULT_SPEED_CM_S,
        durations=None,
        v_start=None,
        v_end=None,
    ) -> int:
        """
        Sweep through a LIST of points as one continuous motion -- velocity is
        carried through interior waypoints instead of being reset to zero at
        each one, unlike calling send_coords() once per point.

        Each of x, y, z, rx, ry, rz is one of:
            None    -> free the whole way (soft-anchored to the start pose,
                       same semantics as send_coords())
            a float -> held at that fixed value across every waypoint
            a list  -> one value per waypoint; this is what defines how many
                       waypoints the path has (all list-valued arguments must
                       be the same length)

        At least one argument must be a list. The current position is the
        implicit start; it is not one of the waypoints you pass in.

            ys = [p[0] for p in points]
            zs = [p[1] for p in points]
            arm.send_path(x=cur_x, y=ys, z=zs, speed=15)

        Parameters
        ----------
        speed : cm/s. Used to derive each segment's duration from its
            distance, exactly like send_coords() -- unless `durations` is
            given explicitly.
        durations : optional list, one entry per waypoint (seconds from the
            previous point, or from the current position for the first one).
            Overrides `speed`; this is the raw distance/duration control if
            you want it directly rather than via a single cm/s figure.
        v_start, v_end : optional (6-long, in the same units as get_angles())
            JOINT velocities to carry into/out of this call, for chaining
            consecutive send_path() calls without a stop at the boundary --
            pass the previous call's `last_plan.exit_velocity`. Default zero:
            start and end at rest.

        Returns 1 on success, 0 on failure; see self.last_error /
        self.last_plan on failure, same as send_coords().

        Known limitation: blending happens in JOINT space (see
        armik.blending for why), so the Cartesian path between two
        consecutive waypoints is not enforced to be a straight line -- only
        the waypoints themselves land exactly where you asked.
        """
        self.last_error = None
        self.last_execution = None

        plan = self.plan_path(x, y, z, rx, ry, rz, speed=speed,
                              durations=durations, v_start=v_start, v_end=v_end)
        self.last_plan = plan
        if not plan.ok:
            self.last_error = plan.error
            log.info("send_path refused: %s", plan.error)
            return FAILURE

        if self.get_single_joint() == 1:
            execution = self._execute_single_joint(plan, speed=speed)
        else:
            execution = self._execute(plan)

        #execution = self._execute(plan)
        
        self.last_execution = execution
        
        if not execution.ok:
            self.last_error = execution.error
            log.info("send_path aborted: %s", execution.error)
            return FAILURE
        return SUCCESS

    def plan_path(
        self,
        x=None, y=None, z=None, rx=None, ry=None, rz=None,
        speed: float = config.DEFAULT_SPEED_CM_S,
        durations=None,
        v_start=None,
        v_end=None,
    ) -> Plan:
        """Work out a send_path() motion WITHOUT moving. Same arguments."""
        return self._plan_path([x, y, z, rx, ry, rz], speed, durations, v_start, v_end)

    # -- convenience --------------------------------------------------------

    def set_single_joint(self, mode: int) -> int:
        return self.conn.set_single_joint(mode)

    def get_single_joint(self) -> int:
        return self.conn.get_single_joint()


    def get_angles(self) -> list[float]:
        """Joint angles in degrees, straight from pymycobot."""
        return self.conn.get_angles()

    def send_angles(self, angles_deg, speed_dps: float) -> int:
        """Direct joint-space command. speed_dps is a float in deg/s."""
        try:
            self.conn.send_angles(angles_deg, speed_dps)
            return SUCCESS
        except (ArmError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return FAILURE

    def move_joints(self, q_target_deg, duration: float) -> int:
        """
        Straight joint-space move over a fixed duration in seconds. Handy for
        homing, where a Cartesian path is unnecessary.
        """
        try:
            duration_s = float(duration)
            if duration_s <= 0:
                raise ValueError("duration must be positive")
            q0 = np.array(self.conn.get_angles(), dtype=float)
            qT = np.asarray(q_target_deg, dtype=float)
            if qT.shape != (config.DOF,):
                raise ValueError(f"expected {config.DOF} joint angles")
            problems = check_joint_limits(qT)
            if problems:
                raise ValueError("target violates joint limits: " + "; ".join(problems))
            n = max(2, int(round(duration_s * self.control_rate_hz)) + 1)
            s = _min_jerk(np.linspace(0.0, 1.0, n))
            wps = q0[None, :] + (qT - q0)[None, :] * s[:, None]
            plan = Plan(ok=True, q_waypoints=wps, duration_s=duration_s)
            ex = self._execute(plan)
            self.last_execution = ex
            if not ex.ok:
                self.last_error = ex.error
                return FAILURE
            return SUCCESS
        except (ArmError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return FAILURE

    def stop(self):
        return self.conn.stop()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self, requested: list, speed: float) -> Plan:
        plan = Plan()
        try:
            mask = np.array([v is not None for v in requested], dtype=bool)
            if not mask.any():
                plan.error = "no coordinates given; constrain at least one"
                return plan
            if speed <= 0:
                plan.error = "speed must be positive"
                return plan
            if not self.conn.is_power_on():
                plan.error = "arm reports power off; call arm.conn.power_on()"
                return plan

            q0 = np.array(self.conn.get_angles(), dtype=float)
            start = pose_coords(q0)                       # mm / deg
            plan.start_coords = _to_cm_list(start)

            if abs(abs(start[4]) - 90.0) < config.GIMBAL_WARN_DEG:
                log.warning(
                    "near Euler gimbal lock (ry=%.1f deg); orientation "
                    "constraints are ill-conditioned here", start[4]
                )

            # --- assemble the target in internal units --------------------
            target = start.copy()
            for i, v in enumerate(requested):
                if v is None:
                    continue
                if i < 3:
                    val = _cm_to_mm(v)
                    if i == 2 and config.Z_RELATIVE_TO_JOINT1:
                        val += _cm_to_mm(config.JOINT_1_HEIGHT_CM)
                    target[i] = val
                else:
                    target[i] = float(v)

            plan.constrained = [n for n, m in zip(config.COORD_NAMES, mask) if m]
            plan.target_coords = _to_cm_list(target)

            # --- distance, and therefore duration -------------------------
            pos_mask = mask[:3]
            delta_pos = np.where(pos_mask, target[:3] - start[:3], 0.0)
            dist_mm = float(np.linalg.norm(delta_pos))
            rot_delta = wrap180(target[3:] - start[3:])
            rot_span = float(np.max(np.abs(np.where(mask[3:], rot_delta, 0.0))))

            if dist_mm < 1e-6 and rot_span < 1e-6:
                plan.error = "already at the requested coordinates"
                return plan

            if dist_mm > 1e-6:
                duration = _mm_to_cm(dist_mm) / speed
            else:
                duration = rot_span / config.DEFAULT_ROT_SPEED_DEG_S
            duration = max(duration, 0.2)

            plan.path_length_cm = round(_mm_to_cm(dist_mm), 4)
            plan.duration_s = duration

            # --- sample the straight line ---------------------------------
            n_by_step = int(np.ceil(max(dist_mm, rot_span) / config.MAX_STEP_MM)) + 1
            n_by_rate = int(round(duration * self.control_rate_hz)) + 1
            n = max(config.MIN_WAYPOINTS, min(n_by_step, n_by_rate))

            s = _min_jerk(np.linspace(0.0, 1.0, n))

            q_wps = np.zeros((n, config.DOF))
            q_wps[0] = q0
            q_seed = q0.copy()

            for k in range(1, n):
                desired = start.copy()
                # Constrained position -> straight line.
                # Constrained rotation -> shortest-arc linear.
                # Free coordinates     -> anchored at the START value, so drift
                #                         is measured against where we began,
                #                         not allowed to creep waypoint to
                #                         waypoint.
                desired[:3] = np.where(pos_mask, start[:3] + delta_pos * s[k], start[:3])
                desired[3:] = np.where(
                    mask[3:], wrap180(start[3:] + rot_delta * s[k]), start[3:]
                )
                q_k = ik.try_solve(desired, mask, q_seed, max_iters=80)
                if q_k is None:
                    plan.error = (
                        f"unreachable along the straight line at {100.0*s[k]:.0f}% "
                        f"of the path (constrained: {', '.join(plan.constrained)}). "
                        f"Nothing was sent to the arm."
                    )
                    return plan
                q_wps[k] = q_k
                q_seed = q_k

            # --- validate the whole path before committing ----------------
            err = self._precheck(q_wps, duration, plan)
            if err:
                plan.error = err
                return plan

            # --- report drift on the free coordinates ---------------------
            end = pose_coords(q_wps[-1])
            drift = {}
            for i, name in enumerate(config.COORD_NAMES):
                if mask[i]:
                    continue
                d = float(end[i] - start[i])
                drift[name] = round(float(_mm_to_cm(d) if i < 3 else wrap180(d)), 3)
            plan.free_drift = drift

            plan.segment_q = np.array([q0, q_wps[-1]])
            plan.segment_durations = np.array([duration])
            
            plan.q_waypoints = q_wps
            plan.ok = True
            return plan

        except (ArmError, ValueError) as exc:
            plan.error = f"{type(exc).__name__}: {exc}"
            return plan

    def _plan_path(self, requested_axes: list, speed: float, durations, v_start, v_end) -> Plan:
        plan = Plan()
        try:
            # --- classify each axis: free / fixed-scalar / per-waypoint list --
            kinds = []
            n_waypoints = None
            for v in requested_axes:
                if v is None:
                    kinds.append(("free", None))
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    kinds.append(("scalar", float(v)))
                    continue
                arr = np.asarray(v, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f"a path coordinate must be a scalar or 1-D "
                                     f"sequence, got shape {arr.shape}")
                if n_waypoints is None:
                    n_waypoints = len(arr)
                elif len(arr) != n_waypoints:
                    plan.error = "all per-waypoint coordinate lists must have the same length"
                    return plan
                kinds.append(("array", arr))

            if n_waypoints is None:
                plan.error = ("at least one coordinate must be given as a list to "
                              "define a path -- use send_coords() for a single point")
                return plan
            if n_waypoints < 1:
                plan.error = "need at least one waypoint"
                return plan

            mask = np.array([k != "free" for k, _ in kinds], dtype=bool)
            if not mask.any():
                plan.error = "no coordinates given; constrain at least one"
                return plan
            if speed <= 0:
                plan.error = "speed must be positive"
                return plan
            if not self.conn.is_power_on():
                plan.error = "arm reports power off; call arm.conn.power_on()"
                return plan

            # --- fill a (n_waypoints, 6) target array in native units (cm/deg) --
            targets_native = np.zeros((n_waypoints, 6))
            for i, (kind, val) in enumerate(kinds):
                if kind == "scalar":
                    targets_native[:, i] = val
                elif kind == "array":
                    targets_native[:, i] = val

            # --- convert to internal units (mm/deg) ----------------------------
            targets = targets_native.copy()
            targets[:, :3] = targets_native[:, :3] * 10.0     # cm -> mm
            if config.Z_RELATIVE_TO_JOINT1:
                targets[:, 2] += _cm_to_mm(config.JOINT_1_HEIGHT_CM)

            q0 = np.array(self.conn.get_angles(), dtype=float)
            start = pose_coords(q0)                       # mm / deg
            plan.start_coords = _to_cm_list(start)
            plan.constrained = [n for n, m in zip(config.COORD_NAMES, mask) if m]

            if abs(abs(start[4]) - 90.0) < config.GIMBAL_WARN_DEG:
                log.warning(
                    "near Euler gimbal lock (ry=%.1f deg); orientation "
                    "constraints are ill-conditioned here", start[4]
                )

            # --- solve IK at each waypoint, seeded from the previous solution --
            # (chained seeding keeps the whole path on one IK branch, exactly
            # like _plan()'s straight-line sampling does within a single call)
            q_list = [q0]
            q_seed = q0.copy()
            for k in range(n_waypoints):
                desired = start.copy()
                desired[mask] = targets[k][mask]
                q_k = ik.try_solve(desired, mask, q_seed, max_iters=100)
                if q_k is None:
                    plan.error = (
                        f"waypoint {k+1} of {n_waypoints} unreachable "
                        f"(constrained: {', '.join(plan.constrained)}). "
                        f"Nothing was sent to the arm."
                    )
                    return plan
                q_list.append(q_k)
                q_seed = q_k
            q_list = np.array(q_list)                      # (n_waypoints+1, 6)
            plan.target_coords = _to_cm_list(pose_coords(q_list[-1]))

            for i, q in enumerate(q_list[1:], start=1):
                problems = check_joint_limits(q)
                if problems:
                    plan.error = f"waypoint {i} violates joint limits: " + "; ".join(problems)
                    return plan

            # --- durations: explicit, or derived from distance/speed per segment
            if durations is not None:
                if isinstance(durations, (int, float)):
                    durations = [durations] * n_waypoints
                durations = np.asarray(durations, dtype=float)
                if len(durations) != n_waypoints:
                    plan.error = "need exactly one duration per waypoint"
                    return plan
                if np.any(durations <= 0):
                    plan.error = "all durations must be positive"
                    return plan
            else:
                durations = np.zeros(n_waypoints)
                prev = start
                for k in range(n_waypoints):
                    desired = start.copy()
                    desired[mask] = targets[k][mask]
                    delta_pos = np.where(mask[:3], desired[:3] - prev[:3], 0.0)
                    dist_mm = float(np.linalg.norm(delta_pos))
                    rot_span = float(np.max(np.abs(
                        np.where(mask[3:], wrap180(desired[3:] - prev[3:]), 0.0)
                    )))
                    if dist_mm > 1e-6:
                        durations[k] = _mm_to_cm(dist_mm) / speed
                    elif rot_span > 1e-6:
                        durations[k] = rot_span / config.DEFAULT_ROT_SPEED_DEG_S
                    else:
                        durations[k] = 0.2   # same point requested twice in a row
                    durations[k] = max(durations[k], 0.05)
                    prev = desired

            plan.path_length_cm = round(
                sum(_mm_to_cm(float(np.linalg.norm(
                    forward_kinematics(q_list[i + 1])[:3, 3] - forward_kinematics(q_list[i])[:3, 3]
                ))) for i in range(n_waypoints)), 4
            )

            v0 = np.asarray(v_start, dtype=float) if v_start is not None else None
            v1 = np.asarray(v_end, dtype=float) if v_end is not None else None
            v_wp = blending.waypoint_velocities(q_list, durations, v_start=v0, v_end=v1)
            plan.entry_velocity, plan.exit_velocity = v_wp[0].copy(), v_wp[-1].copy()

            err = self._precheck_blended(q_list, v_wp, durations, plan)
            if err:
                plan.error = err
                return plan

            waypoints, timestamps = self._sample_blended(q_list, v_wp, durations)
            plan.q_waypoints = waypoints
            plan.segment_q = q_list
            plan.segment_durations = durations
            plan.timestamps = timestamps
            plan.duration_s = float(timestamps[-1])
            plan.waypoint_times = [float(t) for t in np.cumsum(durations)]

            end = pose_coords(q_list[-1])
            drift = {}
            for i, name in enumerate(config.COORD_NAMES):
                if mask[i]:
                    continue
                d = float(end[i] - start[i])
                drift[name] = round(float(_mm_to_cm(d) if i < 3 else wrap180(d)), 3)
            plan.free_drift = drift

            plan.ok = True
            return plan

        except (ArmError, ValueError) as exc:
            plan.error = f"{type(exc).__name__}: {exc}"
            return plan

    def _precheck(self, waypoints: np.ndarray, duration: float, plan: Plan) -> str | None:
        """Return an error string if the path is not executable, else None."""
        dt = duration / (len(waypoints) - 1)
        dps = np.abs(np.diff(waypoints, axis=0)) / dt
        plan.peak_joint_dps = float(np.max(dps)) if dps.size else 0.0

        limits = np.array(config.MAX_JOINT_SPEED_DPS)
        over = np.where(np.max(dps, axis=0) > limits)[0]
        if over.size:
            worst = ", ".join(
                f"J{i+1} needs {np.max(dps[:, i]):.0f} deg/s (max {limits[i]:.0f})"
                for i in over
            )
            return (
                f"too fast for the hardware: {worst}. Lower `speed` "
                f"(currently completes in {duration:.2f}s)."
            )

        for tag, q in (("start", waypoints[0]), ("end", waypoints[-1])):
            problems = check_joint_limits(q)
            if problems:
                return f"{tag} configuration violates joint limits: " + "; ".join(problems)

        sample = waypoints[:: max(1, len(waypoints) // 10)]
        worst_m = min(manipulability(q) for q in sample)
        if worst_m < 1e3:
            log.warning(
                "path passes near a singularity (manipulability %.3g); "
                "expect rough joint motion there", worst_m
            )
        return None

    def _precheck_blended(self, q_list, v_wp, durations, plan: Plan) -> str | None:
        """
        Like _precheck(), but for a blended multi-waypoint path: uses the
        ANALYTIC Hermite derivative on a dense grid per segment rather than a
        finite difference of the streamed samples, because the true peak speed
        of a Hermite curve can fall between streamed setpoints.
        """
        limits = np.array(config.MAX_JOINT_SPEED_DPS)
        peak_dps = 0.0
        for i in range(len(durations)):
            vel = blending.hermite_velocity(
                q_list[i], v_wp[i], q_list[i + 1], v_wp[i + 1], durations[i], 200
            )
            seg_peak = np.max(np.abs(vel), axis=0)
            peak_dps = max(peak_dps, float(np.max(seg_peak)))
            over = np.where(seg_peak > limits)[0]
            if over.size:
                worst = ", ".join(
                    f"J{j+1} needs {seg_peak[j]:.0f} deg/s (max {limits[j]:.0f})"
                    for j in over
                )
                return (
                    f"segment {i+1} of {len(durations)} too fast for the hardware: "
                    f"{worst}. Lower `speed`, or give that segment more time via "
                    f"`durations`."
                )
        plan.peak_joint_dps = peak_dps

        for tag, q in (("start", q_list[0]), ("end", q_list[-1])):
            problems = check_joint_limits(q)
            if problems:
                return f"{tag} configuration violates joint limits: " + "; ".join(problems)

        worst_m = min(manipulability(q) for q in q_list)
        if worst_m < 1e3:
            log.warning(
                "path passes near a singularity (manipulability %.3g); "
                "expect rough joint motion there", worst_m
            )
        return None

    def _sample_blended(self, q_list, v_wp, durations):
        """Sample every Hermite segment at the control rate and concatenate."""
        all_q, all_t = [q_list[0][None, :]], [np.array([0.0])]
        t_offset = 0.0
        for i in range(len(durations)):
            n = max(2, int(round(durations[i] * self.control_rate_hz)) + 1)
            seg_q = blending.hermite_sample(
                q_list[i], v_wp[i], q_list[i + 1], v_wp[i + 1], durations[i], n
            )
            seg_t = t_offset + np.linspace(0.0, durations[i], n)
            all_q.append(seg_q[1:])     # skip the sample shared with the previous segment
            all_t.append(seg_t[1:])
            t_offset += durations[i]
        return np.vstack(all_q), np.concatenate(all_t)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, plan: Plan) -> Execution:
        """
        Stream the planned joint setpoints on an ABSOLUTE-deadline schedule.

        Absolute deadlines matter: a naive `time.sleep(dt)` in a loop
        accumulates the send time plus scheduler jitter on every iteration, so
        a 3-second trajectory quietly becomes 3.4 seconds. Scheduling against
        t0 + timestamps[k] keeps total duration honest.

        plan.timestamps, if set, gives the exact time (seconds from the start
        of execution) each waypoint should be reached -- used by send_path()
        for a blended multi-segment path, where segments can have different
        durations. If unset, the setpoints are assumed evenly spaced across
        plan.duration_s (the single-target send_coords()/move_joints() case).
        """
        ex = Execution()
        wps = plan.q_waypoints
        n = len(wps)
        timestamps = (
            plan.timestamps if plan.timestamps is not None
            else np.linspace(0.0, plan.duration_s, n)
        )
        gain = config.STREAM_SPEED_GAIN

        try:
            t0 = time.perf_counter()
            for k in range(1, n):
                deadline = t0 + timestamps[k]
                now = time.perf_counter()
                dt_k = max(timestamps[k] - timestamps[k - 1], 1e-6)
                if now < deadline:
                    time.sleep(deadline - now)
                elif now > deadline + dt_k:
                    ex.late_deadlines += 1

                q_k = wps[k]
                step_dps = float(np.max(np.abs(q_k - wps[k - 1])) / dt_k)
                self.conn.send_angles(q_k, max(step_dps * gain, 1.0))

                ex.t_cmd.append(time.perf_counter() - t0)
                ex.q_cmd.append([float(v) for v in q_k])

            ex.duration_s = time.perf_counter() - t0
            ex.setpoints = len(ex.q_cmd)
            ex.ok = True
        except (ArmError, ValueError) as exc:
            ex.error = f"{type(exc).__name__}: {exc}"
            self.conn.stop()
        return ex

    
    def _execute_single_joint(self, plan: Plan, speed: float | None = None) -> Execution:
        """
        Execute a path by moving exactly one joint at a time.

        For every waypoint:
            J1 -> target
            wait until J1 reaches target
            SINGLE_JOINT_DELAY
            J2 -> target
            wait until J2 reaches target
            SINGLE_JOINT_DELAY
            ...
            J6 -> target

        The first waypoint in plan.q_waypoints is the current configuration.
        """

        #print("A: Intro execute_single_joint")

        ex = Execution()
        q_waypoints = plan.segment_q

        if q_waypoints is None or len(q_waypoints) < 2:
            ex.error = "single-joint execution requires at least one target waypoint"
            return ex

        try:
            t0 = time.perf_counter()

            current = np.array(self.conn.get_angles(), dtype=float)

            #print("B: before for-loops")

            for waypoint_idx in range(1, len(q_waypoints)):
                
                #print("Iteration #", waypoint_idx)

                target = np.asarray(q_waypoints[waypoint_idx], dtype=float)

                #print("0", end='')

                n_moving = sum(
                    1 for j in range(config.DOF)
                    if abs(float(target[j]) - float(current[j])) > config.SINGLE_JOINT_TOL_DEG
                )

                for joint_idx in range(config.DOF):
                    joint_id = joint_idx + 1
                    target_angle = float(target[joint_idx])
                    current_angle = float(current[joint_idx])
                    
                    #print("1", end='')

                    # Nothing to do if this joint is already at the target.
                    if abs(target_angle - current_angle) <= config.SINGLE_JOINT_TOL_DEG:
                        continue

                    if speed is not None:
                        joint_speed = float(speed)
                    elif plan.segment_durations is not None:
                        seg_duration = float(plan.segment_durations[waypoint_idx - 1]) / n_moving
                        joint_speed = abs(target_angle - current_angle) / max(seg_duration, 1e-6)
                    else:
                        joint_speed = float(config.MAX_JOINT_SPEED_DPS[joint_idx])

                    joint_speed = min(joint_speed, config.MAX_JOINT_SPEED_DPS[joint_idx])
                    
                    #Use the configured hardware ceiling for this joint.
                    #joint_speed = 20 #float(config.MAX_JOINT_SPEED_DPS[joint_idx])

                    #print("2", end='')

                    candidate = current.copy()
                    candidate[joint_idx] = target_angle
                    tip_mm = forward_kinematics(candidate)[:3, 3]
                    bounds_error = check_workspace_bounds(tip_mm)
                    if bounds_error is not None:
                        ex.error = f"J{joint_id} refused -- {bounds_error}"
                        return ex

                    self.conn.send_angle(
                        joint_id,
                        target_angle,
                        joint_speed,
                    )

                    #print("3", end='')

                    # Wait until the servo actually reaches the target.
                    self._wait_for_joint(
                        joint_id,
                        target_angle,
                    )

                    #print("4")

                    # Record the commanded configuration.
                    current[joint_idx] = target_angle

                    ex.t_cmd.append(time.perf_counter() - t0)
                    ex.q_cmd.append(current.tolist())

                    #print("C: before SINGLE_JOINT_DELAY")

                    # Delay BEFORE starting the next joint.
                    time.sleep(config.SINGLE_JOINT_DELAY)
                    
                if waypoint_idx < len(q_waypoints) - 1:
                    time.sleep(config.SINGLE_JOINT_DELAY_BETWEEN_POINTS)
                    

            ex.duration_s = time.perf_counter() - t0
            ex.setpoints = len(ex.q_cmd)
            ex.ok = True

        except (ArmError, ValueError) as exc:
            ex.error = f"{type(exc).__name__}: {exc}"
            self.conn.stop()

        return ex


    def _wait_for_joint(
        self,
        joint_id: int,
        target_deg: float,
    ) -> None:
        """
        Block until the requested joint is sufficiently close to its target.
        """

        period = 1.0 / config.SINGLE_JOINT_POLL_HZ
        deadline = time.perf_counter() + config.SINGLE_JOINT_TIMEOUT_S

        while True:
            angles = self.conn.get_angles()
            current = float(angles[joint_id - 1])

            print(f"  J{joint_id} target={target_deg:.2f} current={current:.2f} "
                  f"gap={abs(current-target_deg):.2f}")

            if abs(current - target_deg) <= config.SINGLE_JOINT_TOL_DEG:
                return
            if time.perf_counter() > deadline:
                raise ArmError(
                    f"J{joint_id} did not reach {target_deg:.2f} deg "
                    f"within {config.SINGLE_JOINT_TIMEOUT_S:.1f}s "
                    f"(currently at {current:.2f} deg)"
                )

            time.sleep(period)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _min_jerk(tau: np.ndarray) -> np.ndarray:
    """
    Minimum-jerk time scaling: zero velocity AND zero acceleration at both
    ends. Without this the arm would start and stop with an instantaneous
    velocity step, which is hard on the gearing and looks abrupt.
    """
    t = np.clip(tau, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def _to_cm_list(coords_mm_deg: np.ndarray) -> list[float]:
    c = coords_mm_deg
    out = [_mm_to_cm(c[0]), _mm_to_cm(c[1]), _mm_to_cm(c[2]),
           float(c[3]), float(c[4]), float(c[5])]
    if config.Z_RELATIVE_TO_JOINT1:
        out[2] -= config.JOINT_1_HEIGHT_CM
    return [round(float(v), 4) for v in out]
