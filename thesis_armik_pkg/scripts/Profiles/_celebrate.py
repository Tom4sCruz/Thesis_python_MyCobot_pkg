"""
SUCCESS ANIMATION helper, shared by HighH-LowR.py / HighH-HighR.py.
=============================================================================

A little "task succeeded" wag played once the whole run is done: a pure
left-right shoulder swing, with the gripper staying on a single vertical
line and upright throughout. Pure numpy; no new dependency.

WHY THIS IS JOINT-SPACE, NOT CARTESIAN
----------------------------------------
Hold J1 (base) fixed at 90 deg. At that base angle, J2 (shoulder), J3
(elbow) and J4 (wrist pitch) all rotate about an axis parallel to world X
(confirmed via armik.kinematics.forward_kinematics: sweeping any of them
individually holds tip x at 0.0 to sub-micron precision) -- so the whole
sub-chain moves in the world YZ-plane, with zero depth component. J2 is
the swept/driven joint (clockwise then counterclockwise); J3/J4 are SOLVED
each step so the tip's world Y stays fixed while Z takes whatever height
is actually available (see below). J5 (wrist roll) and J6 are held fixed
throughout -- sweeping J5 breaks the x=0 plane constraint badly, so it must
not move.

The fully-extended "arm straight up" pose (J2=J3=0) is a genuine kinematic
singularity of the general 6-DOF Cartesian solver (near-zero manipulability
-- armik.ik/plan_coords refuses even a +/-1cm z nudge there, "unreachable
along the straight line"), and letting J1 float to dodge it drifts off 90
deg by about a degree. So this motion is authored directly in joint space
instead of going through arm.send_path()/plan_coords() like every other
move in these scripts.

WHY Z DIPS AT THE EXTREMES INSTEAD OF BOBBING INDEPENDENTLY
-------------------------------------------------------------
With J1=90 and J2=0, world Y=0 at max Z is the single highest point in the
whole reachable envelope for that Y -- only reachable at J2=0. As J2 swings
away from 0, the max Z reachable while holding Y=0 sags roughly
quadratically (empirically ~ -0.024 mm per deg^2 near this arm's own
straight-up pose). So the height target here tracks that natural envelope
with margin (z_apex_mm - z_sag_coeff * j2_deg**2, with z_sag_coeff kept
below the arm's own natural sag rate) rather than fighting it with an
independently authored bob -- asking for MORE height than the envelope
allows is exactly what makes the solver below diverge.

THE COMPENSATION SOLVE
------------------------
solve_shoulder_compensation is a small damped 2-unknown Newton solve for
(J3, J4) given J1, J2 (fixed) and a target (Y, Z): finite-difference 2x2
Jacobian + damped least squares + a per-step angle clamp, mirroring this
package's own 6-DOF IK (armik/ik.py's IK_DAMPING/IK_STEP_CLAMP_DEG) at a
much smaller scale. This 2-unknown subsystem is NOT singular at J2=0 --
it's a different, better-conditioned problem than the full 6-DOF Cartesian
one (confirmed by testing: condition numbers stayed in the 3-470 range
across a +/-20 deg sweep). The damping+clamping matters because, unlike
armik.ik.solve, this solver has no analytic feasibility check of its own:
a plain undamped Newton solve diverges violently (angles blowing up) when
asked for an unreachable (Y, Z) target (e.g. the exact apex off-center);
damped+clamped, it instead stalls near the closest reachable point.

SHAPE
-----
build_oscillation_waypoints runs J2 through `cycles` full sine periods
(0 -> +/-amplitude -> 0, ending back at 0) and then, without pausing,
continues into an eased settle so the very LAST waypoint lands at
final_shoulder_deg instead of snapping back to center -- one continuous
motion, no stop in between. J3/J4 are solved at every sample, warm-started
from the previous sample for continuity and solver speed.

celebration_durations is the same EASE_IN/EASE_OUT (0..10) time-warp shape
HighH-LowR.py's get_durations() uses for the Cartesian reach/carry arcs,
lifted out standalone: there's no chord length / cruise speed to derive a
duration from here -- duration_s is always given explicitly.

STATIC-WRIST-PITCH ALTERNATIVE (solve_tip_point_2link / solve_j6_level /
build_straight_arm_oscillation_waypoints), used by HighH-LowR.py only
----------------------------------------------------------------------------
An earlier version of this alternative held J4 (wrist-pitch) actively
compensating every waypoint (exact in simulation, ~1e-16 error) -- but on
real hardware J4 couldn't track that continuously-changing target fast or
precisely enough, so the "head and neck" wasn't actually straight in
practice. This version holds J4 STATIC instead (one servo doing nothing,
not chasing a moving target) and uses J6 (tool-roll) for the remaining
compensation job -- but J6 compensates something different from what J4
used to, not the same thing by another route; see below.

Reference pose q=[90, 0, 0, 0, -90, 0] (base, shoulder, elbow, wrist-
pitch, wrist-roll, tool-roll) gives tip pointing EXACTLY (1,0,0) -- true
+X (confirmed via forward_kinematics). Bonus fact this unlocks: at this
J5=-90, the pointing axis (3rd column of the rotation matrix) is EXACTLY
world +X for ANY (J2, J3, J4) -- not approximately, to ~1e-16/1e-32 over
large asymmetric sweeps -- because the reference pointing vector lands
exactly ON the J2/J3/J4 shared rotation axis (world X, since they all
rotate about an axis parallel to world X at base=90), and rotating a
vector about an axis it already lies on never changes it. So J4's old
compensation was never actually protecting orientation -- pointing
direction is invariant regardless of J4. What it WAS doing was keeping
the rigid "neck" (wrist-pitch pivot to tip, 73.18mm) pointing in a
constant direction relative to world, which is what made "wrist-pivot on
the line" equivalent to "tip on the line." Hold J4 static instead and the
neck swings around that shared X-axis as J2+J3 changes -- a purely
POSITIONAL effect (tip can land up to ~46mm lateral / ~17mm vertical off
the wrist-pivot's own position), not an orientation one.

So solve_tip_point_2link solves for (J2, J3) to put the ACTUAL TIP (not
the wrist-pivot) on the vertical line directly -- a damped 2-unknown
Newton solve (not closed-form law-of-cosines like the wrist-pivot-only
version was, since the tip's position relative to the pivot now depends
on J2+J3 too). It needs MUCH lighter damping than this module's other
solver (~0.03, not ~1.0): the Jacobian is EXACTLY rank-1 at (J2,J3)=(0,0)
-- a true singularity, stronger than the old near-full-extension
near-singularity -- and the usual damping over-suppresses the one
direction that still has gradient there, stalling completely for small
dips. build_straight_arm_oscillation_waypoints handles this by nudging the
solver's seed a small deliberate step in the intended direction whenever
the alternating elbow branch switches side (dip=0 crossings, where a bare
zero/zero warm-start would sit exactly on the singular point and default
to whichever side its null-space happens to favor, not necessarily the
intended one) -- see its own docstring.

Since J4 contributes nothing to orientation (confirmed above), the only
thing left to "level" is roll about the pointing axis itself -- rolling
the OTHER two rotation-matrix columns, which (because forward is exactly
world +X) live entirely in the world YZ-plane. solve_j6_level is an exact
closed form for this, mirroring _gaze.py's _look_at_matrix technique
(`right = normalize(cross(up, forward))`): J6_level = atan2(-z0, y0),
where (y0, z0) are the chosen column's world Y/Z at J6=0. Verified: lands
that column EXACTLY on (0,1,0) (Z-component ~1e-16) at every tested
(J2,J3), versus Z-components up to 0.94 (badly un-level) at J6=0.

The reference pose sits exactly at this sub-chain's full-extension
singularity (the highest the tip can reach) -- so it can only dip DOWN
from there, and small dips need disproportionately large J2/J3 swings (a
square-root-type relationship typical near full extension).
build_straight_arm_oscillation_waypoints dips `cycles` times (a (1-cos)/2
shape, always >= 0, since it can't go above the reference) and eases into
a smaller final_dip_mm resting bend instead of snapping back fully
straight -- one continuous motion, same shape as the Newton-based
build_oscillation_waypoints above.
"""

from __future__ import annotations

import math

import numpy as np

from armik import config, kinematics

_SOLVE_DAMPING = 1.0
_SOLVE_STEP_CLAMP_DEG = 5.0
_SOLVE_MAX_ITERS = 20
_SOLVE_TOL_MM = 0.01
_SOLVE_FD_EPS_DEG = 0.01


def _smootherstep(p):
    p = np.clip(p, 0.0, 1.0)
    return 6 * p ** 5 - 15 * p ** 4 + 10 * p ** 3


def _tip_yz(j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, j6_deg):
    T = kinematics.forward_kinematics([j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, j6_deg])
    return float(T[1, 3]), float(T[2, 3])


def solve_shoulder_compensation(j1_deg, j2_deg, y_target_mm, z_target_mm,
                                j5_deg, j6_deg, j3_seed_deg, j4_seed_deg):
    """Damped 2-unknown Newton solve for (J3, J4) deg so that, with J1/J2/J5/J6
    fixed as given, the tip (world Y, Z) lands at (y_target_mm, z_target_mm).
    Warm-started from (j3_seed_deg, j4_seed_deg). See the module docstring's
    THE COMPENSATION SOLVE section for why this is damped+clamped rather than
    plain Newton. Returns (j3_deg, j4_deg) -- the closest reachable point if
    the target itself turns out to be infeasible, not an exception."""
    j3, j4 = float(j3_seed_deg), float(j4_seed_deg)
    lam2 = _SOLVE_DAMPING ** 2
    eps = _SOLVE_FD_EPS_DEG
    for _ in range(_SOLVE_MAX_ITERS):
        y0, z0 = _tip_yz(j1_deg, j2_deg, j3, j4, j5_deg, j6_deg)
        err = np.array([y_target_mm - y0, z_target_mm - z0])
        if np.hypot(*err) < _SOLVE_TOL_MM:
            break
        y3, z3 = _tip_yz(j1_deg, j2_deg, j3 + eps, j4, j5_deg, j6_deg)
        y4, z4 = _tip_yz(j1_deg, j2_deg, j3, j4 + eps, j5_deg, j6_deg)
        J = np.array([[(y3 - y0) / eps, (y4 - y0) / eps],
                      [(z3 - z0) / eps, (z4 - z0) / eps]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(2), err)
        biggest = float(np.max(np.abs(dq)))
        if biggest > _SOLVE_STEP_CLAMP_DEG:
            dq *= _SOLVE_STEP_CLAMP_DEG / biggest
        j3 += float(dq[0])
        j4 += float(dq[1])
    return j3, j4


def build_oscillation_waypoints(base_j1_deg, shoulder_amplitude_deg, direction_sign,
                                cycles, final_shoulder_deg, z_apex_mm, z_sag_coeff,
                                j5_deg, j6_deg, j3_seed_deg, j4_seed_deg,
                                n_oscillate_waypoints, n_settle_waypoints):
    """Build the (n_oscillate_waypoints + n_settle_waypoints, 6) absolute
    joint-angle array (degrees) for the whole wag -- see the module
    docstring's SHAPE section. y_target is held at 0 throughout (the world Y
    of the straight-up staging pose). Returns an ndarray; pair with
    celebration_durations(..., n_waypoints=n_oscillate_waypoints +
    n_settle_waypoints + 1) for the timestamps."""
    n_osc = int(n_oscillate_waypoints)
    n_set = int(n_settle_waypoints)

    u1 = np.linspace(0.0, float(cycles), n_osc + 1)[1:]
    j2_osc = float(direction_sign) * float(shoulder_amplitude_deg) * np.sin(2.0 * np.pi * u1)

    u2 = np.linspace(0.0, 1.0, n_set + 1)[1:]
    j2_settle = j2_osc[-1] + (float(final_shoulder_deg) - j2_osc[-1]) * _smootherstep(u2)

    j2_all = np.concatenate([j2_osc, j2_settle])

    j3, j4 = float(j3_seed_deg), float(j4_seed_deg)
    rows = []
    for j2 in j2_all:
        z_target = float(z_apex_mm) - float(z_sag_coeff) * float(j2) ** 2
        j3, j4 = solve_shoulder_compensation(
            base_j1_deg, float(j2), 0.0, z_target, j5_deg, j6_deg, j3, j4)
        rows.append([base_j1_deg, float(j2), j3, j4, j5_deg, j6_deg])
    return np.array(rows, dtype=float)


def celebration_durations(ease_in, ease_out, duration_s, n_waypoints):
    """`n_waypoints`-1 segment durations (s), shaped by the EASE_IN /
    EASE_OUT dials (0..10, no physical meaning) exactly like get_durations()
    -- see its doc in HighH-LowR.py. duration_s is the total time, always
    explicit (there is no cruise-speed-derived fallback here)."""
    n_waypoints = int(n_waypoints)
    a = float(np.clip(ease_in, 0.0, 10.0)) / 10.0
    b = float(np.clip(ease_out, 0.0, 10.0)) / 10.0
    r_in = 0.05 + 0.45 * a
    r_out = 0.05 + 0.45 * b
    if r_in + r_out > 1.0:
        k = 1.0 / (r_in + r_out)
        r_in *= k
        r_out *= k

    tau = np.linspace(0.0, 1.0, 2001)
    v = np.ones_like(tau)
    m_in = tau < r_in
    p = tau[m_in] / r_in
    v[m_in] = (1.0 - a) * p + a * _smootherstep(p)
    m_out = tau > 1.0 - r_out
    q = (1.0 - tau[m_out]) / r_out
    v[m_out] = (1.0 - b) * q + b * _smootherstep(q)
    v = np.maximum(v, 1e-6)

    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tau))])
    s_norm = s / s[-1]
    T = float(duration_s)

    ss = np.linspace(0.0, 1.0, n_waypoints)
    tau_k = np.interp(ss, s_norm, tau)
    t_k = tau_k * T
    return np.maximum(np.diff(t_k), 1e-3).tolist()


# ---------------------------------------------------------------------------
# Static-wrist-pitch alternative: tip-targeting 2-link Newton solve + J6
# leveling. See the module docstring's STATIC-WRIST-PITCH ALTERNATIVE
# section. Used by HighH-LowR.py only -- HighH-HighR.py still uses the
# Newton-based solve_shoulder_compensation / build_oscillation_waypoints
# above.
# ---------------------------------------------------------------------------

_TIP_SOLVE_DAMPING = 0.03      # much lighter than _SOLVE_DAMPING above -- the
                               # Jacobian is EXACTLY rank-1 at (J2,J3)=(0,0)
                               # (see module docstring); the usual damping
                               # over-suppresses the one direction that still
                               # has gradient there and stalls for small dips
_TIP_SOLVE_STEP_CLAMP_DEG = 5.0
_TIP_SOLVE_MAX_ITERS = 30
_TIP_SOLVE_TOL_MM = 0.01
_TIP_SOLVE_FD_EPS_DEG = 0.01


def solve_tip_point_2link(j1_deg, target_y_mm, target_z_mm, j4_deg, j5_deg,
                          j2_seed_deg, j3_seed_deg):
    """Damped 2-unknown Newton solve for (J2, J3) deg so the TIP (not the
    wrist-pitch pivot) lands at (target_y_mm, target_z_mm), with J4 held at
    j4_deg (not compensating). Warm-started from (j2_seed_deg, j3_seed_deg).
    Uses _TIP_SOLVE_DAMPING, much lighter than this module's other solver --
    see module docstring for why."""
    j2, j3 = float(j2_seed_deg), float(j3_seed_deg)
    lam2 = _TIP_SOLVE_DAMPING ** 2
    eps = _TIP_SOLVE_FD_EPS_DEG
    for _ in range(_TIP_SOLVE_MAX_ITERS):
        y0, z0 = _tip_yz(j1_deg, j2, j3, j4_deg, j5_deg, 0.0)
        err = np.array([target_y_mm - y0, target_z_mm - z0])
        if np.hypot(*err) < _TIP_SOLVE_TOL_MM:
            break
        y2, z2 = _tip_yz(j1_deg, j2 + eps, j3, j4_deg, j5_deg, 0.0)
        y3, z3 = _tip_yz(j1_deg, j2, j3 + eps, j4_deg, j5_deg, 0.0)
        J = np.array([[(y2 - y0) / eps, (y3 - y0) / eps],
                      [(z2 - z0) / eps, (z3 - z0) / eps]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(2), err)
        biggest = float(np.max(np.abs(dq)))
        if biggest > _TIP_SOLVE_STEP_CLAMP_DEG:
            dq *= _TIP_SOLVE_STEP_CLAMP_DEG / biggest
        j2 += float(dq[0])
        j3 += float(dq[1])
    return j2, j3


def solve_j6_level(j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, axis_index=0):
    """Closed-form J6 deg that makes rotation-matrix column `axis_index`
    (0 or 1 -- whichever matches the gripper's physical finger-open/width
    direction; flip if the gripper ends up on its side) land horizontal
    (world Z-component = 0). Exact, not approximate, since the pointing
    axis is exactly world +X at this arm's J5=-90 (see module docstring) --
    mirrors _gaze.py's _look_at_matrix world-up projection technique."""
    T = kinematics.forward_kinematics([j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, 0.0])
    y0, z0 = float(T[1, axis_index]), float(T[2, axis_index])
    return math.degrees(math.atan2(-z0, y0))


def build_straight_arm_oscillation_waypoints(base_j1_deg, dip_mm, final_dip_mm,
                                             cycles, elbow_branch_sign, j4_static_deg,
                                             j5_deg, level_axis_index,
                                             n_oscillate_waypoints, n_settle_waypoints):
    """Build the (n_oscillate_waypoints + n_settle_waypoints, 6) absolute
    joint-angle array for the dip-and-rise wag: the TIP's world Z runs
    `cycles` full dips below the reference height (0 -> -dip_mm -> 0,
    `cycles` times -- a (1-cos)/2 shape, since it can only go DOWN from the
    reference, never above), then continues without pausing into an eased
    settle so the very LAST waypoint lands at a final_dip_mm dip instead of
    snapping back to the reference. J4 stays at j4_static_deg throughout
    (not compensating -- see module docstring). The (J2, J3) branch
    ALTERNATES sides each cycle (cycle 0 uses elbow_branch_sign, cycle 1 the
    opposite, ...; the settle phase counts as one more alternation too) --
    at each branch switch (always a dip=0 crossing) the solver's seed is
    deliberately nudged a small step in the intended direction instead of
    warm-started from the literal previous (0, 0) sample, since (0, 0) is
    exactly the Newton solve's singular point and would otherwise default
    to whichever side its null-space happens to favor, not necessarily the
    intended one. J6 is computed via solve_j6_level from the resulting
    (J2, J3, J4) so the gripper stays level throughout. J1, J5 held fixed."""
    n_osc = int(n_oscillate_waypoints)
    n_set = int(n_settle_waypoints)
    cycles = int(cycles)

    _, ref_z_mm = _tip_yz(base_j1_deg, 0.0, 0.0, j4_static_deg, j5_deg, 0.0)

    u1 = np.linspace(0.0, 1.0, n_osc + 1)[1:]
    shape_osc = (1.0 - np.cos(2.0 * np.pi * cycles * u1)) / 2.0
    dip_osc = float(dip_mm) * shape_osc

    cycle_idx = np.clip(np.floor(cycles * u1 - 1e-9).astype(int), 0, cycles - 1)
    branch_osc = float(elbow_branch_sign) * np.where(cycle_idx % 2 == 0, 1.0, -1.0)

    u2 = np.linspace(0.0, 1.0, n_set + 1)[1:]
    dip_settle = dip_osc[-1] + (float(final_dip_mm) - dip_osc[-1]) * _smootherstep(u2)
    branch_settle = np.full(n_set, -branch_osc[-1])

    dip_all = np.concatenate([dip_osc, dip_settle])
    branch_all = np.concatenate([branch_osc, branch_settle])

    j2, j3 = 0.0, 0.0
    prev_branch = None
    rows = []
    for dip, branch in zip(dip_all, branch_all):
        if prev_branch is not None and branch != prev_branch:
            j2, j3 = branch * -1.0, branch * 1.0
        z_target = ref_z_mm - float(dip)
        j2, j3 = solve_tip_point_2link(base_j1_deg, 0.0, z_target, j4_static_deg,
                                       j5_deg, j2, j3)
        j6 = solve_j6_level(base_j1_deg, j2, j3, j4_static_deg, j5_deg, level_axis_index)
        rows.append([base_j1_deg, j2, j3, j4_static_deg, j5_deg, j6])
        prev_branch = branch
    return np.array(rows, dtype=float)


_SPEED_SAFETY_FRACTION = 0.9   # target this fraction of the hard firmware cap,
                               # not 100% of it -- same margin-below-the-wall
                               # spirit as config.JOINT_LIMIT_MARGIN_DEG


def min_feasible_duration_s(q_waypoints, unit_durations, speed_limits_dps=None):
    """Minimum total duration (seconds) for q_waypoints (the implicit start
    row included, so len(q_waypoints) == len(unit_durations) + 1) that keeps
    every joint's implied deg/s within speed_limits_dps (defaults to
    armik.config.MAX_JOINT_SPEED_DPS, shaved by _SPEED_SAFETY_FRACTION) --
    the same hardware caps armik.arm.Arm's plan_path()/_precheck already
    enforce for ordinary Cartesian moves. arm._execute() -- used directly by
    the celebration animation to dodge the straight-up pose's Cartesian
    singularity -- applies NO such check itself, so a dip/cycle/waypoint-
    count combination that demands more speed than the real servo can
    deliver would otherwise go unnoticed (mock has no execution lag to
    reveal it either). unit_durations should be PROPORTIONAL segment
    durations summing to 1 (e.g. celebration_durations(..., duration_s=1.0))
    -- scaling them uniformly by this function's return value is how the
    caller gets an actually-safe timeline."""
    limits = np.asarray(
        speed_limits_dps if speed_limits_dps is not None else config.MAX_JOINT_SPEED_DPS,
        dtype=float) * _SPEED_SAFETY_FRACTION
    q = np.asarray(q_waypoints, dtype=float)
    unit_durations = np.asarray(unit_durations, dtype=float)
    dq = np.abs(np.diff(q, axis=0))                     # (n-1, 6)
    rate_at_unit = dq / unit_durations[:, None]          # deg per unit duration
    required_s_per_joint = rate_at_unit.max(axis=0) / limits
    return float(np.max(required_s_per_joint))
