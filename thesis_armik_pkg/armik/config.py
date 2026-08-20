"""
Robot description and tunable constants for the myCobot 280 (Jetson Nano).

Everything downstream -- FK, the Jacobian, IK, every Cartesian number this
package reports -- is only as correct as the DH table below. Run
scripts/verify_fk.py against your actual arm before trusting any of it.

UNITS AT THIS LAYER: millimetres and degrees. The centimetre conversion happens
only at the public API boundary in arm.py, in exactly two functions.
"""

from __future__ import annotations

import numpy as np

DOF = 6

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

# Standard (distal) Denavit-Hartenberg:
#   A_i = Rz(theta_i + offset_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)
#
# Row format: (theta_offset_deg, d_mm, a_mm, alpha_deg)
DH_TABLE = [
    (0.0, 131.56, 0.0, 90.0),
    (-90.0, 0.0, -110.4, 0.0),
    (0.0, 0.0, -96.0, 0.0),
    (-90.0, 64.62, 0.0, 90.0),
    (90.0, 73.18, 0.0, -90.0),
    (0.0, 48.6, 0.0, 0.0),
]

# Height of the base -- i.e. where joint 1 sits -- in CM.
#
# !!! DO NOT ADD THIS TO THE z YOU GET FROM get_coords() !!!
#
# DH_TABLE row 1 already has d1 = 131.56 mm = 13.156 cm. That places frame 0 at
# the BOTTOM of the base (the table surface) and frame 1 at the joint-1 axis.
# So the z that FK produces is ALREADY measured from the table. Adding this
# constant on top would double-count the base height and put every Cartesian
# target ~13 cm too high.
#
# It exists here for two reasons only:
#   (a) so the number is written down; and
#   (b) if you want z measured from the JOINT 1 AXIS instead of the table,
#       set Z_RELATIVE_TO_JOINT1 = True below.
#
# Measurement note: a tape measure to the top of the base housing gives ~15 cm,
# but the kinematically meaningful number is the distance to the joint-1 axis,
# which the DH table puts at 13.156 cm. Resolve that ~1.8 cm discrepancy with
# scripts/verify_fk.py before trusting Cartesian targets.
JOINT_1_HEIGHT_CM = 13.156

# False -> z measured from the table / base bottom (matches firmware get_coords)
# True  -> z measured from the joint 1 axis
Z_RELATIVE_TO_JOINT1 = False

# Euler convention relating a rotation matrix to (rx, ry, rz).
# "xyz_fixed" means R = Rz(rz) @ Ry(ry) @ Rx(rx).
EULER_CONVENTION = "xyz_fixed"


# ---------------------------------------------------------------------------
# Single-joint motion
# ---------------------------------------------------------------------------

# Delay between one joint reaching its target and the next joint starting.
SINGLE_JOINT_DELAY = 0.15  # seconds

# How close a joint must be to its target before it is considered stopped.
SINGLE_JOINT_TOL_DEG = 2.0

# How often send_path() checks whether the active joint reached its target.
SINGLE_JOINT_POLL_HZ = 25.0

# Max seconds to wait for a single joint to reach its target before giving up.
SINGLE_JOINT_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Single-joint workspace safety
# ---------------------------------------------------------------------------

# Full 3D distance from the base origin the tip may reach, in mm. ~280 mm
# matches the myCobot 280's published working radius -- verify against your
# actual arm, same as everything else in this file.
MAX_REACH_MM = 280.0

# z of the base mounting plate, in mm, in the SAME frame get_coords() uses.
# At or above this height, only the reach sphere is checked.
BASE_Z_MM = 0.0

# z of the physical table/ground, in mm, in the SAME frame get_coords() uses.
# Only enforced when the tip is at or below it.
GROUND_Z_MM = -60.0

# Minimum horizontal distance (sqrt(x^2+y^2)) the tip must keep from the
# vertical axis through the base whenever it's at or below GROUND_Z_MM.
MIN_BASE_DIST_MM = 60.0


# ---------------------------------------------------------------------------
# Joint limits
# ---------------------------------------------------------------------------

JOINT_LIMITS_DEG = [
    (-168.0, 168.0),
    (-135.0, 135.0),
    (-150.0, 150.0),
    (-145.0, 145.0),
    (-165.0, 165.0),
    (-175.0, 175.0),
]

JOINT_LIMIT_MARGIN_DEG = 3.0

# Conservative per-joint ceiling, deg/s. Measure these rather than trusting them.
#MAX_JOINT_SPEED_DPS = [110.0, 110.0, 110.0, 130.0, 130.0, 150.0]
MAX_JOINT_SPEED_DPS = [160.0] * 6

# ---------------------------------------------------------------------------
# pymycobot interface
# ---------------------------------------------------------------------------

DEFAULT_PORT = "/dev/ttyTHS1"
DEFAULT_BAUDRATE = 1000000

# pymycobot's send_angles(angles, speed) takes an INTEGER 0-100 in arbitrary
# units. There is no deg/s field in the serial protocol. This constant is the
# empirical bridge: deg/s of the fastest joint at speed=100. CALIBRATE IT.
DEG_PER_S_AT_SPEED_100 = 120.0
FIRMWARE_SPEED_MIN = 1
FIRMWARE_SPEED_MAX = 100

# Rate at which joint setpoints are streamed along the Cartesian path. The
# serial link realistically tops out around 20-50 Hz on this hardware.
CONTROL_RATE_HZ = 25.0

# Each streamed setpoint is only one tick away, so we ask the firmware for a
# slightly higher speed than strictly needed to make sure the servo arrives
# before the next setpoint rather than lagging the whole trajectory.
STREAM_SPEED_GAIN = 1.6

# ---------------------------------------------------------------------------
# Cartesian API defaults
# ---------------------------------------------------------------------------

DEFAULT_SPEED_CM_S = 5.0
DEFAULT_ROT_SPEED_DEG_S = 30.0

# Path resolution: aim for one IK solve per this much tip travel.
MAX_STEP_MM = 3.0
MIN_WAYPOINTS = 8

# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------

# Position error is in mm, rotation error in degrees. This makes 1 degree of
# rotation error cost the same as 1 mm of position error.
ROT_WEIGHT_MM_PER_DEG = 1.0

# How hard the UNCONSTRAINED coordinates are held at their starting value.
# This is the mechanism behind "change the unmentioned coordinates as little as
# possible": they are soft-anchored to where they started, and only yield when
# the constrained coordinates genuinely need them to.
#
# Tuned empirically: on a 3 cm z-only move, 0.08 lets rx drift ~11 deg, while
# 2.0 holds it to ~0.3 deg with no loss of reachability. Raise for tighter
# holding; lower if a reachable target starts being reported unreachable
# because the anchors are fighting the constraints.
FREE_ANCHOR_WEIGHT = 2.0

IK_MAX_ITERS = 120
# Levenberg-Marquardt damping. Higher = more stable near singularities but
# slower to converge. Tuned empirically: at 3.0 a plain 6-DOF move needed ~129
# iterations (exceeding IK_MAX_ITERS and reporting a false "unreachable"); at
# 1.0 the same move converges in ~16 iterations, and across a spread of seed
# configurations including near-singular ones it solves 13/16 test targets
# versus 3/16 at 3.0 -- with the remaining 3 being genuinely out of reach
# (they still fail at 300 iterations).
IK_DAMPING = 1.0
IK_STEP_CLAMP_DEG = 5.0
POS_TOL_MM = 0.5
ROT_TOL_DEG = 0.5

# Euler angles degenerate as the middle angle approaches +/-90 deg.
GIMBAL_WARN_DEG = 3.0

COORD_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def joint_limits_array() -> np.ndarray:
    """(6, 2) array of SOFT joint limits in degrees."""
    lim = np.array(JOINT_LIMITS_DEG, dtype=float)
    lim[:, 0] += JOINT_LIMIT_MARGIN_DEG
    lim[:, 1] -= JOINT_LIMIT_MARGIN_DEG
    return lim
