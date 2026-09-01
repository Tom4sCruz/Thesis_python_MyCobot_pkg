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
# Tool / end-effector offset (TCP)
# ---------------------------------------------------------------------------
#
# Rigid transform from the bare flange (DH frame 6) to the point you actually
# command -- the tip of the gripper. Once nonzero, EVERY Cartesian quantity in
# this package is measured to that point: forward_kinematics, both Jacobians,
# IK, get_coords(), send_coords()/send_path(), and the workspace safety checks.
#
# TOOL_OFFSET_MM : (dx, dy, dz) in the FLANGE frame, millimetres. Frame-6 +z
#   points straight out of the flange face (DH row 6: alpha=0, d6=48.6), so a
#   gripper mounted straight along the wrist axis is a pure +z translation.
# TOOL_RPY_DEG : orientation of the tool (gripper) frame vs the flange, same
#   Euler convention as everything else (EULER_CONVENTION => R = Rz@Ry@Rx).
#   The gripper here is bolted on rolled about the flange axis, so this is a
#   pure z roll: (0, 0, -45). rpy_to_matrix(0, 0, -45) = Rz(-45), which does
#   NOT move the tool z axis, so TOOL_OFFSET_MM stays a straight-out offset.
#   Nonzero here means get_coords()/send_coords() rx/ry/rz describe the GRIPPER
#   frame (a level gripper reads rz ~= 0) and IK works in gripper-frame Euler.
#
# ALL ZERO == bare flange == byte-identical to before this constant existed.
#
# 90 mm and -45 deg are STARTING VALUES. Measure the real flange-face-to-tip
# distance; confirm the roll sign/size on hardware (a level gripper should read
# rz ~= 0 -- if not, try +135 / -45 / +45). DH-table validation is unaffected:
# scripts/verify_fk.py deliberately compares the FLANGE against firmware FK.
#
# Do this ONLY here in Python. Do NOT also call the firmware's
# set_tool_reference()/set_end_type() -- this package bypasses firmware
# kinematics on purpose and doing both would double-count the offset.
TOOL_OFFSET_MM = (0.0, 0.0, 90.0)
TOOL_RPY_DEG = (0, 0, 0) #(0.0, 0.0, -45.0)


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

# Delay AFTER the arm fully reaches one waypoint (every joint needed for it
# has arrived) and BEFORE starting toward the next waypoint, in single-joint
# mode. Separate from SINGLE_JOINT_DELAY, which is the pause between
# individual JOINTS while still reaching a single point -- this is the pause
# between POINTS themselves once one is fully reached.
SINGLE_JOINT_DELAY_BETWEEN_POINTS = 2  # seconds

# ---------------------------------------------------------------------------
# Single-joint workspace safety
# ---------------------------------------------------------------------------

# Full 3D distance from the base origin the tip may reach, in mm. ~280 mm
# matches the myCobot 280's published working radius -- verify against your
# actual arm, same as everything else in this file.
#
# This now bounds the TOOL TIP (see TOOL_OFFSET_MM), not the bare flange. 480 =
# ~280 mm flange radius + ~90 mm tool + margin. Raise it further for a longer
# tool; lower it back toward 280-330 for a bare flange.
MAX_REACH_MM = 480 #280.0

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
MAX_JOINT_SPEED_DPS = [240.0, 240.0, 240.0, 300.0, 300.0, 350.0]

# ---------------------------------------------------------------------------
# Jerk injection -- DELIBERATE jitter (armik/jerk.py; arm.jerk / .random_twitch
# / .twitch_intensity)
# ---------------------------------------------------------------------------
#
# This is the opposite of _min_jerk in arm.py: instead of smoothing a move, it
# corrupts the streamed setpoints so the arm visibly shakes -- for comparing
# smooth vs. jerky motion. All zero-safe: arm.jerk = 0 AND arm.random_twitch = 0
# (or arm.twitch_intensity = 0) -> motion is byte-for-byte unchanged.
#
# arm.jerk is a DIMENSIONLESS roughness dial (0 = smooth, ~1-3 subtle, ~5-10
# violent). It drives two effects, scaled by the two constants below:
JERK_TREMOR_DEG_PER_UNIT = 0.30   # RMS joint deflection (deg) per unit of arm.jerk
JERK_SPEED_FRAC_PER_UNIT = 0.05   # pace-modulation std (fraction) per unit of arm.jerk

# AR(1) correlation for both the tremor noise and the pace modulation. 0 = white
# (buzzy), ->1 = slow sway. ~0.5 reads as a hand tremor at CONTROL_RATE_HZ.
JERK_TREMOR_CORRELATION = 0.5

# Commanded-speed multiplier is clamped to this band after modulation, so a
# perturbed setpoint can neither stall nor bolt.
JERK_SPEED_FACTOR_MIN = 0.30
JERK_SPEED_FACTOR_MAX = 2.50

# Discrete twitch ("flinch") shape, in control ticks.
JERK_TWITCH_RISE_TICKS = 1        # ticks to ramp a twitch to full amplitude
JERK_TWITCH_DECAY_TICKS = 4       # ticks to decay it back to zero

# Hard per-joint cap on the TOTAL perturbation (tremor + twitch), in degrees.
JERK_MAX_DEG = 8.0

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
# Gripper
# ---------------------------------------------------------------------------
# The serial protocol for this arm has NO angular gripper command. pymycobot
# exposes set_gripper_value(value, speed): value 0-100 (0 = fully closed,
# 100 = fully open), speed an integer 1-100. Arm.send_gripper(deg) takes an
# opening ANGLE and maps [0, MAX_GRIPPER_DEG] linearly onto that 0-100 value:
#   send_gripper(0)               -> closed
#   send_gripper(MAX_GRIPPER_DEG) -> fully open
MAX_GRIPPER_DEG = 120.0
GRIPPER_DEFAULT_SPEED = 50   # pymycobot gripper speed, integer 1-100

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
