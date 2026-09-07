"""
_rviz_bridge.py -- stream a mock (no-hardware) armik run into RViz2.

Used only by the movement-profile scripts in this directory when they are run
with --rviz (which implies --mock). On the ROS 2 graph it publishes:

  * sensor_msgs/JointState on /joint_states -- the live simulated joint pose,
    polled from Arm.get_angles() at ~30 Hz (degrees -> radians), named for the
    mycobot_280_jn adaptive-gripper URDF so robot_state_publisher animates the
    RobotModel. When a gripper_source is given, a 7th joint `gripper_controller`
    is published too (mapped from the mock's 0..100 gripper value) so the finger
    links have a transform and open/close; robot_state_publisher derives the
    five <mimic> finger joints from it.
  * visualization_msgs/Marker on /visualization_marker -- a LINE_STRIP of the
    whole precomputed Cartesian path, a CUBE_LIST of green 3.5 cm cubes at the
    cube start positions, and (optionally) a live SPHERE at the tool tip.

armik stays ROS-free: this module is the only place rclpy is imported, the
import is guarded, and nothing here runs unless a profile script asks for it.

RUN CONTEXT
-----------
rclpy exists only inside the Study-docker 'hri_thesis' container (ROS 2 Humble).
On the bare host, start() raises RuntimeError with a pointer to the container and
the profile run continues without visualization.

FRAMES / UNITS
--------------
armik joint angles are DEGREES, index 0..5 == J1..J6; the URDF wants RADIANS in
the URDF_JOINT_NAMES order below -- a per-joint math.radians() is the whole
conversion (see mycobot_ros2 follow_display.py, which feeds get_radians()
straight through in this same order).

armik Cartesian coords are CENTIMETRES, tool tip, arm base frame, z from the
table (config.Z_RELATIVE_TO_JOINT1 == False). RViz's fixed frame is `joint1`,
the ROOT link of the JN URDF (no `g_base`). armik's DH base frame is
AXIS-ALIGNED with `joint1` (verified numerically: armik +X/+Y/+Z map straight to
joint1 +X/+Y/+Z), so the mapping is an identity + cm->m scale:
    x_j1 = x_cm / 100 ,  y_j1 = y_cm / 100 ,  z_j1 = z_cm / 100
(An earlier version copied follow_display.py's `-y, x` swap -- that is for the
FIRMWARE's get_coords(), whose frame is yawed +90 deg from the URDF; applying it
to armik's kinematics.pose_coords() double-rotated the path.)

CALIBRATION
-----------
The URDF puts J1->J2 at 157.56 mm vs armik's d1 = 131.56 mm, so there is a
residual ~2-3 cm vertical difference at flange height (less near the table). The
live green tip sphere is drawn from Arm.get_coords() through the SAME mapping as
the path line: if it rides the URDF gripper tip the mapping is right; if it sits
consistently above/below, put the negated offset (metres, joint1 frame) in
PATH_FRAME_OFFSET_M below -- the path + cube markers move with it.
"""

from __future__ import annotations

import math
import threading
import time

# --- guarded ROS 2 import -----------------------------------------------------
_RCLPY_IMPORT_ERROR = None
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from builtin_interfaces.msg import Duration as _DurationMsg
    from geometry_msgs.msg import Point as _Point
    from sensor_msgs.msg import JointState as _JointState
    from std_msgs.msg import ColorRGBA as _ColorRGBA
    from visualization_msgs.msg import Marker as _Marker
except Exception as exc:                       # noqa: BLE001 -- any import failure
    _RCLPY_IMPORT_ERROR = exc


# The 6 arm revolute joints, in FK / chain order == armik joint index 0..5.
# Same names in the m5 and the mycobot_280_jn adaptive-gripper URDFs.
URDF_JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]

# The single actuated finger joint of the adaptive gripper. robot_state_publisher
# derives the five <mimic> finger joints from it, so we publish only this one.
GRIPPER_JOINT_NAME = "gripper_controller"
# URDF <limit lower="-0.74" upper="0.15"> for gripper_controller. Which end is
# "open" is a guess -- if the fingers move the wrong way, swap these two.
GRIPPER_OPEN_RAD = 0.15
GRIPPER_CLOSED_RAD = -0.74

MARKER_FRAME = "joint1"

# metres, joint1 frame -- see CALIBRATION above. (0, 0, 0) until measured.
PATH_FRAME_OFFSET_M = (0.0, 0.0, 0.0)

# DISPLAY-ONLY joint slew cap (deg/s) for the NON-sequential path (HighH-LowR,
# HighH-LowR_claude). Their streamed motion peaks ~123 deg/s, so 150 lets it
# through untouched while ramping any instantaneous jump. 0 disables (raw
# pass-through). Not used when sequential_joints=True (see VIZ_SEQ_MOVE_S).
VIZ_JOINT_SLEW_DPS = 150.0

# sequential_joints=True (LowH-HighR): wall-time each joint takes to animate to
# its next value, and each is shown moving alone. Keep it < LowH-HighR's
# RVIZ_JOINT_MOVE_S so the animation finishes before the mock sends the next
# joint. Change the two together to speed up / slow down that visualization.
VIZ_SEQ_MOVE_S = 0.8
_SEQ_TOL_RAD = math.radians(0.5)   # "this joint changed" threshold

# Marker colours as (r, g, b, a); the 4th value is opacity, 0 = invisible,
# 1 = solid. This is the one place to tune path / cube transparency.
PATH_COLOR_RGBA = (0.00, 0.90, 1.00, 0.55)   # cyan, semi-transparent
CUBE_COLOR_RGBA = (0.10, 0.75, 0.20, 0.35)   # green, semi-transparent
TIP_COLOR_RGBA = (0.10, 1.00, 0.20, 0.90)    # green, near-solid


def _cm_xyz_to_frame(x_cm, y_cm, z_cm):
    """armik base-frame centimetres -> RViz `joint1`-frame metres (identity + offset)."""
    ox, oy, oz = PATH_FRAME_OFFSET_M
    return (float(x_cm) / 100.0 + ox,
            float(y_cm) / 100.0 + oy,
            float(z_cm) / 100.0 + oz)


def _grip_val_to_rad(value):
    """pymycobot gripper value (0 closed .. 100 open) -> gripper_controller radians."""
    frac = max(0.0, min(1.0, float(value) / 100.0))
    return GRIPPER_CLOSED_RAD + frac * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


def _rgba(t):
    """(r, g, b, a) tuple -> std_msgs/ColorRGBA (rosidl msgs are keyword-only)."""
    return _ColorRGBA(r=float(t[0]), g=float(t[1]), b=float(t[2]), a=float(t[3]))


class RvizBridge:
    """Publishes /joint_states + /visualization_marker for a mock profile run.

    joint_source  : zero-arg callable -> length-6 sequence of joint angles in
                    DEGREES (pass ``arm.get_angles``).
    path_xyz_cm   : iterable of (x, y, z) tuples in CM -- the full precomputed
                    Cartesian path, already concatenated by the caller.
    cube_points   : iterable of (x, y, z) tuples in CM -- cube start positions.
    tip_source    : optional zero-arg callable -> ``arm.get_coords()`` (cm); when
                    given, a live green sphere is drawn at the tool tip.
    gripper_source: optional zero-arg callable -> gripper value 0..100 (pass
                    ``arm.get_gripper_value``); when given, `gripper_controller`
                    is published so the adaptive-gripper fingers render and move.
    sequential_joints: True for single-joint profiles (LowH-HighR) -- animate
                    exactly one arm joint at a time, each over VIZ_SEQ_MOVE_S,
                    instead of the deg/s slew. Default False.
    """

    def __init__(self, joint_source, path_xyz_cm=None, cube_points=None,
                 tip_source=None, gripper_source=None, sequential_joints=False,
                 node_name="thesis_mock_rviz_bridge",
                 joint_rate_hz=30.0, marker_period_s=2.0):
        self._joint_source = joint_source
        self._tip_source = tip_source
        self._gripper_source = gripper_source
        self._sequential = bool(sequential_joints)
        self._path_xyz_cm = [tuple(p) for p in path_xyz_cm] if path_xyz_cm else []
        self._cube_points = [tuple(p) for p in cube_points] if cube_points else []
        self._node_name = node_name
        self._joint_period = 1.0 / float(joint_rate_hz)
        self._marker_period = float(marker_period_s)

        self._node = None
        self._executor = None
        self._thread = None
        self._joint_pub = None
        self._marker_pub = None
        self._joint_msg = None
        self._line_marker = None
        self._cube_marker = None
        self._owns_rclpy = False
        self._started = False
        self._stopped = False
        self._grip_pub = gripper_source is not None
        self._last_grip_rad = _grip_val_to_rad(0)   # mock starts closed (value 0)
        self._disp_q = None                         # slew-filtered displayed pose
        self._last_tick_t = 0.0
        # sequential mode: one active arm joint animated over VIZ_SEQ_MOVE_S
        self._active = None
        self._active_from = 0.0
        self._active_target = 0.0
        self._active_t0 = 0.0
        self._grip_from = None
        self._grip_target = 0.0
        self._grip_t0 = 0.0

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        if _RCLPY_IMPORT_ERROR is not None:
            raise RuntimeError(
                "--rviz needs rclpy (ROS 2), which is not importable here. Run "
                "the script inside the Study-docker 'hri_thesis' container with "
                "/opt/ros/humble and the mycobot_ros2 overlay sourced. "
                f"Original import error: {_RCLPY_IMPORT_ERROR!r}"
            )
        if self._started:
            return
        self._started = True

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        self._node = Node(self._node_name)
        self._joint_pub = self._node.create_publisher(_JointState, "joint_states", 10)
        self._marker_pub = self._node.create_publisher(_Marker, "visualization_marker", 10)

        self._joint_msg = _JointState()
        self._joint_msg.name = list(URDF_JOINT_NAMES)
        if self._grip_pub:
            self._joint_msg.name.append(GRIPPER_JOINT_NAME)
        self._joint_msg.velocity = []
        self._joint_msg.effort = []

        self._line_marker = self._build_line_marker()
        self._cube_marker = self._build_cube_marker()

        # One publish while this is still the only thread -> no cross-thread
        # races on the publishers; the spin thread takes over from here.
        self._publish_markers()

        self._node.create_timer(self._joint_period, self._on_joint_timer)
        self._node.create_timer(self._marker_period, self._publish_markers)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name=self._node_name,
                                        daemon=True)
        self._thread.start()

    def _spin(self):
        try:
            self._executor.spin()
        except Exception:                     # noqa: BLE001 -- shutdown races
            pass

    def stop(self):
        """Idempotent, never raises (called from a finally: block)."""
        if not self._started or self._stopped:
            return
        self._stopped = True
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:              # noqa: BLE001
            print(f"  (rviz bridge shutdown: {exc})")

    # -- optional: redraw the path after a mid-run mutation ----------------

    def update_path(self, path_xyz_cm):
        """Rebuild the LINE_STRIP (e.g. after HighH-LowR's nudge rewrites a
        segment). The next marker tick publishes the new line."""
        self._path_xyz_cm = [tuple(p) for p in path_xyz_cm] if path_xyz_cm else []
        if self._node is not None:
            self._line_marker = self._build_line_marker()

    # -- publishing -------------------------------------------------------

    def _on_joint_timer(self):
        try:
            q_deg = list(self._joint_source())
        except Exception:                     # noqa: BLE001 -- never kill the timer
            return
        if len(q_deg) < 6:
            return
        msg = self._joint_msg
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.position = [math.radians(float(v)) for v in q_deg[:6]]
        if self._grip_pub:
            try:
                v = self._gripper_source()
                if v is not None:
                    self._last_grip_rad = _grip_val_to_rad(v)
            except Exception:                 # noqa: BLE001 -- keep last known
                pass
            msg.position.append(self._last_grip_rad)   # name has 7 -> position 7
        msg.position = self._slew(msg.position)
        try:
            self._joint_pub.publish(msg)
        except Exception:                     # noqa: BLE001
            pass

    def _slew(self, target):
        """Display-only: ease the published pose toward `target` so the mock's
        instantaneous jumps render as visible motion. Executor-thread only."""
        now = time.monotonic()
        if self._disp_q is None or len(self._disp_q) != len(target):
            self._disp_q = list(target)
            self._last_tick_t = now
            self._active = None
            self._grip_from = None
            return list(self._disp_q)
        if self._sequential:
            return self._slew_sequential(target, now)
        if VIZ_JOINT_SLEW_DPS <= 0.0:
            self._disp_q = list(target)
            self._last_tick_t = now
            return list(self._disp_q)
        dt = now - self._last_tick_t
        self._last_tick_t = now
        max_step = math.radians(VIZ_JOINT_SLEW_DPS) * max(dt, 1e-3)
        out = []
        for cur, tgt in zip(self._disp_q, target):
            d = tgt - cur
            out.append(tgt if abs(d) <= max_step else cur + math.copysign(max_step, d))
        self._disp_q = out
        return list(out)

    def _slew_sequential(self, target, now):
        """Animate exactly one arm joint at a time, each over VIZ_SEQ_MOVE_S; the
        gripper (index 6) animates concurrently over the same duration."""
        dur = max(VIZ_SEQ_MOVE_S, 1e-3)
        n_arm = min(6, len(target))

        # gripper: independent fixed-duration lerp toward its target
        if len(target) > 6:
            gt = target[6]
            if self._grip_from is None or abs(gt - self._grip_target) > _SEQ_TOL_RAD:
                self._grip_from = self._disp_q[6]
                self._grip_target = gt
                self._grip_t0 = now
            f = min(1.0, (now - self._grip_t0) / dur)
            self._disp_q[6] = self._grip_from + f * (self._grip_target - self._grip_from)

        # arm: latch the lowest changed joint, animate it alone to completion
        if self._active is None:
            for i in range(n_arm):
                if abs(target[i] - self._disp_q[i]) > _SEQ_TOL_RAD:
                    self._active = i
                    self._active_from = self._disp_q[i]
                    self._active_target = target[i]
                    self._active_t0 = now
                    break
        if self._active is not None:
            i = self._active
            if abs(target[i] - self._active_target) > _SEQ_TOL_RAD:
                self._active_target = target[i]      # re-commanded mid-move (rare)
            f = (now - self._active_t0) / dur
            if f >= 1.0:
                self._disp_q[i] = self._active_target
                self._active = None
            else:
                self._disp_q[i] = self._active_from + f * (self._active_target - self._active_from)
        return list(self._disp_q)

    def _publish_markers(self):
        try:
            now = self._node.get_clock().now().to_msg()
            if self._line_marker is not None:
                self._line_marker.header.stamp = now
                self._marker_pub.publish(self._line_marker)
            if self._cube_marker is not None:
                self._cube_marker.header.stamp = now
                self._marker_pub.publish(self._cube_marker)
            if self._tip_source is not None:
                tip = self._tip_marker(now)
                if tip is not None:
                    self._marker_pub.publish(tip)
        except Exception:                     # noqa: BLE001
            pass

    # -- marker construction --------------------------------------------

    @staticmethod
    def _point(xyz_cm):
        x, y, z = _cm_xyz_to_frame(*xyz_cm[:3])
        return _Point(x=x, y=y, z=z)

    def _build_line_marker(self):
        if len(self._path_xyz_cm) < 2:
            return None
        m = _Marker()
        m.header.frame_id = MARKER_FRAME
        m.ns = "thesis_path"
        m.id = 0
        m.type = _Marker.LINE_STRIP
        m.action = _Marker.ADD
        m.scale.x = 0.004
        m.color = _rgba(PATH_COLOR_RGBA)
        m.pose.orientation.w = 1.0
        m.points = [self._point(p) for p in self._path_xyz_cm]
        m.lifetime = _DurationMsg(sec=0, nanosec=0)
        return m

    def _build_cube_marker(self):
        if not self._cube_points:
            return None
        m = _Marker()
        m.header.frame_id = MARKER_FRAME
        m.ns = "thesis_cubes"
        m.id = 1
        m.type = _Marker.CUBE_LIST
        m.action = _Marker.ADD
        # 3.5 cm cube edge -- the real cubes; CUBE_LIST centres one on each point.
        m.scale.x = m.scale.y = m.scale.z = 0.035
        m.color = _rgba(CUBE_COLOR_RGBA)
        m.pose.orientation.w = 1.0
        m.points = [self._point(p) for p in self._cube_points]
        m.lifetime = _DurationMsg(sec=0, nanosec=0)
        return m

    def _tip_marker(self, stamp):
        try:
            c = self._tip_source()
        except Exception:                    # noqa: BLE001
            return None
        if not c or len(c) < 3:
            return None
        m = _Marker()
        m.header.frame_id = MARKER_FRAME
        m.header.stamp = stamp
        m.ns = "thesis_tip"
        m.id = 2
        m.type = _Marker.SPHERE
        m.action = _Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.03
        m.color = _rgba(TIP_COLOR_RGBA)
        x, y, z = _cm_xyz_to_frame(c[0], c[1], c[2])
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.pose.orientation.w = 1.0
        m.lifetime = _DurationMsg(sec=0, nanosec=0)
        return m
