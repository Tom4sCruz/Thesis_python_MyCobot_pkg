"""
mock_display.launch.py -- RViz2 + robot_state_publisher for a mock armik run.

Deliberately does NOT start joint_state_publisher: the profile script's --rviz
bridge (scripts/Profiles/_rviz_bridge.RvizBridge) is the sole publisher of
/joint_states. The vendor mycobot_280 test.launch.py would start a second
/joint_states publisher (its gui:=false branch) and the RobotModel would jump
between the two.

Defaults are self-contained files in this repo:
  * model     -> ros/urdf/mycobot_280_jn_adaptive_gripper.urdf  (MyCobot 280 +
                 Jetson Nano + adaptive gripper -- matches the real arm; a
                 vendored copy with one upstream XML typo fixed)
  * rvizconfig -> ros/rviz/mock_display.rviz  (Marker display already bound to
                 /visualization_marker so the path + cubes show with no clicks)

Only `mycobot_description` must be colcon-built + sourced (the URDF's meshes are
package://mycobot_description/... ). Inside the Study-docker 'hri_thesis'
container:

    source /opt/ros/humble/setup.bash
    cd /ros2_ws
    colcon build --symlink-install --packages-select mycobot_description
    source install/setup.bash
    ros2 launch /thesis/thesis_armik_pkg/ros/launch/mock_display.launch.py

then, in a second sourced shell:

    cd /thesis/thesis_armik_pkg
    python3 scripts/Profiles/HighH-LowR.py --mock --yes --rviz
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

_ROS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def generate_launch_description():
    default_model = os.path.join(_ROS_DIR, "urdf", "mycobot_280_jn_adaptive_gripper.urdf")
    default_rviz = os.path.join(_ROS_DIR, "rviz", "mock_display.rviz")

    model_arg = DeclareLaunchArgument("model", default_value=default_model)
    rviz_arg = DeclareLaunchArgument("rvizconfig", default_value=default_rviz)

    # The URDF carries the xacro namespace + one unused <xacro:property>, so it
    # is expanded through xacro (matches the vendor launches).
    robot_description = ParameterValue(
        Command(["xacro ", LaunchConfiguration("model")]), value_type=str
    )

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rvizconfig")],
    )

    return LaunchDescription([model_arg, rviz_arg, rsp_node, rviz_node])
