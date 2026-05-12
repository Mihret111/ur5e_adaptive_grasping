from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    use_sim_time = LaunchConfiguration("use_sim_time")

    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Universal Robots model type.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time.",
        ),
    ]

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(
            Path("srdf") / "ur.srdf.xacro",
            {"name": ur_type},
        )
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl", "pilz_industrial_motion_planner", "chomp"],
        )
        .to_moveit_configs()
    )
    print("MOVEIT CONFIG KEYS:", moveit_config.to_dict().keys())
    print("PLANNING PIPELINES:", moveit_config.planning_pipelines)

    move_to_pose_server = Node(
        package="moveit_bridge",
        executable="move_to_pose_server",
        name="move_to_pose_server",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription(
        declared_arguments + [
            move_to_pose_server,
        ]
    )