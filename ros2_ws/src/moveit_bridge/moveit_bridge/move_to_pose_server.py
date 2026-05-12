from launch.actions import reset_launch_configurations
import traceback

from pathlib import Path
from moveit_configs_utils import MoveItConfigsBuilder

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from moveit_interfaces.action import MoveToPose

from moveit.planning import MoveItPy


class MoveToPoseServer(Node):
    """
    Action server that receives an end-effector pose goal and uses MoveItPy
    to plan and execute a UR5e trajectory.

    Role:
      - This node is the main node acting as the bridge for arm-motion.
      - Isaac/TrialRunner will not import MoveIt directly.
      - High-level grasp actions will call this action repeatedly.
    """

    def __init__(self):
        super().__init__("moveit_py")

        self.get_logger().info("Initializing MoveItPy...")

        # This creates a MoveItPy interface inside this node/process.
        # It reads the robot description/planning configuration from the
        # running ROS/MoveIt environment.


        moveit_config = (
            MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
            .robot_description_semantic(
                Path("srdf") / "ur.srdf.xacro",
                {"name": "ur5e"},
            )
            .planning_pipelines(
                default_planning_pipeline="ompl",
                pipelines=["ompl", "pilz_industrial_motion_planner", "chomp"],
            )
            .to_moveit_configs()
        )
        # This line MUST come before using config_dict
        config_dict = moveit_config.to_dict()

        ## ------------------------------------------------------------------
        # MoveItPy expects planning_pipelines.pipeline_names, while the UR
        # MoveIt config builder gives planning_pipelines as a plain list.
        # This adapter makes the UR config compatible with MoveItPy.
        # ------------------------------------------------------------------
        pipeline_names = config_dict.get(
            "planning_pipelines",
            ["ompl", "pilz_industrial_motion_planner", "chomp"],
        )

        if isinstance(pipeline_names, list):
            config_dict["planning_pipelines"] = {
                "pipeline_names": pipeline_names
            }

        # Default single-pipeline planning parameters for MoveItPy.
        # These are used when planning_component.plan() is called without
        # explicit request parameters.
        config_dict["plan_request_params"] = {
            "planning_attempts": 1,
            "planning_pipeline": "ompl",
            "planner_id": "RRTConnectkConfigDefault",
            "max_velocity_scaling_factor": 0.2,
            "max_acceleration_scaling_factor": 0.2,
            "planning_time": 5.0,
        }

        # Optional named planner configs for later use.
        config_dict["ompl_rrtc"] = {
            "plan_request_params": {
                "planning_attempts": 1,
                "planning_pipeline": "ompl",
                "planner_id": "RRTConnectkConfigDefault",
                "max_velocity_scaling_factor": 0.2,
                "max_acceleration_scaling_factor": 0.2,
                "planning_time": 5.0,
            }
        }

        config_dict["pilz_ptp"] = {
            "plan_request_params": {
                "planning_attempts": 1,
                "planning_pipeline": "pilz_industrial_motion_planner",
                "planner_id": "PTP",
                "max_velocity_scaling_factor": 0.2,
                "max_acceleration_scaling_factor": 0.2,
                "planning_time": 5.0,
            }
        }

        config_dict["chomp_default"] = {
            "plan_request_params": {
                "planning_attempts": 1,
                "planning_pipeline": "chomp",
                "max_velocity_scaling_factor": 0.2,
                "max_acceleration_scaling_factor": 0.2,
                "planning_time": 5.0,
            }
        }

        self.moveit = MoveItPy(
            node_name="moveit_py",
            config_dict=config_dict,
        )

        ## Default planning group for Universal Robots MoveIt config.
        self.default_planning_group = "ur_manipulator"

        self.planning_component = self.moveit.get_planning_component(
            self.default_planning_group
        )

        self.get_logger().info(
            f"MoveItPy ready. Default planning group: {self.default_planning_group}"
        )

        self._server = ActionServer(
            self,
            MoveToPose,
            "move_to_pose",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("MoveToPose action server is ready.")

    def goal_callback(self, goal_request):
        self.get_logger().info("Received MoveToPose goal request")

        if goal_request.planning_group and goal_request.planning_group != self.default_planning_group:
            self.get_logger().warn(
                f"Requested planning_group='{goal_request.planning_group}', "
                f"but this first version supports only '{self.default_planning_group}'."
            )
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn("Cancel request received.")
        # For this first implementation, we accept cancellation requests.
        # Mid-execution stopping will be improved later.
        return CancelResponse.ACCEPT

    def _publish_feedback(self, goal_handle, phase, progress):
        feedback = MoveToPose.Feedback()
        feedback.phase = phase
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(f"[Feedback] {phase} ({progress:.2f})")

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        result = MoveToPose.Result()

        target_pose = goal.target_pose

        # Use goal values if provided, otherwise safe defaults.
        planning_group = goal.planning_group or self.default_planning_group
        end_effector_link = goal.end_effector_link or "tool0"

        self.get_logger().info("Accepted MoveToPose goal")
        self.get_logger().info(f"  planning_group: {planning_group}")
        self.get_logger().info(f"  end_effector_link: {end_effector_link}")
        self.get_logger().info(f"  frame_id: {target_pose.header.frame_id}")
        self.get_logger().info(
            "  target position: "
            f"x={target_pose.pose.position.x:.3f}, "
            f"y={target_pose.pose.position.y:.3f}, "
            f"z={target_pose.pose.position.z:.3f}"
        )
        self.get_logger().info(
            "  target orientation: "
            f"x={target_pose.pose.orientation.x:.3f}, "
            f"y={target_pose.pose.orientation.y:.3f}, "
            f"z={target_pose.pose.orientation.z:.3f}, "
            f"w={target_pose.pose.orientation.w:.3f}"
        )

        try:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "Goal canceled before planning."
                return result

            self._publish_feedback(goal_handle, "setting_start_state", 0.10)

            # Start from the currently reported robot state.
            self.planning_component.set_start_state_to_current_state()

            self._publish_feedback(goal_handle, "setting_pose_goal", 0.20)

            # Set the end-effector pose goal.
            self.planning_component.set_goal_state(
                pose_stamped_msg=target_pose,
                pose_link=end_effector_link,
            )

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "Goal canceled before planning."
                return result

            self._publish_feedback(goal_handle, "planning", 0.45)

            plan_result = self.planning_component.plan()

            if not plan_result:
                goal_handle.abort()
                result.success = False
                result.message = "MoveIt planning failed."
                self.get_logger().error(result.message)
                return result

            self._publish_feedback(goal_handle, "executing", 0.75)

            robot_trajectory = plan_result.trajectory

            # Empty controllers means use MoveIt's configured controller manager
            self.moveit.execute(robot_trajectory, controllers=[])

            self._publish_feedback(goal_handle, "done", 1.00)

            goal_handle.succeed()
            result.success = True
            result.message = "MoveIt plan and execution completed successfully."
            return result

        except Exception as e:
            self.get_logger().error(f"Exception in MoveToPose action: {e}")
            self.get_logger().error(traceback.format_exc())

            goal_handle.abort()
            result.success = False
            result.message = f"Exception: {e}"
            return result


def main(args=None):
    rclpy.init(args=args)

    node = MoveToPoseServer()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()