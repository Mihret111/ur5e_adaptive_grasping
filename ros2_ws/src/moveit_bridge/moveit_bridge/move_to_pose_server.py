import traceback

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
        super().__init__("move_to_pose_server")

        self.get_logger().info("Initializing MoveItPy...")

        # This creates a MoveItPy interface inside this node/process.
        # It reads the robot description/planning configuration from the
        # running ROS/MoveIt environment.
        self.moveit = MoveItPy(node_name="b2b_moveit_py")

        # Default planning group for Universal Robots MoveIt config.
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