#!/usr/bin/env python3

import json
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from moveit_interfaces.action import MoveToPose

# Class to send MoveIt goal from JSON file
class MoveToPoseJsonClient(Node):
    # Initialize the node
    def __init__(self):
        super().__init__("move_to_pose_json_client")
        self._client = ActionClient(self, MoveToPose, "move_to_pose")

    # Send MoveIt goal from JSON file
    def send_goal_from_file(self, json_path):
        # Open the JSON file
        with open(json_path, "r") as f:
            command = json.load(f)

        goal_cfg = command["moveit_goal"]

        # Create a PoseStamped message
        pose = PoseStamped() 
        pose.header.frame_id = goal_cfg["frame_id"]         # Set the frame_id

        # Set the position and orientation
        pose.pose.position.x = float(goal_cfg["position"]["x"])
        pose.pose.position.y = float(goal_cfg["position"]["y"])
        pose.pose.position.z = float(goal_cfg["position"]["z"])

        pose.pose.orientation.x = float(goal_cfg["orientation"]["x"])
        pose.pose.orientation.y = float(goal_cfg["orientation"]["y"])
        pose.pose.orientation.z = float(goal_cfg["orientation"]["z"])
        pose.pose.orientation.w = float(goal_cfg["orientation"]["w"])

        # Create a MoveToPose goal
        goal = MoveToPose.Goal()
        goal.target_pose = pose
        goal.planning_group = goal_cfg.get("planning_group", "ur_manipulator")
        goal.end_effector_link = goal_cfg.get("end_effector_link", "tool0")
        goal.velocity_scaling = float(goal_cfg.get("velocity_scaling", 0.2))
        goal.acceleration_scaling = float(goal_cfg.get("acceleration_scaling", 0.2))

        # Wait for action server
        self.get_logger().info(f"Waiting for /move_to_pose action server...")
        self._client.wait_for_server()

        self.get_logger().info(
            "Sending MoveToPose goal from JSON: "
            f"x={pose.pose.position.x:.3f}, "
            f"y={pose.pose.position.y:.3f}, "
            f"z={pose.pose.position.z:.3f}"
        )

        # send goal asynchronously
        send_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )

        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        # Check if goal was accepted
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected.")
            return False

        self.get_logger().info("Goal accepted.")
        # Get the result from the goal handle

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result

        self.get_logger().info(f"Result success: {result.success}")
        self.get_logger().info(f"Result message: {result.message}")

        return result.success

    # 
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"Feedback: {feedback.phase} ({feedback.progress:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)

    node = MoveToPoseJsonClient()

    json_path = "/tmp/cogar_b2b/target_command.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    try:
        node.send_goal_from_file(json_path)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()