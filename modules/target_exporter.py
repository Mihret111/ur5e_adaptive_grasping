# modules/target_exporter.py

import json
import os
from datetime import datetime


def export_moveit_target_command(
    output_path,
    target,
    target_local_base,
    hover_height=0.40,
):
    """
    Export a MoveIt target command generated from the Isaac SceneBuilder target.

    This keeps Isaac and ROS2 separated:
      Isaac writes the intended command.
      ROS2 reads it and sends the action goal.

    target_local_base:
      object position expressed in Isaac UR5e base frame.
    """

    x = float(target_local_base[0])
    y = float(target_local_base[1])
    z = float(target_local_base[2]) + float(hover_height)

    command = {
        "created_at": datetime.now().isoformat(),
        "source": "isaac_scene_builder",

        "target_object": {
            "label": target.get("label"),
            "shape": target.get("shape"),
            "material": target.get("material_name"),
            "world_pos": list(target.get("world_pos", [])),
            "prim_path": target.get("prim_path"),
        },

        "moveit_goal": {
            "frame_id": "base_link",
            "planning_group": "ur_manipulator",
            "end_effector_link": "tool0",

            "position": {
                "x": x,
                "y": y,
                "z": z,
            },

            # Temporary orientation:
            # keeps approximately the same downward tool convention we tested.
            "orientation": {
                "x": -0.707,
                "y": 0.001,
                "z": 0.001,
                "w": 0.707,
            },

            "velocity_scaling": 0.2,
            "acceleration_scaling": 0.2,
        },

        "assumptions": {
            "frame_mapping": "Isaac target transformed into Isaac UR5e base frame and treated as MoveIt base_link frame.",
            "robot_state_sync": "Isaac robot state and MoveIt mock hardware state are not yet synchronized.",
            "tcp": "MoveIt tool0 is used; 2FG7 grasp-center TCP is not yet modeled.",
            "purpose": "arm-only hover target test, not full grasp validation.",
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(command, f, indent=2)

    print(f"[TargetExporter] Wrote MoveIt target command to: {output_path}")
    print(
        "[TargetExporter] Hover goal: "
        f"x={x:.3f}, y={y:.3f}, z={z:.3f}"
    )

    return command