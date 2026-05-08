# modules/config_validator.py

def _require(config, key_path):
    """
    Check that a nested key exists.... Eg: require(config, "paths.robot.flange_prim")
    """
    current = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config key: {key_path}")
        current = current[key]
    return current


def validate_config(config, table_materials, table_seat_slots):
    """
    Validate the minimum configuration contract required by SceneBuilder
    """

    required_keys = [
        # environment
        "num_trials",
        "build_room",
        "room_size",
        "room_wall_thickness",
        "floor_color",
        "room_color",
        "table_size",
        "table_leg_radius",
        "arm_min_reach",
        "arm_max_reach",
        "reach_safety_margin",
        "near_edge_padding",

        # objects.yaml
        "num_objects_range",
        "object_margin",
        "object_spacing_padding",
        "object_mass.min_kg",
        "object_mass.max_kg",
        "grip_range_mm.min",
        "grip_range_mm.max",
        "shapes",
        "colors",
        "object_physics_materials",

        # gripper.yaml
        "gripper_friction.static_friction",
        "gripper_friction.dynamic_friction",
        "gripper_friction.restitution",
        "gripper_solver_position_iterations",
        "gripper_solver_velocity_iterations",
        "min_grip_force",
        "max_grip_force",

        # paths.yaml
        "paths.robot.ur5e_base_link",
        "paths.robot.ur5e_joints_base",
        "paths.robot.flange_prim",
        "paths.gripper.left_finger_link",
        "paths.gripper.right_finger_link",
        "paths.scene_spawn.root",
    ]

    for key in required_keys:
        _require(config, key)

    if not table_materials:
        raise ValueError("table_materials is empty. Check table.yaml -> materials")

    if not table_seat_slots:
        raise ValueError("table_seat_slots is empty. Check table.yaml -> seat_slots")

    print("  [ConfigValidator] Configuration contract OK")
    print(f"  [ConfigValidator] table materials: {len(table_materials)}")
    print(f"  [ConfigValidator] table seat slots: {len(table_seat_slots)}")