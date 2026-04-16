import os
import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

SCRIPT_DIR = "/home/gian_galv/Documents/isaacsim_ws/mir_ur_station/usd/robots/meshes/gripper_2fg7"
URDF_PATH = os.path.join(SCRIPT_DIR, "2fg7.urdf")

import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = True
import_config.make_default_prim = True
import_config.self_collision = False
import_config.distance_scale = 1.0
import_config.density = 0.0
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.default_drive_strength = 1e6
import_config.default_position_drive_damping = 1e4
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

result, robot_model = omni.kit.commands.execute(
    "URDFParseFile",
    urdf_path=URDF_PATH,
    import_config=import_config
)

for joint_name in robot_model.joints:
    robot_model.joints[joint_name].drive.strength = 1e6
    robot_model.joints[joint_name].drive.damping = 1e4

result, prim_path = omni.kit.commands.execute(
    "URDFImportRobot",
    urdf_robot=robot_model,
    import_config=import_config,
)

print("Imported to stage at: " + str(prim_path))
