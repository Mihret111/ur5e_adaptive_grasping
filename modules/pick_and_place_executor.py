# modules/pick_and_place_executor.py
import os
import asyncio

from modules.arm_controller import UR5EController
from modules.gripper_controller import Gripper2FG7


class PickAndPlaceExecutor:
    """
    Minimal Isaac-native generic grasp executor.

    First goal:
      - initialize arm controller
      - initialize 2FG7 controller
      - compute pick waypoints
      - execute a cautious generic pick sequence

    This is NOT yet object-aware adaptive grasping.
    This is the generic baseline behavior.
    """

    def __init__(self, config: dict):
        self.config = config

        print("\n[PickAndPlaceExecutor] Initializing Isaac-native controllers...")

        self.arm = UR5EController(config)

        gripper_paths = config.get("paths", {}).get("gripper", {})

        left_joint = gripper_paths.get(
            "left_finger_joint",
            "/onrobot_2fg7/joints/left_finger_joint",
        )
        right_joint = gripper_paths.get(
            "right_finger_joint",
            "/onrobot_2fg7/joints/right_finger_joint",
        )

        self.gripper = Gripper2FG7(
            left_joint_path=left_joint,
            right_joint_path=right_joint,
            config=config,
        )
        self.move_home_before_pick = config.get("move_home_before_pick", True)   #
        print("[PickAndPlaceExecutor] Ready.")

    async def _step_gripper_for_seconds(self, seconds: float):
        """
        Advance the gripper state machine while Isaac physics steps.
        """
        import omni.kit.app

        app = omni.kit.app.get_app()
        frames = max(1, int(seconds * 60))

        for _ in range(frames):
            self.gripper.update()
            await app.next_update_async()

    async def open_gripper(self):
        print("[Executor] Opening gripper...")
        self.gripper.open()
        await self._step_gripper_for_seconds(1.0)
        print(f"[Executor] Gripper state: {self.gripper.get_state()}")

    async def close_gripper(self, force_n=None):
        print("[Executor] Closing gripper...")
        self.gripper.close(force_n=force_n)
        await self._step_gripper_for_seconds(2.0)
        print(f"[Executor] Gripper state: {self.gripper.get_state()}")
        print(f"[Executor] Has object: {self.gripper.has_object()}")

    # implement move to pregrasp_approach position( a point above target pos with constant height of pregrasp_height)
    async def run_generic_pick(self, scene_info: dict) -> bool:
        """
        Run a first generic pick attempt using the professor's Isaac-native
        arm and gripper controllers.

        This is intentionally conservative.
        """
        # Get the pick target and table info from the scene_info, but first i need to know the structure of scene_info
        target = scene_info["pick_target"]
        table_info = scene_info["table_info"]

        # extract the object world position and table height and pan to object angle
        object_world_pos = target["world_pos"]
        table_height = scene_info.get("table_height", table_info["table_size"][2])
        pan_to_object_deg = scene_info.get("pan_to_table_deg", 0.0)

        print("\n[Executor] Generic pick target:")
        print(f"  label: {target.get('label')}")
        print(f"  shape: {target.get('shape')}")
        print(f"  material: {target.get('material_name')}")
        print(f"  world_pos: {object_world_pos}")
        print(f"  table_height: {table_height:.3f}")
        print(f"  pan_to_object_deg: {pan_to_object_deg:.2f}")

        self.arm.set_table_info(table_info)

        print("\n[Executor] Arm status before planning:")
        print(self.arm.get_status())

        # Start from a known safe posture (go to home position)
        if self.config.get("move_home_before_pick", True):
            print("\n[Executor] Moving arm home...")
            await self.arm.move_home(duration=4.0, steps=200)
        else:
            print("\n[Executor] Skipping home motion; starting from current pose.")

        print("\n[Executor] Opening gripper before approach...")
        await self.open_gripper()

        print("\n[Executor] Computing pick joints...")
        # implemented a simple pick_result using compute pick joints for now 
        # TODO:  (if possible)need to improve for robust picking using ML based model later
        pick_result = self.arm.compute_pick_joints(
            object_world_pos=object_world_pos,
            pan_to_object_deg=pan_to_object_deg,
            table_height=table_height,
            object_metadata=target,
            prim_path=target.get("prim_path"),
        )

        print("\n[Executor] Pick planning result:")
        print(f"  target_force_n: {pick_result.get('target_force_n')}")
        print(f"  grasp_strategy: {pick_result.get('grasp_strategy')}")
        print(f"  object_height: {pick_result.get('object_height')}")
        print(f"  IK meta: {pick_result.get('ik_meta')}")
        print(f"  waypoints: {list(pick_result.get('joints', {}).keys())}")

        # check if the pick_result has joints and the joints are not empty
        if "joints" not in pick_result or not pick_result["joints"]:
            print("[Executor] ❌ No pick joints computed.")
            return False

        # First milestone:
        # move only to safe_above before attempting full grasp
        safe_above = pick_result["joints"].get("safe_above")
        if safe_above is None:
            print("[Executor] ❌ No safe_above waypoint.")
            return False

        print("\n[Executor] Moving to safe_above only...")
        # using move_via_safe_height to move to safe_above, 
        ok = await self.arm.move_via_safe_height(
            safe_above,
            duration=3.0,
            steps=150,
        )

        if not ok:
            print("[Executor] ❌ Failed to reach safe_above.")
            return False

        print("[Executor] ✅ Reached safe_above.")     # implement move to pregrasp_approach position( a point above target pos with constant height of pregrasp_height)

        #TODO Once safe_above is reliable, I will enable the full pick sequence
        return True