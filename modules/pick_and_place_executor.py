# modules/pick_and_place_executor.py
from launch.actions import reset_launch_configurations
from launch.actions import reset_launch_configurations
import os
import asyncio

from modules.arm_controller import UR5EController
from modules.gripper_controller import Gripper2FG7
import omni.kit.app

class PickAndPlaceExecutor:
    """
    Minimal Isaac-native generic grasp executor.

    First goal:
      - initialize arm controller
      - initialize 2FG7 controller
      - compute pick waypoints
      - execute a generic pick sequence

    TODO object-aware adaptive grasping. This is a generic baseline behavior
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
        # self.move_home_before_pick = config.get("move_home_before_pick", True)   #
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

    # helper to just pause and hold the gripper open or close for inspection 
    async def _hold_for_inspection(self, seconds: float = None):
        if not self.config.get("debug_hold_after_stage", False):
            return

        import omni.kit.app

        if seconds is None:
            seconds = float(self.config.get("inspection_hold_seconds", 2.0))

        app = omni.kit.app.get_app()
        frames = max(1, int(seconds * 60))

        for _ in range(frames):
            self.gripper.update()
            await app.next_update_async()
            
    # implement move to pregrasp_approach position (a point above target pos with constant height of pregrasp_height)
    async def run_generic_pick(self, scene_info: dict) -> bool:
        """
        Run a first generic pick attempt using the Isaac-native
        arm and gripper controllers.

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

        #-------------------------
        # implemented a simple pick_result using compute pick joints for now 
        # TODO:  (if possible)need to improve for robust picking using ML based model later
        max_attempts = int(self.config.get("max_grasp_attempts"))

        for attempt in range(max_attempts):
            print(f"\n[Executor] Grasp attempt {attempt + 1}/{max_attempts}")

            # Recompute each attempt because the arm controller may select
            # a different valid grasp orientation candidate.
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

            joints = pick_result.get("joints", {})
            safe_above = joints.get("safe_above")
            pre_grasp = joints.get("pre_grasp")
            grasp = joints.get("grasp")


            if safe_above is None or pre_grasp is None or grasp is None:
                print("[Executor] ❌ Missing one or more required waypoints.")
                return False

            print("\n[Executor] Moving to safe_above...")
            ok = await self.arm.move_via_safe_height(
                safe_above,
                duration=3.0,
                steps=150,
            )
            if not ok:
                print("[Executor] ❌ Failed to reach safe_above.")
                return False

            print("[Executor] ✅ Reached safe_above.")

            print("\n[Executor] Moving to pre_grasp...")
            ok = await self.arm.move_to(
                pre_grasp,
                duration=3.0,
                steps=150,
                check_table_collision=True,
            )
            if not ok:
                print("[Executor] ❌ Failed to reach pre_grasp.")
                await self.arm.move_via_safe_height(safe_above, duration=3.0, steps=150)
                return False

            print("[Executor] ✅ Reached pre_grasp.")

            print("\n[Executor] Moving to grasp pose...")
            ok = await self.arm.move_to(
                grasp,
                duration=3.0,
                steps=150,
                check_table_collision=True,
            )
            if not ok:
                print("[Executor] ❌ Failed to reach grasp pose.")
                await self.arm.move_to(pre_grasp, duration=2.0, steps=100)
                await self.arm.move_via_safe_height(safe_above, duration=3.0, steps=150)
                return False

            print("[Executor] ✅ Reached grasp pose.")

            print("\n[Executor] Closing gripper at grasp pose...")
            target_force = pick_result.get("target_force_n", None)
            await self.close_gripper(force_n=target_force)

            diag = self.gripper.get_diagnostics()
            print("\n[Executor] Gripper diagnostics after close:")
            print(diag)

            # ──── Test if object is held ────
            if self.gripper.has_object():
                print("[Executor] ✅ Object detected in gripper.")

                if not self.config.get("enable_lift_test", False):
                    print("[Executor] Lift disabled for now. Ending after successful grasp.")
                    return True

            # ──── Perform the lift ────
                lift = joints.get("lift")
                if lift is None:
                    print("[Executor] ❌ No lift waypoint available.")
                    return False

                print("\n[Executor] Lifting object...")

                ok = await self.arm.move_to(
                    lift,
                    duration=3.0,
                    steps=150,
                    check_table_collision=True,
                    step_callback=self.gripper.update,    # reinforces the gripper hold
                )

                if not ok:
                    print("[Executor] ❌ Lift motion failed.")
                    return False

                # Let the gripper hold stabilize briefly
                await self._step_gripper_for_seconds(0.5)

                still_holding = self.gripper.has_object()

                print("\n[Executor] Gripper diagnostics after lift:")
                print(self.gripper.get_diagnostics())

                if still_holding:
                    print("[Executor] ✅ Object still held after lift.")
                    return True

                print("[Executor] ❌ Object lost during lift.")
                return False

            # ──── No object detected ────
            # print("[Executor] ❌ No object detected after close.")
            
            # ──── retry or give up ────
            if attempt < max_attempts - 1:
                print("[Executor] Retrying safely: open → pre_grasp → safe_above")

                self.gripper.open()
                await self._step_gripper_for_seconds(1.0)

                await self.arm.move_to(
                    pre_grasp,
                    duration=2.0,
                    steps=100,
                    check_table_collision=True,
                )

                await self.arm.move_via_safe_height(
                    safe_above,
                    duration=3.0,
                    steps=150,
                )

            else:
                print("[Executor] ❌ All grasp attempts failed.")
                await self._hold_for_inspection()
                return False