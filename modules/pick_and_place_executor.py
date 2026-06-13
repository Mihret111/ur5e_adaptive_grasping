# modules/pick_and_place_executor.py

from modules.arm_controller import UR5EController
from modules.gripper_controller import Gripper2FG7
from modules.micro_lift_validator import MicroLiftValidator
import omni.kit.app
from pxr import UsdGeom, Sdf, Usd    # import required usd modules

class PickAndPlaceExecutor:
    """
    Minimal Isaac-native generic grasp executor.

    First goal:
      - initialize arm controller
      - initialize 2FG7 controller
      - compute pick waypoints
      - execute a generic pick sequence

    TODO object-aware adaptive grasping. This is just a generic baseline behavior
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
        self.micro_lift_validator = MicroLiftValidator(config)
        # self.move_home_before_pick = config.get("move_home_before_pick", True)   #
        print("[PickAndPlaceExecutor] Ready.")

    async def _step_gripper_for_seconds(self, seconds: float):
        """
        Advance the gripper state machine while Isaac physics steps.
        """
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


        if seconds is None:
            seconds = float(self.config.get("inspection_hold_seconds", 2.0))

        app = omni.kit.app.get_app()
        frames = max(1, int(seconds * 60))

        for _ in range(frames):
            self.gripper.update()
            await app.next_update_async()
            
    # helper to get object position from prim path
    def _get_prim_world_pos(self, prim_path: str):
        stage = omni.usd.get_context().get_stage()    # get the stage 
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path)) # get the prim from prim path

        if not prim.IsValid(): # if the prim is not valid, print an error message and return None
            print(f"[Executor] ⚠️ Prim not found: {prim_path}")
            return None
        
        xf = UsdGeom.Xformable(prim) # get the xformable interface from the prim
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) # compute the local to world transform
        p = mtx.ExtractTranslation() # extract the translation from the transform

        return [float(p[0]), float(p[1]), float(p[2])]

    # ─────────────────────────────────────────────────────────────────────
    # Helper to always  reset robot before the trial starts
    # ─────────────────────────────────────────────────────────────────────
    async def reset_robot_for_trial(self):
        """
        Clean robot/gripper reset for repeated Isaac Script Editor runs.

        Important:
        Do NOT step physics for the gripper before commanding the arm.
        Otherwise old UR5e drive targets from the previous run may move the arm.
        """
        print("\n[Executor] Resetting robot for trial...")

        # 1. Tell gripper to open, but do NOT step it alone yet.
        # The gripper will update while the arm goes home.
        print("[Executor] Reset: command gripper open, no pre-home stepping.")
        
        print("[TRACE_RESET] before gripper.open")
        self.gripper.open()
        print("[TRACE_RESET] after gripper.open, before arm.move_home")

        # 2. Immediately command arm home slowly.
        print("[Executor] Reset: moving arm home immediately...")
        await self.arm.move_home(
            duration=float(self.config.get("startup_home_duration", 6.0)),
            steps=int(self.config.get("startup_home_steps", 360)),
        )

        # 3. Now let the gripper finish opening and physics settle.
        print("[Executor] Reset: settling after home...")

        print("[TRACE_RESET] after arm.move_home, before post-home settle")
        await self._step_gripper_for_seconds(
            float(self.config.get("post_home_settle_seconds", 0.8))
        )
        print("[TRACE_RESET] after post-home settle")

        print("[Executor] Robot reset complete.")
        
    # ─────────────────────────────────────────────────────────────────────────
    # helper to reset object to its original position
    # ─────────────────────────────────────────────────────────────────────────
    async def reset_object_for_trial(self):
        """
        Smooth reset at the beginning of a trial.

        Assumes preplay sync already prevented stale-drive wake-up.
        """
        print("\n[Executor] Resetting robot for trial...")

        # Command gripper open but do not do long pre-home stepping.
        print("[Executor] Reset: command gripper open.")
        self.gripper.open()

        print("[Executor] Reset: moving arm home smoothly...")
        await self.arm.move_home(
            duration=float(self.config.get("startup_home_duration", 6.0)),
            steps=int(self.config.get("startup_home_steps", 360)),
        )

        print("[Executor] Reset: settling after home...")
        await self._step_gripper_for_seconds(
            float(self.config.get("post_home_settle_seconds", 0.8))
        )

        print("[Executor] Robot reset complete.")
        
    # ─────────────────────────────────────────────────────────────────────────
    # helper to park robot after trial
    # ─────────────────────────────────────────────────────────────────────────
    async def park_robot_after_trial(self):
        """
        Park robot safely at the end of a trial.

        This reduces stale target problems in the next run.
        """
        if not self.config.get("park_robot_after_trial", True):
            return

        print("\n[Executor] Parking robot after trial...")

        self.gripper.open()
        await self._step_gripper_for_seconds(
            float(self.config.get("park_gripper_open_seconds", 0.8))
        )

        await self.arm.move_home(
            duration=float(self.config.get("park_home_duration", 5.0)),
            steps=int(self.config.get("park_home_steps", 300)),
        )

        await self._step_gripper_for_seconds(
            float(self.config.get("park_settle_seconds", 0.5))
        )

        print("[Executor] Robot parked.")
    
    #-------------------------------
    #  helpers to check grasp success:-
    # -------------------------------
    # 1. calculate distance between two points
    def _distance(self, a, b):
        import math
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    # 2. helper to validate grasp just after closing gripper
    def validate_after_close(self, target, object_pos_before_close):
        object_pos_after = self._get_prim_world_pos(target.get("prim_path"))
        flange_pos = self.arm.get_flange_world_pos()

        result = {
            "stage": "after_close",
            "gripper_has_object": self.gripper.has_object(),
            "object_pos_before": object_pos_before_close,
            "object_pos_after": object_pos_after,
            "flange_pos": flange_pos,
            "object_shift": None,
            "flange_object_distance": None,
            "success": False,
            "reasons": [],
        }

        if object_pos_before_close is None or object_pos_after is None:
            result["reasons"].append("missing object pose")
            return result

        shift = self._distance(object_pos_before_close, object_pos_after)
        dist = self._distance(object_pos_after, flange_pos)

        result["object_shift"] = shift
        result["flange_object_distance"] = dist

        max_shift = float(self.config.get("max_object_shift_during_close_m", 0.03))
        max_dist = float(self.config.get("max_object_flange_distance_after_close_m", 0.25))

        if not result["gripper_has_object"]:
            result["reasons"].append("gripper_has_object false")

        if shift > max_shift:
            result["reasons"].append(
                f"object shifted too much during close: {shift:.3f} > {max_shift:.3f}"
            )

        if dist > max_dist:
            result["reasons"].append(
                f"object too far from flange after close: {dist:.3f} > {max_dist:.3f}"
            )

        result["success"] = len(result["reasons"]) == 0
        return result

    # 3. helper to validate grasp after lift
    def validate_after_lift(self, target, object_pos_before_lift, table_height):
        object_pos_after = self._get_prim_world_pos(target.get("prim_path"))
        flange_pos = self.arm.get_flange_world_pos()

        result = {
            "stage": "after_lift",
            "gripper_has_object": self.gripper.has_object(),
            "object_pos_before_lift": object_pos_before_lift,
            "object_pos_after_lift": object_pos_after,
            "flange_pos": flange_pos,
            "object_lift_delta_z": None,
            "flange_object_distance": None,
            "success": False,
            "reasons": [],
        }

        if object_pos_before_lift is None or object_pos_after is None:
            result["reasons"].append("missing object pose")
            return result

        dz = object_pos_after[2] - object_pos_before_lift[2]
        dist = self._distance(object_pos_after, flange_pos)

        result["object_lift_delta_z"] = dz
        result["flange_object_distance"] = dist

        min_lift_delta = float(self.config.get("min_success_lift_delta_m", 0.04))
        min_above_table = float(self.config.get("min_object_above_table_m", 0.03))
        max_dist = float(self.config.get("max_object_flange_distance_after_lift_m", 0.25))

        if not result["gripper_has_object"]:
            result["reasons"].append("gripper_has_object false")

        if dz < min_lift_delta:
            result["reasons"].append(
                f"object did not lift enough: dz={dz:.3f} < {min_lift_delta:.3f}"
            )

        if object_pos_after[2] < table_height + min_above_table:
            result["reasons"].append(
                f"object not above table enough: z={object_pos_after[2]:.3f}"
            )

        if dist > max_dist:
            result["reasons"].append(
                f"object too far from flange after lift: {dist:.3f} > {max_dist:.3f}"
            )

        result["success"] = len(result["reasons"]) == 0
        return result

    # Method to get last trial log as .json file and write it to the output directory with a timestamp and create the dir if not exists
    def get_last_trial_log(self):
        return getattr(self, "_last_trial_log", None)
        
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

        # print information about the pick target
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

        trial_log = {
            "target": {
                "label": target.get("label", "unknown"),
                "shape": target.get("shape", "unknown"),
                "material": target.get("material_name", "unknown"),
                "mass": target.get("mass", "unknown"),
                "grip_dim_mm": target.get("grip_dim_mm", "unknown"),
                "prim_path": target.get("prim_path", "unknown"),
            },
            "attempts": [],
            "trial_success": False,
            "final_reason": None,
        }

        print("\n[Executor] Robot reset already handled by TrialRunner.")

        #-------------------------
        # implemented a simple pick_result using compute pick joints for now 
        # TODO:  (if possible)need to improve for robust picking using ML based model later
        max_attempts = int(self.config.get("max_grasp_attempts", 2))

        for attempt in range(max_attempts):
            print(f"\n[Executor] Grasp attempt {attempt + 1}/{max_attempts}")
            attempt_log = {
                "attempt": attempt + 1,
                "stored_object_pos": target.get("world_pos"),
                "actual_object_pos": None,
                "safe_above_ok": False,
                "pre_grasp_ok": False,
                "grasp_ok": False,
                "preclose_diagnostics": None,
                "preclose_geometry_gate": None,
                "close_validation": None,
                "micro_lift_validation": None,
                "lift_validation": None,
                "gripper_diagnostics_after_close": None,
                "gripper_diagnostics_after_micro_lift": None,
                "gripper_diagnostics_after_lift": None,
                "success": False,
                "failure_reason": None,
            }
            actual_object_pos = self._get_prim_world_pos(target.get("prim_path"))

            if actual_object_pos is None:
                actual_object_pos = target["world_pos"]

            attempt_log["actual_object_pos"] = actual_object_pos

            print("[Executor] Stored object pos:", target["world_pos"])
            print("[Executor] Actual object pos:", actual_object_pos)

            # TODO(mihret): make this smarter. 
            # Instead of always computing a new grasp for every attempt, 
            # we could cache the first result and try it multiple times, 
            # only recomputing if the first attempt fails (e.g. after a failed pick or after moving home). TO BE IMPROVED LATER

            # Recompute each attempt because the arm controller may select
            # a different valid grasp orientation candidate.
            pick_result = self.arm.compute_pick_joints(
                object_world_pos=actual_object_pos,    # pass actual_object_pos instead of the stale object_world_pos in target to make it robust to small errors in target position during runtime    
                pan_to_object_deg=pan_to_object_deg,
                table_height=table_height,
                object_metadata=target,
                prim_path=target.get("prim_path"),
            )

            # Added a break with a False return if the IK fails to produce a valid plan.
            # This ensures that the robot does not attempt to execute a failed plan.
            if pick_result.get("planning_failed", False):
                print("[Executor] ❌ Planner rejected the grasp before execution.")
                print("[Executor] IK diagnostics:")
                print(pick_result.get("ik_meta", {}))
                attempt_log["failure_reason"] = "planning_failed_before_safe_above"
                trial_log["attempts"].append(attempt_log)
                trial_log["final_reason"] = "planning_failed_before_safe_above"
                self._last_trial_log = trial_log
                return False
            joints = pick_result["joints"]

            # Print-out of the joints. This is useful for debugging
            # and understanding what the robot is doing.
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
                attempt_log["failure_reason"] = "failed_to_reach_safe_above"
                trial_log["attempts"].append(attempt_log)
                return False

            print("[Executor] ✅ Reached safe_above.")
            attempt_log["safe_above_ok"] = True

            ## read actual object pose again 
            actual_object_pos = self._get_prim_world_pos(target.get("prim_path"))
            if actual_object_pos is None:
                actual_object_pos = target["world_pos"]
            
            attempt_log["actual_object_pos"] = actual_object_pos
            
            pick_result = self.arm.compute_pick_joints(
                object_world_pos=actual_object_pos,    # pass actual_object_pos instead of the stale object_world_pos in target to make it robust to small errors in target position during runtime    
                pan_to_object_deg=pan_to_object_deg,
                table_height=table_height,
                object_metadata=target,
                prim_path=target.get("prim_path"),
            )
            if pick_result.get("planning_failed", False):
                print("[Executor] ❌ Replanning from safe_above was rejected.")
                print("[Executor] IK diagnostics:")
                print(pick_result.get("ik_meta", {}))
                attempt_log["failure_reason"] = "planning_failed_after_safe_above"
                trial_log["attempts"].append(attempt_log)
                trial_log["final_reason"] = "planning_failed_after_safe_above"
                self._last_trial_log = trial_log
                return False

            ## get the computed joints again
            joints = pick_result.get("joints", {})
            # safe_above = joints.get("safe_above")
            pre_grasp = joints.get("pre_grasp")
            grasp = joints.get("grasp")

            ## if pre grasp is None, then recompute pick joints
            if pre_grasp is None:
                print("[Executor] ❌ Missing pre_grasp. Recomputing pick joints...")
                continue

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
                attempt_log["failure_reason"] = "failed_to_reach_pre_grasp"
                trial_log["attempts"].append(attempt_log)
                return False

            print("[Executor] ✅ Reached pre_grasp.")
            attempt_log["pre_grasp_ok"] = True

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
                attempt_log["failure_reason"] = "failed_to_reach_grasp_pose"
                trial_log["attempts"].append(attempt_log)
                return False

            print("[Executor] ✅ Reached grasp pose.")
            attempt_log["grasp_ok"] = True

            # ──── Diagnostic-only snapshot before gripper closure ────
            # We do not reject a grasp yet.  First verify the professor USD's
            # actual flange-to-finger axis convention using measured evidence.
            if self.config.get("enable_preclose_diagnostics", True):
                preclose_object_pos = self._get_prim_world_pos(
                    target.get("prim_path")
                )

                if preclose_object_pos is not None:
                    preclose_diag = self.arm.get_preclose_geometry_diagnostics(
                        object_world_pos=preclose_object_pos,
                        object_metadata=target,
                        planned_flange_world_pos=(
                            pick_result.get("flange_targets", {}).get("grasp")
                        ),
                    )
                    attempt_log["preclose_diagnostics"] = preclose_diag

                    print("\n[Executor] Pre-close geometry diagnostics (no rejection):")
                    print(
                        "  planned flange tracking error: "
                        f"{preclose_diag['planned_flange_tracking_error_m']}"
                    )
                    print(
                        "  flange-object delta: "
                        f"{preclose_diag['flange_object_delta_m']}"
                    )
                    print(
                        "  closest local flange-axis candidate: "
                        f"{preclose_diag['closest_axis_candidate']}"
                    )
                    print(
                        "  closest candidate grasp-centre distance: "
                        f"{preclose_diag['closest_axis_distance_m']:.4f} m"
                    )

                    if self.config.get(
                        "preclose_diagnostics_print_all_axes", True
                    ):
                        for axis_name, axis_info in (
                            preclose_diag["candidate_axes"].items()
                        ):
                            print(
                                f"    {axis_name}: "
                                f"centre={axis_info['grasp_centre_world_pos']} "
                                f"distance={axis_info['grasp_centre_object_distance_m']:.4f} m"
                            )

                    # ──── Calibrated pre-close geometry gate ────
                    preclose_gate = self.arm.evaluate_preclose_geometry_gate(
                        preclose_diag
                    )
                    attempt_log["preclose_geometry_gate"] = preclose_gate

                    print("\n[Executor] Calibrated pre-close geometry gate:")
                    print(
                        "  calibrated finger axis:    "
                        f"{preclose_gate['calibrated_finger_axis_local']}"
                    )
                    print(
                        "  geometry_ok:               "
                        f"{preclose_gate['geometry_ok']}"
                    )
                    print(
                        "  grasp-centre XY error:     "
                        f"{preclose_gate['grasp_centre_xy_error_m']:.4f} m"
                    )
                    print(
                        "  vertical finger overlap:   "
                        f"{preclose_gate['vertical_overlap_m']:.4f} m"
                    )
                    print(
                        "  flange tracking error:     "
                        f"{preclose_gate['planned_flange_tracking_error_m']}"
                    )

                    if (
                        self.config.get("enable_preclose_geometry_gate", True)
                        and not preclose_gate["geometry_ok"]
                    ):
                        print(
                            "[Executor] ❌ Rejecting close: pre-close geometry "
                            "is not plausible."
                        )
                        for reason in preclose_gate["reasons"]:
                            print(f"  - {reason}")

                        attempt_log["failure_reason"] = (
                            "preclose_geometry_gate_failed"
                        )
                        trial_log["attempts"].append(attempt_log)
                        trial_log["trial_success"] = False
                        trial_log["final_reason"] = (
                            "preclose_geometry_gate_failed"
                        )
                        self._last_trial_log = trial_log

                        # The fingers are still open. Retreat gently instead of
                        # issuing a meaningless close command.
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
                        return False
                else:
                    print(
                        "[Executor] ⚠️ Pre-close diagnostics skipped: "
                        "object pose unavailable"
                    )

            # ──── Now try to close the gripper and validate the grasp. ────
            print("\n[Executor] Closing gripper at grasp pose...")

            # Object pose just before closing; used for close/contact validation, NOT lift validation.
            object_pos_before_close = self._get_prim_world_pos(target.get("prim_path"))

            # TODO: use the same approach as in compute_pick_joints to get target force
            target_force = pick_result.get("target_force_n", None)      # TODO target force based on object what ? investigate this more
            await self.close_gripper(force_n=target_force)

            diag = self.gripper.get_diagnostics()
            print("\n[Executor] Gripper diagnostics after close:")
            # print(diag)

            # ──── Validate grasp/contact after closing ────
            close_validation = self.validate_after_close(
                target=target,
                object_pos_before_close=object_pos_before_close,
            )

            print("\n[Executor] Close validation:")
            print(close_validation)
            attempt_log["close_validation"] = close_validation
            attempt_log["gripper_diagnostics_after_close"] = self.gripper.get_diagnostics()

            # ──── Handle close validation failure: retry or fail ────
            if not close_validation["success"]:
                print("[Executor] ❌ Close validation failed. Will retry if attempts remain.")
                for reason in close_validation["reasons"]:
                    print(f"  - {reason}")

                attempt_log["failure_reason"] = str(close_validation["reasons"])
                trial_log["attempts"].append(attempt_log)

                if attempt < max_attempts - 1:
                    print("[Executor] Retrying safely: pause → open → pre_grasp → safe_above")

                    await self._step_gripper_for_seconds(
                        float(self.config.get("post_failed_close_pause_seconds", 0.5))
                    )

                    self.gripper.open()
                    await self._step_gripper_for_seconds(
                        float(self.config.get("post_open_settle_seconds", 0.8))
                    )

                    await self.arm.move_to(
                        pre_grasp,
                        duration=float(self.config.get("retry_to_pregrasp_duration", 3.0)),
                        steps=int(self.config.get("retry_to_pregrasp_steps", 180)),
                        check_table_collision=True,
                    )

                    await self._step_gripper_for_seconds(
                        float(self.config.get("retry_mid_settle_seconds", 0.3))
                    )

                    await self.arm.move_via_safe_height(
                        safe_above,
                        duration=float(self.config.get("retry_to_safe_duration", 4.0)),
                        steps=int(self.config.get("retry_to_safe_steps", 240)),
                    )

                    continue

                trial_log["trial_success"] = False
                trial_log["final_reason"] = "all_attempts_failed_close_validation"
                self._last_trial_log = trial_log
                return False

            else:
                print("[Executor] ✅ Close validation passed.")

                # ──── hold for 1 second after close validation, 
                # this is in case hysics contact may need a short stabilization time before arm motion begins. ────
                await self._step_gripper_for_seconds(
                    float(self.config.get("post_close_hold_seconds", 1.0))
                )

                # ──── Micro-lift proof-of-hold checkpoint ────
                # A partially closed gripper can report contact even after the
                # object has slipped.  Before committing to a full lift, move
                # only a few centimetres and verify that the object follows the
                # calibrated grasp centre.
                if self.config.get("enable_micro_lift_test", False):
                    micro_lift = joints.get("micro_lift")
                    if micro_lift is None:
                        print("[Executor] ❌ No micro_lift waypoint available.")
                        attempt_log["failure_reason"] = "missing_micro_lift_waypoint"
                        trial_log["attempts"].append(attempt_log)
                        trial_log["final_reason"] = "missing_micro_lift_waypoint"
                        self._last_trial_log = trial_log
                        return False

                    object_before_micro = self._get_prim_world_pos(
                        target.get("prim_path")
                    )
                    geometry_before_micro = (
                        self.arm.get_calibrated_capture_geometry_world()
                    )

                    print("\n[Executor] Performing micro-lift verification checkpoint...")
                    ok = await self.arm.move_to(
                        micro_lift,
                        duration=float(self.config.get("micro_lift_duration", 2.0)),
                        steps=int(self.config.get("micro_lift_steps", 120)),
                        check_table_collision=True,
                        step_callback=self.gripper.update,
                    )

                    if not ok:
                        print("[Executor] ❌ Micro-lift motion failed.")
                        attempt_log["failure_reason"] = "micro_lift_motion_failed"
                        trial_log["attempts"].append(attempt_log)
                        trial_log["final_reason"] = "micro_lift_motion_failed"
                        self._last_trial_log = trial_log
                        return False

                    await self._step_gripper_for_seconds(
                        float(self.config.get("post_micro_lift_settle_seconds", 0.4))
                    )

                    object_after_micro = self._get_prim_world_pos(
                        target.get("prim_path")
                    )
                    geometry_after_micro = (
                        self.arm.get_calibrated_capture_geometry_world()
                    )

                    micro_validation = self.micro_lift_validator.evaluate(
                        object_pos_before=object_before_micro,
                        object_pos_after=object_after_micro,
                        flange_pos_before=geometry_before_micro["flange_world_pos"],
                        flange_pos_after=geometry_after_micro["flange_world_pos"],
                        grasp_centre_before=geometry_before_micro["grasp_centre_world_pos"],
                        grasp_centre_after=geometry_after_micro["grasp_centre_world_pos"],
                        gripper_has_object=self.gripper.has_object(),
                    )

                    print("\n[Executor] Micro-lift validation:")
                    print(micro_validation)
                    attempt_log["micro_lift_validation"] = micro_validation
                    attempt_log["gripper_diagnostics_after_micro_lift"] = (
                        self.gripper.get_diagnostics()
                    )

                    if not micro_validation["success"]:
                        print("[Executor] ❌ Micro-lift validation failed.")
                        for reason in micro_validation["reasons"]:
                            print(f"  - {reason}")
                        attempt_log["failure_reason"] = "micro_lift_validation_failed"
                        trial_log["attempts"].append(attempt_log)
                        trial_log["final_reason"] = "micro_lift_validation_failed"
                        self._last_trial_log = trial_log

                        # Safe reactive recovery: keep hold pressure while
                        # retreating upward, then release only at safe height.
                        await self.arm.move_via_safe_height(
                            safe_above,
                            duration=float(self.config.get("retry_to_safe_duration", 4.0)),
                            steps=int(self.config.get("retry_to_safe_steps", 240)),
                            step_callback=self.gripper.update,
                        )
                        self.gripper.open()
                        await self._step_gripper_for_seconds(
                            float(self.config.get("post_open_settle_seconds", 0.8))
                        )
                        return False

                    print("[Executor] ✅ Micro-lift passed: object followed gripper.")

                    if not self.config.get("enable_lift_test", False):
                        print("[Executor] Full lift disabled. Ending after validated micro-lift.")
                        attempt_log["success"] = True
                        trial_log["trial_success"] = True
                        trial_log["final_reason"] = (
                            "micro_lift_validation_passed_full_lift_disabled"
                        )
                        trial_log["attempts"].append(attempt_log)
                        self._last_trial_log = trial_log
                        return True

                elif not self.config.get("enable_lift_test", False):
                    print("[Executor] Lift disabled for now. Ending after successful close validation.")
                    attempt_log["success"] = True
                    trial_log["trial_success"] = True
                    trial_log["final_reason"] = "close_validation_passed_lift_disabled"
                    trial_log["attempts"].append(attempt_log)
                    self._last_trial_log = trial_log
                    return True

                # ──── Perform the full lift ────
                lift = joints.get("lift")
                if lift is None:
                    print("[Executor] ❌ No lift waypoint available.")
                    return False

                # Object pose immediately before lift.
                # This is the reference for lift delta.
                object_pos_before_lift = self._get_prim_world_pos(target.get("prim_path"))

                print("\n[Executor] Lifting object...")

                ok = await self.arm.move_to(
                    lift,
                    duration=float(self.config.get("lift_duration", 5.0)),
                    steps=int(self.config.get("lift_steps", 300)),
                    check_table_collision=True,
                    step_callback=self.gripper.update,
                )

                if not ok:
                    print("[Executor] ❌ Lift motion failed.")
                    return False

                # Let the gripper/object settle briefly after lift.
                await self._step_gripper_for_seconds(0.5)

                print("\n[Executor] Gripper diagnostics after lift:")
                # print(self.gripper.get_diagnostics())

                # ──── Validate actual object lift ────
                lift_validation = self.validate_after_lift(
                    target=target,
                    object_pos_before_lift=object_pos_before_lift,
                    table_height=table_height,
                )

                print("\n[Executor] Lift validation:")
                print(lift_validation)

                attempt_log["lift_validation"] = lift_validation
                attempt_log["gripper_diagnostics_after_lift"] = self.gripper.get_diagnostics()

                if lift_validation["success"]:
                    print("[Executor] ✅ Validated lift: object moved with gripper.")
                    attempt_log["success"] = True
                    trial_log["trial_success"] = True
                    trial_log["final_reason"] = "lift_validation_passed"
                    trial_log["attempts"].append(attempt_log)

                    self._last_trial_log = trial_log
                    return True

                print("[Executor] ❌ Lift validation failed.")
                for reason in lift_validation["reasons"]:
                    print(f"  - {reason}")
                    attempt_log["failure_reason"] = "lift_validation_failed"
                    trial_log["attempts"].append(attempt_log)
                    self._last_trial_log = trial_log
                    return False
 
                ###

            # ──── No object detected? RETRY with NEW GRASP pose?  ────
            print("[Executor] ❌ No object detected after close.")
            
            # retry or give up 
            if attempt < max_attempts - 1:
                print("[Executor] Retrying safely: pause → open → pre_grasp → safe_above")

                await self._step_gripper_for_seconds(
                    float(self.config.get("post_failed_close_pause_seconds", 0.5))
                )

                self.gripper.open()
                await self._step_gripper_for_seconds(
                    float(self.config.get("post_open_settle_seconds", 0.8))
                )

                # retreat to pre_grasp with new approach
                
                await self.arm.move_to(
                    pre_grasp,
                    duration=float(self.config.get("retry_to_pregrasp_duration", 3.0)),
                    steps=int(self.config.get("retry_to_pregrasp_steps", 180)),    # these 
                    check_table_collision=True,
                )

                await self._step_gripper_for_seconds(
                    float(self.config.get("retry_mid_settle_seconds", 0.3))
                )

                await self.arm.move_via_safe_height(
                    safe_above,
                    duration=float(self.config.get("retry_to_safe_duration", 4.0)),
                    steps=int(self.config.get("retry_to_safe_steps", 240)),
                )

            else:
                print("[Executor] ❌ All grasp attempts failed.")
                await self._hold_for_inspection()
                return False