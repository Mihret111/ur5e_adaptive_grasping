# modules/pick_and_place_executor.py

from modules.arm_controller import UR5EController
from modules.gripper_controller import Gripper2FG7
from modules.micro_lift_validator import MicroLiftValidator
from modules.soft_object_observer import SoftObjectObserver
from modules.retry_policy import RetryPolicy
from modules.trial_diagnostics import (
    compact_pick_result,
    grip_feasibility,
    json_safe,
    make_event,
    pose_snapshot,
    selected_config_snapshot,
)
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
        self.soft_observer = SoftObjectObserver(config)
        self.retry_policy = RetryPolicy(config)
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

    async def close_gripper(
        self,
        force_n=None,
        expected_grip_dim_m=None,
        hold_settle_extra_s: float = 0.0,
        hold_extra_close_m=None,
    ):
        """Close gripper with optional object-size-aware contact expectation.

        expected_grip_dim_m is the estimated object width/diameter along the
        gripper closing direction.

        hold_settle_extra_s is added by RetryPolicy when a retry needs more
        contact stabilization before validation.
        """
        print("[Executor] Closing gripper...")

        self.gripper.close(
            force_n=force_n,
            expected_grip_dim_m=expected_grip_dim_m,
            hold_extra_close_m=hold_extra_close_m,
        )

        # Wait until the gripper leaves CLOSING, or until timeout.
        # This avoids validating while the gripper is still moving.
        close_resolution = await self._wait_for_gripper_close_resolution(
            float(self.config.get("gripper_close_wait_timeout_s", 6.0))
        )

        # attempt_log["gripper_close_resolution"] = json_safe(close_resolution)

        print(f"[Executor] Gripper close resolved as: {close_resolution}")

        hold_settle_time = float(
            self.config.get("post_close_hold_seconds", 1.0)
        ) + float(
            hold_settle_extra_s
        )

        await self._step_gripper_for_seconds(hold_settle_time)

        print(f"[Executor] Gripper state: {self.gripper.get_state()}")
        print(f"[Executor] Has object: {self.gripper.has_object()}")
        
        return close_resolution

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

        # Imported deformable USD assets are wrapped in an Xform whose root
        # transform is bottom-centre, not object-centre.  Their visual/simulation
        # mesh can also deform without a rigid-body pose update.  For these
        # targets we use the composed bounding-box centre as the best available
        # runtime observation.
        try:
            pose_attr = prim.GetAttribute("b2b:poseSource")
            pose_source = pose_attr.Get() if pose_attr and pose_attr.IsValid() else None
            if pose_source == "bbox_center":
                bbox_cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    useExtentsHint=False,
                )
                box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                mn = box.GetMin()
                mx = box.GetMax()
                return [
                    float((mn[0] + mx[0]) / 2.0),
                    float((mn[1] + mx[1]) / 2.0),
                    float((mn[2] + mx[2]) / 2.0),
                ]
        except Exception as e:
            print(f"[Executor] ⚠️ bbox_center pose read failed for {prim_path}: {e}")

        xf = UsdGeom.Xformable(prim) # get the xformable interface from the prim
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) # compute the local to world transform
        p = mtx.ExtractTranslation() # extract the translation from the transform

        return [float(p[0]), float(p[1]), float(p[2])]

    def _is_soft_target(self, target: dict) -> bool:
        return self.soft_observer.is_soft_target(target)

    def _observe_target(self, target: dict, stage_name: str = "observe"):
        """Return live soft-object observation when available.

        For deformable USD objects this is the perceptual state used by
        planning and validation.  Rigid targets return None and keep the old
        root-transform behaviour.
        """
        if not self._is_soft_target(target):
            return None
        obs = self.soft_observer.observe(target)
        if obs:
            print(
                f"[SoftObserver:{stage_name}] "
                f"source={obs.get('pose_source')} "
                f"center=({obs['center'][0]:.4f},{obs['center'][1]:.4f},{obs['center'][2]:.4f}) "
                f"bottom_z={obs.get('bottom_z'):.4f} top_z={obs.get('top_z'):.4f} "
                f"h={obs.get('height_m'):.4f} table_gap={obs.get('table_gap_m')} "
                f"warnings={obs.get('warnings', [])}"
            )
        else:
            print(f"[SoftObserver:{stage_name}] unavailable; falling back to root transform")
        return obs

    def _get_observed_object_pos(self, target: dict, stage_name: str = "object"):
        """Return grasp-relevant object centre.

        Soft objects use SoftObjectObserver's visible-bbox centre.  Rigid
        objects use the original root transform.
        """
        obs = self._observe_target(target, stage_name=stage_name)
        if obs and obs.get("center") is not None:
            return list(obs["center"])
        prim_path = target.get("prim_path") if isinstance(target, dict) else None
        return self._get_prim_world_pos(prim_path)

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

    def _last_known_object_pos_from_attempt(self, attempt_log: dict):
        """Return the last object position observed inside a previous attempt."""
        if not attempt_log:
            return None

        micro = attempt_log.get("micro_lift_validation") or {}
        if micro.get("object_pos_after") is not None:
            return micro.get("object_pos_after")

        close = attempt_log.get("close_validation") or {}
        if close.get("object_pos_after") is not None:
            return close.get("object_pos_after")

        if attempt_log.get("actual_object_pos") is not None:
            return attempt_log.get("actual_object_pos")

        return None

    def _object_retry_displacement_summary(self, current_pos, previous_attempt: dict):
        """Log how far the object moved between retry attempts."""
        previous_pos = self._last_known_object_pos_from_attempt(previous_attempt)

        summary = {
            "previous_last_known_object_pos": previous_pos,
            "current_refreshed_object_pos": current_pos,
            "displacement_since_previous_attempt_m": None,
            "horizontal_displacement_since_previous_attempt_m": None,
            "retry_used_refreshed_pose": current_pos is not None,
        }

        if previous_pos is None or current_pos is None:
            return summary

        summary["displacement_since_previous_attempt_m"] = self._distance(
            previous_pos,
            current_pos,
        )
        summary["horizontal_displacement_since_previous_attempt_m"] = (
            (
                (float(previous_pos[0]) - float(current_pos[0])) ** 2
                + (float(previous_pos[1]) - float(current_pos[1])) ** 2
            ) ** 0.5
        )
        return summary

    # 2. helper to validate grasp just after closing gripper
    def validate_after_close(self, target, object_pos_before_close):
        object_pos_after = self._get_observed_object_pos(target, stage_name="after_close")
        flange_pos = self.arm.get_flange_world_pos()
        soft_obs_after = self._observe_target(target, stage_name="after_close_validation")

        result = {
            "stage": "after_close",
            "validation_mode": "soft_bbox" if self._is_soft_target(target) else "rigid_root",
            "gripper_has_object": self.gripper.has_object(),
            "object_pos_before": object_pos_before_close,
            "object_pos_after": object_pos_after,
            "soft_observation_after": soft_obs_after,
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

        if self._is_soft_target(target) and soft_obs_after:
            result["soft_metrics"] = {
                "height_m": soft_obs_after.get("height_m"),
                "deformation_ratio_z": soft_obs_after.get("deformation_ratio_z"),
                "table_gap_m": soft_obs_after.get("table_gap_m"),
                "collision_visible_bottom_offset_m": soft_obs_after.get("collision_visible_bottom_offset_m"),
                "warnings": soft_obs_after.get("warnings", []),
            }

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

        if self._is_soft_target(target) and soft_obs_after:
            max_table_gap = float(self.config.get("max_soft_table_gap_m", 0.003))
            table_gap = soft_obs_after.get("table_gap_m")
            if table_gap is not None and table_gap > max_table_gap:
                result["reasons"].append(
                    f"soft object not in table/contact-consistent pose: table_gap={table_gap:.4f} > {max_table_gap:.4f}"
                )

        result["success"] = len(result["reasons"]) == 0
        return result

    # 3. helper to validate grasp after lift
    def validate_after_lift(self, target, object_pos_before_lift, table_height):
        object_pos_after = self._get_observed_object_pos(target, stage_name="after_full_lift")
        flange_pos = self.arm.get_flange_world_pos()
        soft_obs_after = self._observe_target(target, stage_name="after_full_lift_validation")

        result = {
            "stage": "after_lift",
            "validation_mode": "soft_bbox" if self._is_soft_target(target) else "rigid_root",
            "gripper_has_object": self.gripper.has_object(),
            "object_pos_before_lift": object_pos_before_lift,
            "object_pos_after_lift": object_pos_after,
            "soft_observation_after": soft_obs_after,
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
        return json_safe(getattr(self, "_last_trial_log", None))
    
    def _apply_retry_adjustments_to_pick_result(
        self,
        pick_result: dict,
        adjustments: dict,
    ) -> dict:
        """Apply retry-time command adjustments to a planned pick result.

        The arm planner computes the geometric plan.
        RetryPolicy may then scale the commanded grasp force for the next attempt.

        Geometry adjustment, such as grasp_z_delta_m, is handled separately
        inside arm_controller through set_runtime_grasp_z_delta().
        """
        if not pick_result:
            return pick_result

        adjustments = adjustments or {}

        force_scale = float(adjustments.get("force_scale", 1.0))

        old_force = float(pick_result.get("target_force_n", 80.0))
        min_force = float(self.config.get("min_grip_force", 40.0))
        max_force = float(self.config.get("max_grip_force", 140.0))

        new_force = max(
            min_force,
            min(max_force, old_force * force_scale),
        )

        pick_result["target_force_n_before_retry_scale"] = old_force
        pick_result["retry_force_scale"] = force_scale
        pick_result["target_force_n"] = new_force

        return pick_result

    async def _recover_to_safe_for_retry(
        self,
        pre_grasp,
        safe_above,
        from_micro_lift: bool = False,
    ):
        """Return to a safe configuration before another grasp attempt.

        This is a recovery fixed-action pattern.

        If failure happened after micro-lift, the arm is already above the
        object and the gripper may still be holding or partially holding
        something. In that case, move upward safely first, then open.

        If failure happened during/after close at grasp pose, open and retreat
        through pre_grasp and safe_above.
        """
        print("[Executor] Recovering safely before retry...")

        # Case A:
        # Failure happened after micro-lift.
        # The safest behaviour is to keep hold pressure while moving upward,
        # then open only at a safer height.
        if from_micro_lift:
            if safe_above is not None:
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
            return

        # Case B:
        # Failure happened before/during close or during close validation.
        # First give the simulator a small pause, then open the gripper.
        await self._step_gripper_for_seconds(
            float(self.config.get("post_failed_close_pause_seconds", 0.5))
        )

        self.gripper.open()
        await self._step_gripper_for_seconds(
            float(self.config.get("post_open_settle_seconds", 0.8))
        )

        # Move back upward along the approach chain.
        if pre_grasp is not None:
            await self.arm.move_to(
                pre_grasp,
                duration=float(self.config.get("retry_to_pregrasp_duration", 3.0)),
                steps=int(self.config.get("retry_to_pregrasp_steps", 180)),
                check_table_collision=True,
                step_callback=self.gripper.update,
            )

        await self._step_gripper_for_seconds(
            float(self.config.get("retry_mid_settle_seconds", 0.3))
        )

        if safe_above is not None:
            await self.arm.move_via_safe_height(
                safe_above,
                duration=float(self.config.get("retry_to_safe_duration", 4.0)),
                steps=int(self.config.get("retry_to_safe_steps", 240)),
                step_callback=self.gripper.update,
            )

    async def _wait_for_gripper_close_resolution(
        self,
        max_seconds: float = None,
    ) -> dict:
        """Wait until the gripper close action reaches a terminal state.

        We should not validate grasp success while the gripper is still CLOSING.

        The close is considered resolved when:
          - the gripper enters HOLDING after plausible contact/squeeze, or
          - the gripper enters HOLDING after a no-object timeout/full close, or
          - the timeout expires.

        Returns a diagnostic dictionary for logging.
        """
        app = omni.kit.app.get_app()

        if max_seconds is None:
            max_seconds = float(
                self.config.get("gripper_close_wait_timeout_s", 6.0)
            )

        # We use frame count because gripper.update() is called once per frame.
        # 6 seconds × 60 Hz = 360 updates, enough for:
        # max_closing_ticks ≈ 180 + squeeze_ticks ≈ 50 + margin.
        updates_per_second = float(
            self.config.get("gripper_close_wait_updates_per_second", 60.0)
        )

        max_frames = max(
            1,
            int(max_seconds * updates_per_second),
        )

        last_state = None
        frames_used = 0

        for i in range(max_frames):
            last_state = self.gripper.update()
            frames_used = i + 1

            await app.next_update_async()

            # The key guard:
            # Do not continue waiting once close has resolved.
            if last_state != self.gripper.CLOSING:
                diagnostics = self.gripper.get_diagnostics()
                return {
                    "resolved": True,
                    "frames_used": frames_used,
                    "max_frames": max_frames,
                    "final_state": last_state,
                    "timeout_s": max_seconds,
                    "gripper_diagnostics": diagnostics,
                }

        # Timeout: close did not resolve.
        diagnostics = self.gripper.get_diagnostics()
        return {
            "resolved": False,
            "frames_used": frames_used,
            "max_frames": max_frames,
            "final_state": last_state,
            "timeout_s": max_seconds,
            "gripper_diagnostics": diagnostics,
            "reason": "gripper_close_wait_timeout",
        }
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
            "schema_version": "b2b_validation_v2_analysis",
            "target": {
                "label": target.get("label", "unknown"),
                "shape": target.get("shape", "unknown"),
                "material": target.get("material_name", "unknown"),
                "mass": target.get("mass", "unknown"),
                "grip_dim_mm": target.get("grip_dim_mm", "unknown"),
                "prim_path": target.get("prim_path", "unknown"),
            },
            "target_full": json_safe(target),
            "initial_soft_observation": json_safe(self._observe_target(target, stage_name="trial_start")),
            "target_feasibility": grip_feasibility(target, self.config),
            "table_info": json_safe(table_info),
            "config_snapshot": selected_config_snapshot(self.config),
            "events": [
                make_event(
                    "trial_log_created",
                    target_label=target.get("label", "unknown"),
                    target_shape=target.get("shape", "unknown"),
                )
            ],
            "attempts": [],
            "trial_success": False,
            "final_reason": None,
        }

        # ─────────────────────────────────────────────────────────────
        # Early affordance/range guard.
        # If the object is clearly outside the hard 2FG7 range, do not spend
        # time planning and do not pretend a pick primitive is reasonable.
        # This is action selection rejecting an unavailable motor schema.
        target_feasibility = trial_log.get("target_feasibility", {}) or {}
        if (not target_feasibility.get("within_hard_range", True)) or str(
            target_feasibility.get("grip_feasibility_class", "")
        ).startswith("infeasible"):
            print("[Executor] ❌ Target rejected before planning: outside hard gripper range.")
            trial_log["trial_success"] = False
            trial_log["final_reason"] = "target_infeasible_for_gripper_range"
            trial_log["events"].append(
                make_event(
                    "target_infeasible_for_gripper_range",
                    grip_dim_m=target_feasibility.get("estimated_object_grip_dim_m"),
                    grip_range_min_m=target_feasibility.get("grip_range_min_m"),
                    grip_range_max_m=target_feasibility.get("grip_range_max_m"),
                    grip_feasibility_class=target_feasibility.get("grip_feasibility_class"),
                )
            )
            self._last_trial_log = trial_log
            return False

        print("\n[Executor] Robot reset already handled by TrialRunner.")

        #-------------------------
        # implemented a simple pick_result using compute pick joints for now 
        # TODO:  (if possible)need to improve for robust picking using ML based model later
        max_attempts = int(self.config.get("max_grasp_attempts", 1))
        current_retry_adjustments = {}

        for attempt in range(max_attempts):
            print(f"\n[Executor] Grasp attempt {attempt + 1}/{max_attempts}")
            attempt_log = {
                "attempt": attempt + 1,
                "stored_object_pos": json_safe(target.get("world_pos")),
                "actual_object_pos": None,
                "object_displacement_since_previous_attempt": None,
                "target_feasibility": grip_feasibility(target, self.config),
                "safe_above_ok": False,
                "pre_grasp_ok": False,
                "grasp_ok": False,
                "planning_initial": None,
                "planning_replanned_after_safe_above": None,
                "preclose_diagnostics": None,
                "preclose_geometry_gate": None,
                "close_validation": None,
                "micro_lift_validation": None,
                "lift_validation": None,
                "gripper_diagnostics_after_close": None,
                "gripper_diagnostics_after_micro_lift": None,
                "gripper_diagnostics_after_lift": None,
                "snapshots": {},
                "events": [make_event("attempt_started", attempt=attempt + 1)],
                "success": False,
                "failure_reason": None,
            }
            attempt_log["retry_adjustments_applied"] = json_safe(
                current_retry_adjustments
            )
            self.arm.set_runtime_grasp_z_delta(
                float(current_retry_adjustments.get("grasp_z_delta_m", 0.0))
            )

            actual_object_pos = self._get_observed_object_pos(target, stage_name="before_planning")

            if actual_object_pos is None:
                actual_object_pos = target["world_pos"]

            attempt_log["actual_object_pos"] = json_safe(actual_object_pos)
            if attempt > 0 and trial_log.get("attempts"):
                attempt_log["object_displacement_since_previous_attempt"] = json_safe(
                    self._object_retry_displacement_summary(
                        actual_object_pos,
                        trial_log["attempts"][-1],
                    )
                )
                attempt_log["events"].append(
                    make_event(
                        "retry_object_pose_refreshed",
                        object_displacement_since_previous_attempt_m=(
                            attempt_log["object_displacement_since_previous_attempt"].get(
                                "displacement_since_previous_attempt_m"
                            )
                            if attempt_log.get("object_displacement_since_previous_attempt")
                            else None
                        ),
                    )
                )
            attempt_log["snapshots"]["before_planning"] = pose_snapshot(
                self, target, "before_planning"
            )

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
            pick_result = self._apply_retry_adjustments_to_pick_result(
                pick_result,
                current_retry_adjustments,
            )

            # Added a break with a False return if the IK fails to produce a valid plan.
            # This ensures that the robot does not attempt to execute a failed plan.
            attempt_log["planning_initial"] = compact_pick_result(pick_result)
            attempt_log["events"].append(
                make_event(
                    "planning_initial_done",
                    planning_failed=pick_result.get("planning_failed", False),
                    target_force_n=pick_result.get("target_force_n"),
                    waypoints=list((pick_result.get("joints", {}) or {}).keys()),
                )
            )

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
            attempt_log["snapshots"]["after_safe_above"] = pose_snapshot(
                self, target, "after_safe_above"
            )
            attempt_log["events"].append(make_event("safe_above_reached"))

            ## read actual object pose again 
            actual_object_pos = self._get_observed_object_pos(target, stage_name="after_safe_above")
            if actual_object_pos is None:
                actual_object_pos = target["world_pos"]
            
            attempt_log["actual_object_pos"] = json_safe(actual_object_pos)
            attempt_log["retry_used_refreshed_pose_after_safe_above"] = True
            attempt_log["snapshots"]["after_safe_above_object_refresh"] = pose_snapshot(
                self, target, "after_safe_above_object_refresh"
            )
            
            pick_result = self.arm.compute_pick_joints(
                object_world_pos=actual_object_pos,    # pass actual_object_pos instead of the stale object_world_pos in target to make it robust to small errors in target position during runtime    
                pan_to_object_deg=pan_to_object_deg,
                table_height=table_height,
                object_metadata=target,
                prim_path=target.get("prim_path"),
            )
            pick_result = self._apply_retry_adjustments_to_pick_result(
                pick_result,
                current_retry_adjustments,
            )
            attempt_log["planning_replanned_after_safe_above"] = compact_pick_result(
                pick_result
            )
            attempt_log["events"].append(
                make_event(
                    "planning_replanned_after_safe_above_done",
                    planning_failed=pick_result.get("planning_failed", False),
                    target_force_n=pick_result.get("target_force_n"),
                    waypoints=list((pick_result.get("joints", {}) or {}).keys()),
                )
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
            attempt_log["snapshots"]["at_grasp_before_preclose"] = pose_snapshot(
                self, target, "at_grasp_before_preclose"
            )
            attempt_log["events"].append(make_event("grasp_pose_reached"))

            # ──── Diagnostic-only snapshot before gripper closure ────
            # We do not reject a grasp yet.  First verify the professor USD's
            # actual flange-to-finger axis convention using measured evidence.
            if self.config.get("enable_preclose_diagnostics", True):
                preclose_object_pos = self._get_observed_object_pos(
                    target, stage_name="preclose"
                )

                if preclose_object_pos is not None:
                    preclose_diag = self.arm.get_preclose_geometry_diagnostics(
                        object_world_pos=preclose_object_pos,
                        object_metadata=target,
                        planned_flange_world_pos=(
                            pick_result.get("flange_targets", {}).get("grasp")
                        ),
                    )
                    attempt_log["preclose_diagnostics"] = json_safe(preclose_diag)

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

                        attempt_log["failure_reason"] = "preclose_geometry_gate_failed"
                        decision = self.retry_policy.decide(trial_log, attempt_log)
                        attempt_log["retry_decision"] = json_safe(decision)
                        trial_log["attempts"].append(attempt_log)

                        if decision["retry"] and attempt < max_attempts - 1:
                            print(f"[Executor] RetryPolicy: {decision['reason']}")
                            current_retry_adjustments = decision.get("adjustments", {})
                            await self._recover_to_safe_for_retry(
                                pre_grasp=pre_grasp,
                                safe_above=safe_above,
                                from_micro_lift=False,
                            )
                            continue

                        trial_log["trial_success"] = False
                        trial_log["final_reason"] = "preclose_geometry_gate_failed"
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
            object_pos_before_close = self._get_observed_object_pos(target, stage_name="before_close")

            # TODO: use the same approach as in compute_pick_joints to get target force
            target_force = pick_result.get("target_force_n", None)

            target_feasibility_for_close = (
                attempt_log.get("target_feasibility")
                or trial_log.get("target_feasibility")
                or {}
            )

            expected_grip_dim_m = target_feasibility_for_close.get(
                "estimated_object_grip_dim_m"
            )

            print(
                "[Executor] Expected grip dimension for close: "
                f"{expected_grip_dim_m}"
            )

            close_resolution = await self.close_gripper(
                force_n=target_force,
                expected_grip_dim_m=expected_grip_dim_m,
                hold_settle_extra_s=float(
                    current_retry_adjustments.get("hold_settle_extra_s", 0.0)
                ),
                hold_extra_close_m=pick_result.get("gripper_hold_extra_close_m"),
            )
            attempt_log["gripper_close_resolution"] = json_safe(close_resolution)

            # get diagnostics from the gripper after close
            diag = self.gripper.get_diagnostics()
            print("\n[Executor] Gripper diagnostics after close:")
            # print(diag)

            # Refuse close-validation if close did not resolve
            if not close_resolution.get("resolved", False):
                print("[Executor] Close did not resolve before validation timeout.")

                attempt_log["close_validation"] = {
                    "stage": "after_close",
                    "success": False,
                    "reasons": ["gripper_close_not_resolved_before_validation"],
                    "gripper_state": close_resolution.get("final_state"),
                }

                attempt_log["gripper_diagnostics_after_close"] = (
                    self.gripper.get_diagnostics()
                )

                attempt_log["failure_reason"] = "close_validation_failed"
                attempt_log["close_failure_reasons"] = [
                    "gripper_close_not_resolved_before_validation"
                ]

                decision = self.retry_policy.decide(trial_log, attempt_log)
                attempt_log["retry_decision"] = json_safe(decision)

                trial_log["attempts"].append(attempt_log)

                if decision["retry"] and attempt < max_attempts - 1:
                    print(f"[Executor] RetryPolicy: {decision['reason']}")
                    current_retry_adjustments = decision.get("adjustments", {})

                    await self._recover_to_safe_for_retry(
                        pre_grasp=pre_grasp,
                        safe_above=safe_above,
                        from_micro_lift=False,
                    )
                    continue

                trial_log["trial_success"] = False
                trial_log["final_reason"] = "gripper_close_not_resolved_before_validation"
                self._last_trial_log = trial_log
                return False
            # 

            # ──── Validate grasp/contact after closing ────
            close_validation = self.validate_after_close(
                target=target,
                object_pos_before_close=object_pos_before_close,
            )


            print("\n[Executor] Close validation:")
            print(close_validation)
            attempt_log["close_validation"] = json_safe(close_validation)
            attempt_log["gripper_diagnostics_after_close"] = json_safe(
                self.gripper.get_diagnostics()
            )
            attempt_log["snapshots"]["after_close_validation"] = pose_snapshot(
                self, target, "after_close_validation"
            )
            attempt_log["events"].append(
                make_event(
                    "close_validation_done",
                    success=close_validation.get("success"),
                    reasons=close_validation.get("reasons", []),
                )
            )

            # ──── Handle close validation failure: retry or fail ────
            if not close_validation["success"]:
                print("[Executor] ❌ Close validation failed. Will retry if attempts remain.")
                for reason in close_validation["reasons"]:
                    print(f"  - {reason}")

                attempt_log["failure_reason"] = "close_validation_failed"
                attempt_log["close_failure_reasons"] = json_safe(
                    close_validation.get("reasons", [])
                )
                decision = self.retry_policy.decide(trial_log, attempt_log)
                attempt_log["retry_decision"] = json_safe(decision)
                trial_log["attempts"].append(attempt_log)

                if decision["retry"] and attempt < max_attempts - 1:
                    print(f"[Executor] RetryPolicy: {decision['reason']}")
                    current_retry_adjustments = decision.get("adjustments", {})
                    await self._recover_to_safe_for_retry(
                        pre_grasp=pre_grasp,
                        safe_above=safe_above,
                        from_micro_lift=False,
                    )
                    continue

                trial_log["trial_success"] = False
                trial_log["final_reason"] = "all_attempts_failed_close_validation"
                self._last_trial_log = trial_log
                return False

            else:
                print("[Executor] ✅ Close validation passed.")

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

                    object_before_micro = self._get_observed_object_pos(
                        target, stage_name="before_micro_lift"
                    )
                    geometry_before_micro = (
                        self.arm.get_calibrated_capture_geometry_world()
                    )

                    print("\n[Executor] Performing micro-lift verification checkpoint...")
                    plan_micro_lift_speed_scale = float(
                        pick_result.get("micro_lift_speed_scale", 1.0) or 1.0
                    )
                    retry_micro_lift_speed_scale = float(
                        current_retry_adjustments.get("micro_lift_speed_scale", 1.0)
                    )
                    # Use the slower/more cautious speed when either the object
                    # strategy or retry policy requests it.
                    micro_lift_speed_scale = min(
                        plan_micro_lift_speed_scale,
                        retry_micro_lift_speed_scale,
                    )
                    base_micro_lift_duration = float(
                        self.config.get("micro_lift_duration", 2.0)
                    )
                    micro_lift_duration = base_micro_lift_duration / max(
                        0.1,
                        micro_lift_speed_scale,
                    )
                    attempt_log["micro_lift_speed_profile"] = json_safe({
                        "plan_micro_lift_speed_scale": plan_micro_lift_speed_scale,
                        "retry_micro_lift_speed_scale": retry_micro_lift_speed_scale,
                        "effective_micro_lift_speed_scale": micro_lift_speed_scale,
                        "duration_s": micro_lift_duration,
                    })

                    ok = await self.arm.move_to(
                        micro_lift,
                        duration=micro_lift_duration,
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

                    object_after_micro = self._get_observed_object_pos(
                        target, stage_name="after_micro_lift"
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
                    attempt_log["micro_lift_validation"] = json_safe(micro_validation)
                    attempt_log["gripper_diagnostics_after_micro_lift"] = json_safe(
                        self.gripper.get_diagnostics()
                    )
                    attempt_log["snapshots"]["after_micro_lift_validation"] = pose_snapshot(
                        self, target, "after_micro_lift_validation"
                    )
                    attempt_log["events"].append(
                        make_event(
                            "micro_lift_validation_done",
                            success=micro_validation.get("success"),
                            reasons=micro_validation.get("reasons", []),
                        )
                    )

                    if not micro_validation["success"]:
                        print("[Executor] ❌ Micro-lift validation failed.")
                        for reason in micro_validation["reasons"]:
                            print(f"  - {reason}")
                        attempt_log["failure_reason"] = "micro_lift_validation_failed"

                        decision = self.retry_policy.decide(trial_log, attempt_log)
                        attempt_log["retry_decision"] = json_safe(decision)
                        trial_log["attempts"].append(attempt_log)

                        if decision["retry"] and attempt < max_attempts - 1:
                            print(f"[Executor] RetryPolicy: {decision['reason']}")
                            current_retry_adjustments = decision.get("adjustments", {})
                            await self._recover_to_safe_for_retry(
                                pre_grasp=pre_grasp,
                                safe_above=safe_above,
                                from_micro_lift=True,
                            )
                            continue

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
                object_pos_before_lift = self._get_observed_object_pos(target, stage_name="before_full_lift")

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

            # ──── Fallback: no object detected after close. ────
            print("[Executor] ❌ No object detected after close.")
            attempt_log["failure_reason"] = "close_validation_failed"
            decision = self.retry_policy.decide(trial_log, attempt_log)
            attempt_log["retry_decision"] = json_safe(decision)
            trial_log["attempts"].append(attempt_log)

            if decision["retry"] and attempt < max_attempts - 1:
                print(f"[Executor] RetryPolicy: {decision['reason']}")
                current_retry_adjustments = decision.get("adjustments", {})
                await self._recover_to_safe_for_retry(
                    pre_grasp=pre_grasp,
                    safe_above=safe_above,
                    from_micro_lift=False,
                )
                continue

            print("[Executor] ❌ All grasp attempts failed.")
            trial_log["trial_success"] = False
            trial_log["final_reason"] = "all_attempts_failed_no_object_after_close"
            self._last_trial_log = trial_log
            await self._hold_for_inspection()
            return False