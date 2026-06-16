# modules/retry_policy.py

class RetryPolicy:
    """
    Failure-aware retry policy for adaptive grasping.

    This module does not move the robot directly. It only decides whether
    another attempt is worth trying, why, and how the next attempt should be
    modified.
    """

    def __init__(self, config: dict):
        self.config = config

    def _is_fragile_material(self, material: str) -> bool:
        material = str(material or "").lower()
        fragile_keywords = ["ceramic", "glass", "fragile", "thin", "delicate"]
        return any(k in material for k in fragile_keywords)
    
    # additional rule for thin objects
    def _is_thin_object(self, target: dict, attempt_log: dict) -> bool:
        """Return True for thin objects where lowering the grasp risks table contact."""
        shape = str(target.get("shape", "")).lower()

        height = None
        plan = (
            attempt_log.get("planning_replanned_after_safe_above")
            or attempt_log.get("planning_initial")
            or {}
        )

        if plan:
            height = plan.get("object_height")

        if height is None:
            height = target.get("height")

        try:
            height = float(height)
        except Exception:
            height = None

        thin_threshold = float(
            self.config.get("retry_thin_object_height_threshold_m", 0.015)
        )

        if shape in ("disc", "disk"):
            return True

        if height is not None and height <= thin_threshold:
            return True

        return False

    def decide(self, trial_log: dict, attempt_log: dict) -> dict:
        failure_reason = attempt_log.get("failure_reason")

        target = trial_log.get("target", {}) or {}
        target_feas = attempt_log.get(
            "target_feasibility",
            trial_log.get("target_feasibility", {}) or {},
        )
        micro = attempt_log.get("micro_lift_validation", {}) or {}

        grip_class = target_feas.get("grip_feasibility_class", "unknown")
        material = target.get("material", "unknown")
        shape = target.get("shape", "unknown")
        fragile = self._is_fragile_material(material)
        thin_object = self._is_thin_object(target, attempt_log)

        decision = {
            "retry": False,
            "reason": "no_retry_rule_matched",
            "adjustments": {},
            "diagnosis": {
                "failure_reason": failure_reason,
                "grip_feasibility_class": grip_class,
                "shape": shape,
                "material": material,
                "fragile_material": fragile,
                "micro_lift_classification": micro.get("failure_classification"),
                "thin_object": thin_object,
            },
        }

        # Do not repeat a pick primitive if the gripper range makes it impossible.
        if str(grip_class).startswith("infeasible"):
            decision["reason"] = "object_not_grippable_with_current_2fg7_range"
            return decision

        if failure_reason == "preclose_geometry_gate_failed":
            decision["retry"] = True
            decision["reason"] = "retry_after_preclose_pose_refresh"
            decision["adjustments"] = {
                "refresh_object_pose": True,
                "grasp_z_delta_m": 0.0,
                "force_scale": 1.0,
                "hold_settle_extra_s": 0.2,
                "micro_lift_speed_scale": 1.0,
            }
            return decision

        if failure_reason == "close_validation_failed":
            decision["retry"] = True
            decision["reason"] = "retry_close_failed_geometry_first"
            decision["adjustments"] = {
                "refresh_object_pose": True,
                "grasp_z_delta_m": 0.0 if thin_object else float(
                    self.config.get("retry_close_fail_grasp_z_delta_m", -0.003)
                ),
                # For fragile materials, prefer geometry/speed correction first.
                "force_scale": 1.0 if fragile else 1.05,
                "hold_settle_extra_s": 0.3,
                "micro_lift_speed_scale": 1.0,
            }
            return decision

        if failure_reason == "micro_lift_validation_failed":
            cls = micro.get("failure_classification", "")

            if cls == "NO_SECURE_CAPTURE_OBJECT_DID_NOT_FOLLOW":
                decision["retry"] = True
                decision["reason"] = "retry_no_secure_capture"
                decision["adjustments"] = {
                    "refresh_object_pose": True,
                    "grasp_z_delta_m": 0.0 if thin_object else float(
                        self.config.get("retry_no_follow_grasp_z_delta_m", -0.004)
                    ),
                    "force_scale": 1.0 if fragile else float(
                        self.config.get("retry_no_follow_force_scale", 1.10)
                    ),
                    "hold_settle_extra_s": float(
                        self.config.get("retry_no_follow_hold_extra_s", 0.4)
                    ),
                    "micro_lift_speed_scale": float(
                        self.config.get("retry_no_follow_micro_lift_speed_scale", 0.80)
                    ),
                }
                return decision

            if cls == "PARTIAL_SLIP_OR_WEAK_CAPTURE":
                decision["retry"] = True
                decision["reason"] = "retry_partial_slip_or_weak_capture"
                decision["adjustments"] = {
                    "refresh_object_pose": False,
                    "grasp_z_delta_m": 0.0 if thin_object else float(
                        self.config.get("retry_partial_slip_grasp_z_delta_m", -0.002)
                    ),
                    "force_scale": 1.0 if fragile else float(
                        self.config.get("retry_partial_slip_force_scale", 1.15)
                    ),
                    "hold_settle_extra_s": float(
                        self.config.get("retry_partial_slip_hold_extra_s", 0.5)
                    ),
                    "micro_lift_speed_scale": float(
                        self.config.get("retry_partial_slip_micro_lift_speed_scale", 0.70)
                    ),
                }
                return decision

            # Unknown micro-lift failure: retry cautiously once.
            decision["retry"] = True
            decision["reason"] = "retry_unknown_micro_lift_failure_cautious"
            decision["adjustments"] = {
                "refresh_object_pose": True,
                "grasp_z_delta_m": 0.0 if thin_object else -0.002,
                "force_scale": 1.0 if fragile else 1.05,
                "hold_settle_extra_s": 0.3,
                "micro_lift_speed_scale": 0.85,
            }
            return decision

        return decision
