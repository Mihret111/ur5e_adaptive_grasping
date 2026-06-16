# modules/retry_policy.py

class RetryPolicy:
    """
    Failure-aware retry policy for adaptive grasping.

    Cognitive-architecture role:
      - It does not move the robot.
      - It reads execution-monitoring evidence from the previous attempt.
      - It selects the next motor-schema parameters.

    In simple words:
      the executor acts, the validators diagnose, this policy decides.
    """

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Basic object interpretation helpers
    # ------------------------------------------------------------------
    def _is_fragile_material(self, material: str) -> bool:
        material = str(material or "").lower()
        fragile_keywords = ["ceramic", "glass", "fragile", "thin", "delicate"]
        return any(k in material for k in fragile_keywords)

    def _is_thin_object(self, target: dict, attempt_log: dict) -> bool:
        """Return True when lowering the grasp could push fingers into the table."""
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

    def _get_plan_height_info(self, attempt_log: dict) -> dict:
        """Get height planning information from the most recent available plan."""
        plan = (
            attempt_log.get("planning_replanned_after_safe_above")
            or attempt_log.get("planning_initial")
            or {}
        )
        return plan.get("height_info", {}) or {}

    def _is_near_table_grasp(self, attempt_log: dict) -> bool:
        """Detect whether the planned grasp is close to the table-clearance limit."""
        h = self._get_plan_height_info(attempt_log)

        if bool(h.get("grasp_z_was_clamped", False)):
            return True

        tool0_z = h.get("tool0_z")
        min_tool0_z = h.get("min_tool0_z_table_clearance")

        try:
            extra_clearance = float(tool0_z) - float(min_tool0_z)
        except Exception:
            return False

        threshold = float(
            self.config.get("retry_near_table_extra_clearance_threshold_m", 0.003)
        )

        return extra_clearance <= threshold

    # ------------------------------------------------------------------
    # Contact interpretation helpers
    # ------------------------------------------------------------------
    def _get_close_diagnostics(self, attempt_log: dict) -> dict:
        """Return gripper diagnostics from the close phase.

        Prefer diagnostics after the final close validation. If that is missing,
        fall back to the close-resolution snapshot.
        """
        diag = attempt_log.get("gripper_diagnostics_after_close")
        if diag:
            return diag or {}

        resolution = attempt_log.get("gripper_close_resolution", {}) or {}
        return resolution.get("gripper_diagnostics", {}) or {}

    def _contact_window(self, diag: dict):
        """Return expected contact window [low, high] if it can be reconstructed."""
        window = diag.get("expected_contact_window_m")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            try:
                return float(window[0]), float(window[1])
            except Exception:
                pass

        expected = diag.get("expected_contact_position_m")
        if expected is None:
            return None

        try:
            expected = float(expected)
        except Exception:
            return None

        lower_tol = float(diag.get(
            "contact_position_tolerance_m",
            self.config.get("gripper_contact_position_tolerance_m", 0.004),
        ))
        upper_tol = float(self.config.get(
            "gripper_contact_position_upper_tolerance_m",
            lower_tol,
        ))

        closed_pos = float(self.config.get("gripper_joint_closed_pos_m", 0.019))
        open_pos = float(self.config.get("gripper_joint_open_pos_m", 0.0))

        low = max(open_pos, expected - lower_tol)
        high = min(closed_pos, expected + upper_tol)
        return low, high

    def _classify_contact_quality(self, attempt_log: dict) -> dict:
        """Classify whether the previous close contact looked physically meaningful.

        This turns low-level gripper readings into a symbolic diagnosis usable by
        action selection.
        """
        diag = self._get_close_diagnostics(attempt_log)

        result = {
            "quality": "unknown",
            "contact_position_m": None,
            "expected_contact_position_m": diag.get("expected_contact_position_m"),
            "expected_contact_window_m": None,
            "has_object": diag.get("has_object"),
            "had_contact": diag.get("had_contact"),
            "close_failure_reason": diag.get("close_failure_reason"),
            "opening_m": diag.get("opening_m"),
            "normalized_contact_in_window": None,
        }

        if not diag:
            result["quality"] = "missing_close_diagnostics"
            return result

        contact = diag.get("contact_position")
        window = self._contact_window(diag)

        if window is not None:
            result["expected_contact_window_m"] = [window[0], window[1]]

        if contact is None:
            reason = diag.get("close_failure_reason")
            if reason in (
                "ignored_implausible_early_stall",
                "closing_timeout_no_plausible_contact",
            ):
                result["quality"] = "no_plausible_contact_or_early_stall"
            elif diag.get("has_object") is False:
                result["quality"] = "no_contact"
            else:
                result["quality"] = "no_contact_position_logged"
            return result

        try:
            contact = float(contact)
        except Exception:
            result["quality"] = "invalid_contact_position"
            return result

        result["contact_position_m"] = contact

        if window is None:
            result["quality"] = "contact_logged_without_expected_window"
            return result

        low, high = window

        if contact < low:
            result["quality"] = "contact_too_early_or_not_reached"
            return result

        if contact > high:
            result["quality"] = "contact_too_far_closed"
            return result

        width = max(1e-9, high - low)
        alpha = (contact - low) / width
        result["normalized_contact_in_window"] = alpha

        if alpha < 0.25:
            result["quality"] = "plausible_contact_low_edge"
        elif alpha > 0.75:
            result["quality"] = "plausible_contact_high_edge"
        else:
            result["quality"] = "plausible_contact_centered"

        return result

    def _safe_z_delta(self, requested_delta: float, thin_object: bool, near_table: bool) -> float:
        """Clamp retry lowering for risky near-table/thin cases."""
        if requested_delta < 0.0 and (thin_object or near_table):
            return 0.0
        return float(requested_delta)

    # ------------------------------------------------------------------
    # Main policy
    # ------------------------------------------------------------------
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
        near_table = self._is_near_table_grasp(attempt_log)
        contact_info = self._classify_contact_quality(attempt_log)
        contact_quality = contact_info.get("quality")

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
                "thin_object": thin_object,
                "near_table_grasp": near_table,
                "micro_lift_classification": micro.get("failure_classification"),
                "contact_quality": contact_quality,
                "contact_info": contact_info,
            },
        }

        # Impossible gripper range: do not repeat the same pick primitive.
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

            # If the gripper stalled early near the open side, lowering is usually
            # the wrong response. On near-table/thin cases, raise slightly to free
            # the fingertips from table/object-edge contact.
            if contact_quality in (
                "no_plausible_contact_or_early_stall",
                "contact_too_early_or_not_reached",
            ):
                z_delta = float(
                    self.config.get("retry_early_stall_raise_z_delta_m", 0.002)
                ) if (thin_object or near_table) else 0.0
                decision["reason"] = "retry_close_failed_no_plausible_contact"
            elif contact_quality == "contact_too_far_closed":
                # The gripper closed too much compared with the object model:
                # likely pose/object mismatch. Refresh pose, do not force lower.
                z_delta = 0.0
                decision["reason"] = "retry_close_failed_contact_too_far_closed"
            else:
                z_delta = self._safe_z_delta(
                    float(self.config.get("retry_close_fail_grasp_z_delta_m", -0.003)),
                    thin_object,
                    near_table,
                )
                decision["reason"] = "retry_close_failed_geometry_first"

            decision["adjustments"] = {
                "refresh_object_pose": True,
                "grasp_z_delta_m": z_delta,
                "force_scale": 1.0 if fragile else float(
                    self.config.get("retry_close_fail_force_scale", 1.05)
                ),
                "hold_settle_extra_s": float(
                    self.config.get("retry_close_fail_hold_extra_s", 0.3)
                ),
                "micro_lift_speed_scale": 1.0,
            }
            return decision

        if failure_reason == "micro_lift_validation_failed":
            cls = micro.get("failure_classification", "")

            plausible_contact = str(contact_quality).startswith("plausible_contact")
            bad_or_missing_contact = contact_quality in (
                "no_plausible_contact_or_early_stall",
                "no_contact",
                "contact_too_early_or_not_reached",
                "missing_close_diagnostics",
            )

            if cls == "NO_SECURE_CAPTURE_OBJECT_DID_NOT_FOLLOW":
                decision["retry"] = True

                if plausible_contact:
                    # The gripper contacted at the expected width, but the object
                    # did not follow. So the next attempt should not blindly lower.
                    # Prefer stronger/longer/slower capture, with only tiny lowering
                    # for non-thin, non-table-limited objects.
                    requested_z = float(
                        self.config.get("retry_plausible_no_follow_grasp_z_delta_m", -0.001)
                    )
                    z_delta = self._safe_z_delta(requested_z, thin_object, near_table)
                    decision["reason"] = "retry_no_follow_after_plausible_contact"
                elif bad_or_missing_contact:
                    z_delta = float(
                        self.config.get("retry_early_stall_raise_z_delta_m", 0.002)
                    ) if (thin_object or near_table) else 0.0
                    decision["reason"] = "retry_no_follow_but_contact_was_not_plausible"
                else:
                    z_delta = self._safe_z_delta(
                        float(self.config.get("retry_no_follow_grasp_z_delta_m", -0.004)),
                        thin_object,
                        near_table,
                    )
                    decision["reason"] = "retry_no_secure_capture"

                decision["adjustments"] = {
                    "refresh_object_pose": True,
                    "grasp_z_delta_m": z_delta,
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

                if plausible_contact:
                    requested_z = float(
                        self.config.get("retry_plausible_partial_slip_grasp_z_delta_m", 0.0)
                    )
                    z_delta = self._safe_z_delta(requested_z, thin_object, near_table)
                    decision["reason"] = "retry_partial_slip_after_plausible_contact"
                else:
                    z_delta = self._safe_z_delta(
                        float(self.config.get("retry_partial_slip_grasp_z_delta_m", -0.002)),
                        thin_object,
                        near_table,
                    )
                    decision["reason"] = "retry_partial_slip_or_weak_capture"

                decision["adjustments"] = {
                    "refresh_object_pose": False if plausible_contact else True,
                    "grasp_z_delta_m": z_delta,
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
                "grasp_z_delta_m": self._safe_z_delta(-0.002, thin_object, near_table),
                "force_scale": 1.0 if fragile else 1.05,
                "hold_settle_extra_s": 0.3,
                "micro_lift_speed_scale": 0.85,
            }
            return decision

        return decision