import math

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
        """Classify the previous close using gripper-native pair diagnostics.

        The balanced gripper already computes a pair-aware contact_quality from
        mean finger closure, asymmetry, and the expected contact window. The
        retry policy must not rebuild the diagnosis from the old scalar
        contact_position, otherwise action selection can disagree with the
        execution monitor.
        """
        diag = self._get_close_diagnostics(attempt_log)

        result = {
            "quality": "unknown",
            "source": "missing",
            "contact_position_m": None,
            "avg_position_m": None,
            "asymmetry_m": None,
            "expected_contact_position_m": diag.get("expected_contact_position_m"),
            "expected_contact_window_m": None,
            "has_object": diag.get("has_object"),
            "had_contact": diag.get("had_contact"),
            "close_failure_reason": diag.get("close_failure_reason"),
            "opening_m": diag.get("opening_m"),
            "normalized_contact_in_window": None,
            "gripper_contact_quality_raw": diag.get("contact_quality"),
        }

        if not diag:
            result["quality"] = "missing_close_diagnostics"
            return result

        window = self._contact_window(diag)
        if window is not None:
            result["expected_contact_window_m"] = [window[0], window[1]]

        # Preferred path: use the balanced gripper's native pair-aware quality.
        raw_quality = diag.get("contact_quality")
        if isinstance(raw_quality, dict):
            q = str(raw_quality.get("quality") or "unknown")
            result["source"] = "balanced_gripper_pair_quality"
            result["quality"] = q
            result["avg_position_m"] = raw_quality.get("avg_position_m")
            result["asymmetry_m"] = raw_quality.get("asymmetry_m")
            result["expected_contact_window_m"] = (
                raw_quality.get("window_m")
                or result["expected_contact_window_m"]
            )
            result["contact_position_m"] = raw_quality.get(
                "min_position_m",
                diag.get("contact_position"),
            )
            result["normalized_contact_in_window"] = raw_quality.get(
                "normalized_avg_in_window"
            )
            result["pair_plausible"] = raw_quality.get("plausible")
            result["pair_clean"] = raw_quality.get("clean")
            result["avg_in_compressed_grace"] = raw_quality.get(
                "avg_in_compressed_grace"
            )
            result["compressed_high_m"] = raw_quality.get("compressed_high_m")
            return result

        # Backward-compatible fallback for older logs/controllers.
        contact = diag.get("contact_position")

        if contact is None:
            reason = diag.get("close_failure_reason")
            if reason in (
                "ignored_implausible_early_stall",
                "ignored_implausible_pair_stall",
                "closing_timeout_no_plausible_contact",
            ):
                result["quality"] = "no_plausible_contact_or_early_stall"
            elif diag.get("has_object") is False:
                result["quality"] = "no_contact"
            else:
                result["quality"] = "no_contact_position_logged"
            result["source"] = "legacy_scalar_no_contact"
            return result

        try:
            contact = float(contact)
        except Exception:
            result["quality"] = "invalid_contact_position"
            result["source"] = "legacy_scalar_invalid"
            return result

        result["source"] = "legacy_scalar_contact_position"
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
    # Object-motion / retry-context helpers
    # ------------------------------------------------------------------
    def _point_distance(self, a, b) -> float:
        """Euclidean distance between two 3D points, or 0 if unavailable."""
        if a is None or b is None:
            return 0.0
        try:
            return math.sqrt(
                (float(a[0]) - float(b[0])) ** 2
                + (float(a[1]) - float(b[1])) ** 2
                + (float(a[2]) - float(b[2])) ** 2
            )
        except Exception:
            return 0.0

    def _horizontal_distance(self, a, b) -> float:
        """Horizontal XY distance between two 3D points, or 0 if unavailable."""
        if a is None or b is None:
            return 0.0
        try:
            return math.sqrt(
                (float(a[0]) - float(b[0])) ** 2
                + (float(a[1]) - float(b[1])) ** 2
            )
        except Exception:
            return 0.0

    def _attempt_object_motion(self, attempt_log: dict) -> dict:
        """Summarize how much the object moved during the failed attempt.

        This is important because a retry after an object has rolled/slid should
        be treated as fresh re-perception, not as a tiny correction to the old
        grasp.
        """
        micro = attempt_log.get("micro_lift_validation", {}) or {}
        close = attempt_log.get("close_validation", {}) or {}

        before = None
        after = None
        source = "none"

        if micro.get("object_pos_before") is not None and micro.get("object_pos_after") is not None:
            before = micro.get("object_pos_before")
            after = micro.get("object_pos_after")
            source = "micro_lift_validation"
        elif close.get("object_pos_before") is not None and close.get("object_pos_after") is not None:
            before = close.get("object_pos_before")
            after = close.get("object_pos_after")
            source = "close_validation"

        total = self._point_distance(before, after)
        horizontal = self._horizontal_distance(before, after)
        dz = 0.0
        try:
            if before is not None and after is not None:
                dz = float(after[2]) - float(before[2])
        except Exception:
            dz = 0.0

        threshold = float(
            self.config.get("retry_object_displacement_fresh_pose_threshold_m", 0.015)
        )
        horizontal_threshold = float(
            self.config.get("retry_object_horizontal_displacement_fresh_pose_threshold_m", threshold)
        )

        significant = (total >= threshold) or (horizontal >= horizontal_threshold)
        disturbed_retry_block_threshold = float(
            self.config.get("retry_abort_object_displacement_threshold_m", threshold)
        )
        disturbed_retry_block_horizontal = float(
            self.config.get("retry_abort_object_horizontal_displacement_threshold_m", disturbed_retry_block_threshold)
        )
        disturbed_for_retry = (total >= disturbed_retry_block_threshold) or (horizontal >= disturbed_retry_block_horizontal)

        return {
            "source": source,
            "object_pos_before": before,
            "object_pos_after": after,
            "object_displacement_m": total,
            "object_horizontal_displacement_m": horizontal,
            "object_delta_z_m": dz,
            "fresh_pose_threshold_m": threshold,
            "fresh_pose_horizontal_threshold_m": horizontal_threshold,
            "significant_object_motion": significant,
            "disturbed_for_normal_retry": disturbed_for_retry,
            "retry_abort_object_displacement_threshold_m": disturbed_retry_block_threshold,
            "retry_abort_object_horizontal_displacement_threshold_m": disturbed_retry_block_horizontal,
        }

    def _is_plausible_contact_quality(self, quality: str) -> bool:
        """Return True for both old and balanced-gripper plausible labels."""
        q = str(quality or "")
        return q.startswith("plausible_contact") or q.startswith("plausible_")

    def _is_low_edge_contact(self, contact_info: dict) -> bool:
        """Detect contact at the lower/open edge of the expected contact window."""
        q = str(contact_info.get("quality", ""))
        if q == "plausible_contact_low_edge":
            return True

        alpha = contact_info.get("normalized_contact_in_window")
        try:
            return float(alpha) < float(
                self.config.get("retry_low_edge_contact_alpha", 0.25)
            )
        except Exception:
            return False

    def _is_borderline_micro_lift_failure(self, micro: dict) -> bool:
        """Detect almost-successful micro-lifts.

        If the object almost followed, lowering the grasp is usually not the
        right retry. Repeat from refreshed pose, hold longer, and lift slower.
        """
        if not micro:
            return False

        thresholds = micro.get("thresholds", {}) or {}
        min_dz = float(
            thresholds.get(
                "min_success_micro_lift_delta_m",
                self.config.get("min_success_micro_lift_delta_m", 0.015),
            )
        )
        min_ratio = float(
            thresholds.get(
                "min_micro_lift_following_ratio",
                self.config.get("min_micro_lift_following_ratio", 0.60),
            )
        )
        max_drift = float(
            thresholds.get(
                "max_micro_lift_relative_drift_m",
                self.config.get("max_micro_lift_relative_drift_m", 0.015),
            )
        )

        dz = micro.get("object_lift_delta_z_m", 0.0)
        ratio = micro.get("following_ratio", 0.0)
        drift = micro.get("relative_grasp_drift_m", 999.0)

        try:
            dz = float(dz)
            ratio = float(ratio)
            drift = float(drift)
        except Exception:
            return False

        dz_margin = float(self.config.get("retry_borderline_lift_delta_margin_m", 0.002))
        ratio_margin = float(self.config.get("retry_borderline_following_ratio_margin", 0.08))
        drift_margin = float(self.config.get("retry_borderline_drift_margin_m", 0.004))

        near_dz = dz >= (min_dz - dz_margin)
        near_ratio = ratio >= (min_ratio - ratio_margin)
        acceptable_drift = drift <= (max_drift + drift_margin)

        return near_dz and near_ratio and acceptable_drift

    # ------------------------------------------------------------------
    # Adaptive safety interpretation helpers
    # ------------------------------------------------------------------
    def _adaptive_safety_outcome(self, attempt_log: dict) -> dict:
        """Return compact adaptive-safety diagnosis from close regulation."""
        outcome = attempt_log.get("adaptive_safety_outcome") or {}
        if outcome:
            return outcome

        resolution = attempt_log.get("gripper_close_resolution") or {}
        reg = resolution.get("adaptive_effort_regulation") or {}
        summary = reg.get("adaptive_safety_summary") or {}
        return {
            "available": bool(reg),
            "requires_attempt_stop": reg.get("final_reason") in (
                "safety_open_limit_reached",
                "safety_abort_requested",
            ),
            "regulation_final_reason": reg.get("final_reason"),
            "safety_summary": summary,
        }

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
        object_motion = self._attempt_object_motion(attempt_log)
        significant_object_motion = bool(
            object_motion.get("significant_object_motion", False)
        )

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
                "object_motion_after_attempt": object_motion,
                "significant_object_motion": significant_object_motion,
                "adaptive_safety_outcome": self._adaptive_safety_outcome(attempt_log),
            },
        }

        # Impossible gripper range: do not repeat the same pick primitive.
        if str(grip_class).startswith("infeasible"):
            decision["reason"] = "object_not_grippable_with_current_2fg7_range"
            return decision

        if failure_reason == "adaptive_safety_relaxation_triggered":
            safety = self._adaptive_safety_outcome(attempt_log)

            if not bool(self.config.get("retry_adaptive_safety_allow_retry", True)):
                decision["retry"] = False
                decision["reason"] = "no_retry_adaptive_safety_retry_disabled"
                decision["diagnosis"]["adaptive_safety_outcome"] = safety
                return decision

            decision["retry"] = True
            decision["reason"] = "retry_after_adaptive_safety_relaxation_safer_grasp"

            # Unsafe deformation means the robot already had to relax/open.
            # The next attempt should be less aggressive and should reacquire
            # pose, because relaxation may leave the object shifted.
            force_scale_key = (
                "retry_adaptive_safety_fragile_force_scale"
                if (fragile or bool(target.get("fragile", False)))
                else "retry_adaptive_safety_force_scale"
            )
            decision["adjustments"] = {
                "refresh_object_pose": bool(self.config.get("retry_adaptive_safety_reacquire_pose", True)),
                "grasp_z_delta_m": 0.0,
                "force_scale": float(self.config.get(force_scale_key, 0.90)),
                "adaptive_effort_target_scale": float(
                    self.config.get("retry_adaptive_safety_effort_target_scale", 0.85)
                ),
                "hold_extra_close_delta_m": float(
                    self.config.get("retry_adaptive_safety_hold_extra_delta_m", -0.00025)
                ),
                "hold_settle_extra_s": float(
                    self.config.get("retry_adaptive_safety_hold_settle_extra_s", 0.5)
                ),
                "micro_lift_speed_scale": float(
                    self.config.get("retry_adaptive_safety_micro_lift_speed_scale", 0.70)
                ),
                "safety_recovery_retry": True,
            }
            decision["diagnosis"]["adaptive_safety_outcome"] = safety
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
                "too_open_or_early",
                "too_asymmetric",
            ):
                z_delta = float(
                    self.config.get("retry_early_stall_raise_z_delta_m", 0.002)
                ) if (thin_object or near_table) else 0.0
                decision["reason"] = "retry_close_failed_no_plausible_contact"
            elif contact_quality in ("contact_too_far_closed", "too_closed_or_missed"):
                # The pair average closed too much compared with the object model:
                # likely pose/object mismatch or a compressed/missed contact.
                # Refresh pose, do not force lower.
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

            if cls == "SOFT_OBJECT_MOTION_OBSERVER_UNRELIABLE":
                # Fail closed.  A retry should not drive back to a stale USD-bbox
                # pose after a deformable object has visibly moved but cannot be
                # reacquired by the current observer.  This is a perception
                # limitation, not a manipulation retry condition.
                decision["retry"] = False
                decision["reason"] = "no_retry_soft_object_pose_observer_unreliable"
                decision["adjustments"] = {
                    "refresh_object_pose": False,
                    "requires_reacquisition": True,
                    "recommended_next_step": "use runtime deformable/vision observer or reset object before retry",
                }
                return decision

            plausible_contact = self._is_plausible_contact_quality(contact_quality)
            bad_or_missing_contact = contact_quality in (
                "no_plausible_contact_or_early_stall",
                "no_contact",
                "contact_too_early_or_not_reached",
                "too_open_or_early",
                "too_asymmetric",
                "missing_close_diagnostics",
            )

            if bool(self.config.get("retry_block_after_disturbed_micro_lift", True)) and motion.get("disturbed_for_normal_retry"):
                decision["retry"] = False
                decision["reason"] = "no_retry_object_disturbed_after_failed_micro_lift"
                decision["adjustments"] = {
                    "object_motion": motion,
                    "recommended_next_step": "reset object or run fresh perception after a controlled retreat; do not chase disturbed soft object",
                }
                return decision

            if cls == "NO_SECURE_CAPTURE_OBJECT_DID_NOT_FOLLOW":
                decision["retry"] = True

                if plausible_contact:
                    # The gripper contacted at the expected width, but the object
                    # did not follow. So the next attempt should not blindly lower.
                    # For thin discs with low-edge contact, raise slightly: this
                    # often means the fingers touched a table/edge artifact before
                    # truly wrapping the object.
                    if thin_object and self._is_low_edge_contact(contact_info):
                        z_delta = float(
                            self.config.get("retry_low_edge_thin_raise_z_delta_m", 0.002)
                        )
                        decision["reason"] = "retry_thin_low_edge_contact_raise_grasp"
                    elif significant_object_motion:
                        z_delta = 0.0
                        decision["reason"] = "retry_no_follow_fresh_pose_after_object_motion"
                    else:
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
                    if significant_object_motion:
                        z_delta = 0.0
                        decision["reason"] = "retry_partial_slip_fresh_pose_after_object_motion"
                    else:
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
            # If the micro-lift almost passed, or the object moved significantly,
            # do not lower the grasp. Treat the next attempt as fresh perception.
            decision["retry"] = True
            borderline = self._is_borderline_micro_lift_failure(micro)
            if plausible_contact and (borderline or significant_object_motion):
                decision["reason"] = "retry_borderline_micro_lift_repeat_fresh_pose"
                z_delta = 0.0
            else:
                decision["reason"] = "retry_unknown_micro_lift_failure_cautious"
                z_delta = self._safe_z_delta(-0.002, thin_object, near_table)

            decision["adjustments"] = {
                "refresh_object_pose": True,
                "grasp_z_delta_m": z_delta,
                "force_scale": 1.0 if fragile else 1.05,
                "hold_settle_extra_s": 0.4 if borderline else 0.3,
                "micro_lift_speed_scale": 0.80 if borderline else 0.85,
                "fresh_reperception_retry": bool(borderline or significant_object_motion),
            }
            return decision

        return decision