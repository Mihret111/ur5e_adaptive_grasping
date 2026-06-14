"""Micro-lift grasp verification

Pure evaluation logic: no Isaac-Sim dependency.  The executor provides
measured object, flange, and calibrated grasp-centre positions before and
after a small lift.  Keeping this class independent from USD/PhysX makes it
straightforward to reuse with a real robot backend later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import math


class MicroLiftValidator:
    """Validate whether an object genuinely followed the gripper during lift-off."""

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _delta_z(after: Sequence[float], before: Sequence[float]) -> float:
        return float(after[2]) - float(before[2])

    def evaluate(
        self,
        *,
        object_pos_before: Optional[Sequence[float]],
        object_pos_after: Optional[Sequence[float]],
        flange_pos_before: Optional[Sequence[float]],
        flange_pos_after: Optional[Sequence[float]],
        grasp_centre_before: Optional[Sequence[float]],
        grasp_centre_after: Optional[Sequence[float]],
        gripper_has_object: bool,
    ) -> Dict[str, Any]:
        """Return structured evidence for the micro-lift checkpoint.

        A secure grasp should make the object move upward with the TCP while
        preserving approximately the same object-to-grasp-centre relationship.
        """
        result: Dict[str, Any] = {
            "stage": "after_micro_lift",
            "gripper_has_object": bool(gripper_has_object),
            "object_pos_before": object_pos_before,
            "object_pos_after": object_pos_after,
            "flange_pos_before": flange_pos_before,
            "flange_pos_after": flange_pos_after,
            "grasp_centre_before": grasp_centre_before,
            "grasp_centre_after": grasp_centre_after,
            "object_lift_delta_z_m": None,
            "flange_lift_delta_z_m": None,
            "following_ratio": None,
            "object_grasp_centre_distance_before_m": None,
            "object_grasp_centre_distance_after_m": None,
            "relative_grasp_drift_m": None,
            "success": False,
            "failure_classification": None,
            "reasons": [],
        }

        required = {
            "object_pos_before": object_pos_before,
            "object_pos_after": object_pos_after,
            "flange_pos_before": flange_pos_before,
            "flange_pos_after": flange_pos_after,
            "grasp_centre_before": grasp_centre_before,
            "grasp_centre_after": grasp_centre_after,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            result["reasons"].append(
                "missing micro-lift measurements: " + ", ".join(missing)
            )
            return result

        object_dz = self._delta_z(object_pos_after, object_pos_before)
        flange_dz = self._delta_z(flange_pos_after, flange_pos_before)
        following_ratio = object_dz / flange_dz if abs(flange_dz) > 1e-9 else None

        before_dist = self._distance(object_pos_before, grasp_centre_before)
        after_dist = self._distance(object_pos_after, grasp_centre_after)

        relative_before = [
            float(object_pos_before[i]) - float(grasp_centre_before[i])
            for i in range(3)
        ]
        relative_after = [
            float(object_pos_after[i]) - float(grasp_centre_after[i])
            for i in range(3)
        ]
        relative_drift = self._distance(relative_before, relative_after)

        result.update(
            {
                "object_lift_delta_z_m": object_dz,
                "flange_lift_delta_z_m": flange_dz,
                "following_ratio": following_ratio,
                "object_grasp_centre_distance_before_m": before_dist,
                "object_grasp_centre_distance_after_m": after_dist,
                "relative_grasp_drift_m": relative_drift,
            }
        )

        min_object_dz = float(
            self.config.get("min_success_micro_lift_delta_m", 0.015)
        )
        min_flange_dz = float(
            self.config.get("min_commanded_micro_lift_delta_m", 0.015)
        )
        min_following_ratio = float(
            self.config.get("min_micro_lift_following_ratio", 0.60)
        )
        max_relative_drift = float(
            self.config.get("max_micro_lift_relative_drift_m", 0.015)
        )

        result["thresholds"] = {
            "min_success_micro_lift_delta_m": min_object_dz,
            "min_commanded_micro_lift_delta_m": min_flange_dz,
            "min_micro_lift_following_ratio": min_following_ratio,
            "max_micro_lift_relative_drift_m": max_relative_drift,
        }

        if not gripper_has_object:
            result["reasons"].append("gripper_has_object false after micro-lift")

        if flange_dz < min_flange_dz:
            result["reasons"].append(
                f"flange did not execute micro-lift: dz={flange_dz:.4f} m < "
                f"{min_flange_dz:.4f} m"
            )

        if object_dz < min_object_dz:
            result["reasons"].append(
                f"object did not follow micro-lift: dz={object_dz:.4f} m < "
                f"{min_object_dz:.4f} m"
            )

        if following_ratio is None:
            result["reasons"].append("following ratio unavailable: flange dz is zero")
        elif following_ratio < min_following_ratio:
            result["reasons"].append(
                f"object following ratio too small: {following_ratio:.3f} < "
                f"{min_following_ratio:.3f}"
            )

        if relative_drift > max_relative_drift:
            result["reasons"].append(
                f"object drifted relative to grasp centre: {relative_drift:.4f} m > "
                f"{max_relative_drift:.4f} m"
            )

        result["success"] = len(result["reasons"]) == 0

        if result["success"]:
            result["failure_classification"] = None
        elif not gripper_has_object:
            result["failure_classification"] = "GRIPPER_LOST_OBJECT"
        elif flange_dz < min_flange_dz:
            result["failure_classification"] = "ARM_DID_NOT_EXECUTE_MICRO_LIFT"
        elif object_dz < min_object_dz and relative_drift > max_relative_drift:
            result["failure_classification"] = "NO_SECURE_CAPTURE_OBJECT_DID_NOT_FOLLOW"
        elif following_ratio is not None and following_ratio < min_following_ratio:
            result["failure_classification"] = "PARTIAL_SLIP_OR_WEAK_CAPTURE"
        elif relative_drift > max_relative_drift:
            result["failure_classification"] = "OBJECT_DRIFTED_IN_GRIPPER"
        else:
            result["failure_classification"] = "MICRO_LIFT_VALIDATION_FAILED_UNKNOWN"

        return result