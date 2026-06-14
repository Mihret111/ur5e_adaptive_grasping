"""Trial-level diagnostic helper

json oriented helpers to  collect all evidence needed to diagnose grasp failures and to feed the RT2 
notebook without changing the actual robot behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time
import math

try:
    import numpy as np
except Exception:  # pragma: no cover - Isaac normally provides numpy
    np = None

def json_safe(value: Any) -> Any:
    """Recursively convert common Isaac/numpy/Python objects to JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    if np is not None:
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, np.generic):
            return json_safe(value.item())

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    # pxr vectors/matrices often support iteration; try that before str().
    try:
        return [json_safe(v) for v in value]
    except Exception:
        return str(value)


def make_event(stage: str, **data: Any) -> Dict[str, Any]:
    """Create a timestamped event record."""
    return json_safe({"t_unix": time.time(), "stage": stage, **data})


def selected_config_snapshot(config: dict) -> Dict[str, Any]:
    """Keep the analysis-relevant config values in every validation log."""
    keys = [
        "num_trials",
        "grip_mode",
        "finger_capture_axis_local",
        "enable_preclose_geometry_gate",
        "enable_preclose_diagnostics",
        "enable_micro_lift_test",
        "enable_lift_test",
        "plan_retreat_during_pick",
        "max_grasp_attempts",
        "arm_home_deg",
        "pre_grasp_above_m",
        "grasp_z_offset",
        "micro_lift_height_m",
        "micro_lift_duration",
        "micro_lift_steps",
        "post_close_hold_seconds",
        "post_micro_lift_settle_seconds",
        "min_success_micro_lift_delta_m",
        "min_commanded_micro_lift_delta_m",
        "min_micro_lift_following_ratio",
        "max_micro_lift_relative_drift_m",
        "max_preclose_grasp_centre_xy_error_m",
        "min_preclose_vertical_overlap_m",
        "max_preclose_flange_tracking_error_m",
        "min_grip_force",
        "max_grip_force",
        "force_safety_factor",
        "gripper_default_force",
        "gripper_hold_extra_close",
        "gripper_hold_force_ratio",
        "gripper_approach_speed",
        "gripper_squeeze_ticks",
        "collision_check_margin",
        "arm_body_clearance",
        "planner_path_sample_count",
        "planner_link_sample_count",
    ]
    snap = {k: config.get(k) for k in keys if k in config}

    grip_mode = config.get("grip_mode", "outwards")
    snap["active_grip_mode"] = grip_mode
    snap["active_grip_config"] = config.get(f"{grip_mode}_grip", {})
    return json_safe(snap)


def _shape_grip_dimension_m(target: dict) -> Optional[float]:
    """Estimate the dimension that the parallel jaw must span."""
    if target is None:
        return None

    if target.get("grip_dim_mm") not in (None, "unknown"):
        try:
            return float(target.get("grip_dim_mm")) / 1000.0
        except Exception:
            pass

    shape = target.get("shape", "")
    if shape == "Cube":
        return float(target.get("size", 0.0)) or None
    if shape == "Rectangle":
        # The intended policy is width-first for rectangles.
        return float(target.get("width", 0.0)) or None
    if shape == "Box":
        sx, sy, *_ = target.get("size_xyz", (0.0, 0.0, 0.0))
        d = min(float(sx), float(sy))
        return d or None
    if shape in ("Cylinder", "Disc", "Sphere"):
        if target.get("radius") is not None:
            return 2.0 * float(target.get("radius"))
    return None


def grip_feasibility(target: dict, config: dict) -> Dict[str, Any]:
    """Check whether the object span is inside the configured 2FG7 grip range."""
    grip_mode = config.get("grip_mode", "outwards")
    grip_cfg = config.get(f"{grip_mode}_grip", {})
    grip_min = float(grip_cfg.get("grip_range_min", 0.035))
    grip_max = float(grip_cfg.get("grip_range_max", 0.073))
    dim = _shape_grip_dimension_m(target)

    reasons = []
    within = None
    if dim is None:
        reasons.append("object grip dimension unavailable")
    else:
        within = grip_min <= dim <= grip_max
        if dim < grip_min:
            reasons.append(
                f"object grip dimension {dim:.4f} m below gripper minimum {grip_min:.4f} m"
            )
        if dim > grip_max:
            reasons.append(
                f"object grip dimension {dim:.4f} m above gripper maximum {grip_max:.4f} m"
            )

    return json_safe(
        {
            "grip_mode": grip_mode,
            "grip_range_min_m": grip_min,
            "grip_range_max_m": grip_max,
            "estimated_object_grip_dim_m": dim,
            "estimated_object_grip_dim_mm": None if dim is None else dim * 1000.0,
            "within_configured_range": within,
            "margin_to_min_m": None if dim is None else dim - grip_min,
            "margin_to_max_m": None if dim is None else grip_max - dim,
            "reasons": reasons,
        }
    )


def compact_pick_result(pick_result: Optional[dict]) -> Optional[Dict[str, Any]]:
    """Store only the analysis-relevant parts of a pick_result."""
    if not pick_result:
        return None
    joints = pick_result.get("joints", {}) or {}
    flange_targets = pick_result.get("flange_targets", {}) or {}
    return json_safe(
        {
            "planning_failed": pick_result.get("planning_failed", False),
            "target_force_n": pick_result.get("target_force_n"),
            "grasp_strategy": pick_result.get("grasp_strategy"),
            "object_height": pick_result.get("object_height"),
            "height_info": pick_result.get("height_info"),
            "ik_meta": pick_result.get("ik_meta"),
            "waypoint_names": list(joints.keys()),
            "flange_targets": flange_targets,
            "joint_targets_deg": joints,
        }
    )


def pose_snapshot(executor: Any, target: dict, stage: str) -> Dict[str, Any]:
    """Capture a compact state snapshot for diagnosis."""
    prim_path = target.get("prim_path") if target else None
    object_pos = None
    try:
        object_pos = executor._get_prim_world_pos(prim_path)
    except Exception as e:
        object_pos = f"unavailable: {e}"

    capture = None
    try:
        capture = executor.arm.get_calibrated_capture_geometry_world()
    except Exception as e:
        capture = {"error": str(e)}

    arm_status = None
    try:
        arm_status = executor.arm.get_status()
    except Exception as e:
        arm_status = {"error": str(e)}

    gripper_diag = None
    try:
        gripper_diag = executor.gripper.get_diagnostics()
    except Exception as e:
        gripper_diag = {"error": str(e)}

    return json_safe(
        {
            "stage": stage,
            "t_unix": time.time(),
            "object_world_pos": object_pos,
            "capture_geometry": capture,
            "arm_status": arm_status,
            "gripper_diagnostics": gripper_diag,
        }
    )
