"""adaptive_safety_monitor.py

Deformation-aware safety monitor

It is a reactive safety layer that fuses:

  1) measured gripper finger-joint effort from ForceObserver, and
  2) live soft-object shape from SoftObjectObserver. (deformation and shape )

- It returns a small decision dictionary that the scalar admittance controller can
respect.  
- This basically is the reactive inhibitor / releaser layer: it can ask the motor 
schema to relax/open when a soft object is being compressed too much, even if the 
force target would suggest closing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional


def _ffloat(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _safe_min(vals: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    return min(clean) if clean else None


def _safe_max(vals: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    return max(clean) if clean else None


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1.0e-12:
        return None
    return numerator / denominator


def _bbox_size_from_obs(soft_obs: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return x/y/z size from direct fields or nested bbox fields."""
    width_x = _ffloat(soft_obs.get("width_x_m"), None)
    width_y = _ffloat(soft_obs.get("width_y_m"), None)
    height = _ffloat(soft_obs.get("height_m"), None)

    if width_x is not None and width_y is not None and height is not None:
        return width_x, width_y, height

    for key in ("simulation_bbox", "simulation_points_bbox", "collision_bbox", "visible_bbox", "wrapper_bbox"):
        bbox = soft_obs.get(key) or {}
        size = bbox.get("size") if isinstance(bbox, dict) else None
        if isinstance(size, (list, tuple)) and len(size) >= 3:
            width_x = width_x if width_x is not None else _ffloat(size[0], None)
            width_y = width_y if width_y is not None else _ffloat(size[1], None)
            height = height if height is not None else _ffloat(size[2], None)
            if width_x is not None and width_y is not None and height is not None:
                break

    return width_x, width_y, height


class AdaptiveSafetyMonitor:
    """Fuse measured effort and deformable shape into a safety/reaction decision."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def _nominal_dimensions(self, soft_obs: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Resolve nominal object dimensions used for ratio calculations.

        SoftObjectObserver may provide nominal_width_m but not nominal_height_m.
        For a cube-like benchmark object, falling back to nominal_width for height
        is physically reasonable and makes the monitor useful without requiring a
        new asset field.
        """
        cfg = self.config
        soft_obs = soft_obs or {}

        nominal_width = _ffloat(
            soft_obs.get("nominal_width_m"),
            _ffloat(cfg.get("adaptive_safety_nominal_width_m"), _ffloat(cfg.get("soft_object_nominal_width_m"), 0.04)),
        )
        nominal_depth = _ffloat(
            soft_obs.get("nominal_depth_m"),
            _ffloat(cfg.get("adaptive_safety_nominal_depth_m"), nominal_width),
        )
        nominal_height = _ffloat(
            soft_obs.get("nominal_height_m"),
            _ffloat(cfg.get("adaptive_safety_nominal_height_m"), nominal_width),
        )
        return {
            "nominal_width_x_m": nominal_width,
            "nominal_width_y_m": nominal_depth,
            "nominal_height_m": nominal_height,
        }

    def assess(
        self,
        *,
        soft_obs: Optional[Dict[str, Any]],
        effort_obs: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        cfg = self.config

        max_effort = _ffloat(
            cfg.get("adaptive_safety_max_effort_sim", cfg.get("adaptive_effort_max_sim", 1.20)),
            1.20,
        )
        warn_effort = _ffloat(
            cfg.get("adaptive_safety_warn_effort_sim", 0.85 * max_effort),
            0.85 * max_effort,
        )

        # Deformation thresholds.
        max_compression = _ffloat(cfg.get("adaptive_safety_max_compression_ratio", 0.25), 0.25)
        warn_compression = _ffloat(cfg.get("adaptive_safety_warn_compression_ratio", 0.18), 0.18)
        min_height_ratio = _ffloat(cfg.get("adaptive_safety_min_height_ratio", 0.70), 0.70)
        warn_height_ratio = _ffloat(cfg.get("adaptive_safety_warn_height_ratio", 0.85), 0.85)
        max_lateral_expansion = _ffloat(cfg.get("adaptive_safety_max_lateral_expansion_ratio", 1.40), 1.40)
        warn_lateral_expansion = _ffloat(cfg.get("adaptive_safety_warn_lateral_expansion_ratio", 1.20), 1.20)
        table_penetration_limit = _ffloat(cfg.get("adaptive_safety_max_table_penetration_m", 0.006), 0.006)
        warn_deformation_score = _ffloat(cfg.get("adaptive_safety_warn_deformation_score", 0.60), 0.60)
        max_deformation_score = _ffloat(cfg.get("adaptive_safety_max_deformation_score", 1.00), 1.00)

        effort = None
        contact_like = False
        if effort_obs:
            effort = _ffloat(effort_obs.get("grip_effort_sim"), None)
            contact_like = bool(effort_obs.get("contact_like_effort", False))

        height_ratio = None
        width_ratio_x = None
        width_ratio_y = None
        width_x_m = None
        width_y_m = None
        height_m = None
        min_shape_ratio = None
        max_width_ratio = None
        compression_x = None
        compression_y = None
        height_loss_ratio = None
        compression_ratio = None
        lateral_expansion_ratio = None
        volume_ratio_est = None
        size_change_abs_max = None
        table_gap = None
        pose_source = None
        soft_available = False
        nominal = self._nominal_dimensions(soft_obs if isinstance(soft_obs, dict) else None)

        if soft_obs and isinstance(soft_obs, dict):
            soft_available = soft_obs.get("center") is not None or soft_obs.get("pose_source") is not None
            pose_source = soft_obs.get("pose_source")
            width_x_m, width_y_m, height_m = _bbox_size_from_obs(soft_obs)

            width_ratio_x = _ffloat(soft_obs.get("width_ratio_x"), None)
            width_ratio_y = _ffloat(soft_obs.get("width_ratio_y"), None)
            height_ratio = _ffloat(soft_obs.get("deformation_ratio_z"), None)

            if width_ratio_x is None:
                width_ratio_x = _ratio(width_x_m, nominal.get("nominal_width_x_m"))
            if width_ratio_y is None:
                width_ratio_y = _ratio(width_y_m, nominal.get("nominal_width_y_m"))
            if height_ratio is None:
                height_ratio = _ratio(height_m, nominal.get("nominal_height_m"))

            table_gap = _ffloat(soft_obs.get("table_gap_m", soft_obs.get("bottom_clearance_m")), None)
            min_shape_ratio = _safe_min([height_ratio, width_ratio_x, width_ratio_y])
            max_width_ratio = _safe_max([width_ratio_x, width_ratio_y])

            compression_x = max(0.0, 1.0 - width_ratio_x) if width_ratio_x is not None else None
            compression_y = max(0.0, 1.0 - width_ratio_y) if width_ratio_y is not None else None
            height_loss_ratio = max(0.0, 1.0 - height_ratio) if height_ratio is not None else None
            compression_ratio = _safe_max([compression_x, compression_y, height_loss_ratio])
            lateral_expansion_ratio = max(0.0, (max_width_ratio or 1.0) - 1.0) if max_width_ratio is not None else None
            if height_ratio is not None and width_ratio_x is not None and width_ratio_y is not None:
                volume_ratio_est = height_ratio * width_ratio_x * width_ratio_y
            size_change_abs_max = _safe_max([
                abs(width_ratio_x - 1.0) if width_ratio_x is not None else None,
                abs(width_ratio_y - 1.0) if width_ratio_y is not None else None,
                abs(height_ratio - 1.0) if height_ratio is not None else None,
            ])

        # A dimensionless severity score used for explanation and later comparison.
        # 0 ≈ no deformation risk, 1 ≈ reaches configured safety limit.
        norm_compression = (compression_ratio / max_compression) if (compression_ratio is not None and max_compression and max_compression > 0) else 0.0
        norm_height_loss = (height_loss_ratio / max(1.0 - min_height_ratio, 1.0e-9)) if height_loss_ratio is not None else 0.0
        norm_lateral = (lateral_expansion_ratio / max(max_lateral_expansion - 1.0, 1.0e-9)) if lateral_expansion_ratio is not None else 0.0
        norm_effort = (effort / max_effort) if (effort is not None and max_effort and max_effort > 0) else 0.0
        deformation_score = max(0.0, norm_compression, norm_height_loss, norm_lateral)
        combined_risk_score = max(deformation_score, 0.75 * norm_effort)

        warnings = []
        unsafe_reasons = []

        effort_state = "unknown"
        if effort is not None:
            if max_effort is not None and effort >= max_effort:
                unsafe_reasons.append(f"effort {effort:.4f} >= max {max_effort:.4f}")
                effort_state = "unsafe_high"
            elif warn_effort is not None and effort >= warn_effort:
                warnings.append(f"effort {effort:.4f} >= warn {warn_effort:.4f}")
                effort_state = "warning_high"
            elif contact_like:
                effort_state = "contact_like_safe"
            else:
                effort_state = "low_or_no_contact"

        deformation_state = "unknown"
        if compression_ratio is not None:
            if compression_ratio >= max_compression:
                unsafe_reasons.append(f"compression_ratio {compression_ratio:.4f} >= max {max_compression:.4f}")
                deformation_state = "unsafe_compression"
            elif compression_ratio >= warn_compression:
                warnings.append(f"compression_ratio {compression_ratio:.4f} >= warn {warn_compression:.4f}")
                deformation_state = "warning_compression"
            else:
                deformation_state = "safe"

        if height_ratio is not None:
            if height_ratio <= min_height_ratio:
                unsafe_reasons.append(f"height_ratio {height_ratio:.4f} <= min {min_height_ratio:.4f}")
                deformation_state = "unsafe_height_loss"
            elif height_ratio <= warn_height_ratio:
                warnings.append(f"height_ratio {height_ratio:.4f} <= warn {warn_height_ratio:.4f}")
                if deformation_state == "safe":
                    deformation_state = "warning_height_loss"

        if lateral_expansion_ratio is not None:
            if max_width_ratio is not None and max_width_ratio >= max_lateral_expansion:
                warnings.append(f"lateral_expansion_ratio {max_width_ratio:.4f} >= warn {max_lateral_expansion:.4f}")
                if deformation_state == "safe":
                    deformation_state = "warning_lateral_expansion"
            elif max_width_ratio is not None and max_width_ratio >= warn_lateral_expansion:
                warnings.append(f"lateral_expansion_ratio {max_width_ratio:.4f} >= warn {warn_lateral_expansion:.4f}")
                if deformation_state == "safe":
                    deformation_state = "warning_lateral_expansion"

        if deformation_score >= max_deformation_score:
            unsafe_reasons.append(f"deformation_score {deformation_score:.4f} >= max {max_deformation_score:.4f}")
            deformation_state = "unsafe_score"
        elif deformation_score >= warn_deformation_score:
            warnings.append(f"deformation_score {deformation_score:.4f} >= warn {warn_deformation_score:.4f}")
            if deformation_state == "safe":
                deformation_state = "warning_score"

        # table_gap < 0 means object bbox bottom is below table top.  Some tiny
        # penetration is expected in PhysX.  Too much is a physical plausibility issue.
        if table_gap is not None and table_gap < -abs(table_penetration_limit):
            warnings.append(f"table penetration {table_gap:.4f} m exceeds {table_penetration_limit:.4f} m")

        unsafe = len(unsafe_reasons) > 0
        warn = (not unsafe) and len(warnings) > 0

        if unsafe:
            deformation_terms = ("compression", "height", "deformation", "lateral")
            recommended_action = (
                "relax_open_deformation"
                if any(any(term in r for term in deformation_terms) for r in unsafe_reasons)
                else "relax_open_effort"
            )
            safety_state = "unsafe"
            safe_to_continue = False
        elif warn:
            # For warnings, keep going but expose the reason.  The executor can
            # later choose whether to relax on warnings too.
            recommended_action = "continue_with_caution"
            safety_state = "warning"
            safe_to_continue = True
        else:
            recommended_action = "continue"
            safety_state = "safe"
            safe_to_continue = True

        return {
            "observer": "AdaptiveSafetyMonitor",
            "available": True,
            "phase": "3.3_deformation_severity_index",
            "safe_to_continue": safe_to_continue,
            "unsafe": unsafe,
            "warning": warn,
            "safety_state": safety_state,
            "recommended_action": recommended_action,
            "reasons": unsafe_reasons,
            "warnings": warnings,
            "effort_sim": effort,
            "effort_state": effort_state,
            "contact_like_effort": contact_like,
            "pose_source": pose_source,
            "soft_observation_available": soft_available,
            "nominal_dimensions_m": nominal,
            "measured_dimensions_m": {
                "width_x_m": width_x_m,
                "width_y_m": width_y_m,
                "height_m": height_m,
            },
            "height_ratio": height_ratio,
            "width_ratio_x": width_ratio_x,
            "width_ratio_y": width_ratio_y,
            "min_shape_ratio": min_shape_ratio,
            "max_width_ratio": max_width_ratio,
            "compression_x_ratio": compression_x,
            "compression_y_ratio": compression_y,
            "height_loss_ratio": height_loss_ratio,
            "compression_ratio_est": compression_ratio,
            "lateral_expansion_ratio": lateral_expansion_ratio,
            "volume_ratio_est": volume_ratio_est,
            "size_change_abs_max": size_change_abs_max,
            "deformation_state": deformation_state,
            "deformation_score": deformation_score,
            "combined_risk_score": combined_risk_score,
            "table_gap_m": table_gap,
            "thresholds": {
                "max_effort_sim": max_effort,
                "warn_effort_sim": warn_effort,
                "max_compression_ratio": max_compression,
                "warn_compression_ratio": warn_compression,
                "min_height_ratio": min_height_ratio,
                "warn_height_ratio": warn_height_ratio,
                "max_lateral_expansion_ratio": max_lateral_expansion,
                "warn_lateral_expansion_ratio": warn_lateral_expansion,
                "max_table_penetration_m": table_penetration_limit,
                "warn_deformation_score": warn_deformation_score,
                "max_deformation_score": max_deformation_score,
            },
            "context": context,
            "interpretation": (
                "deformation-aware monitor using measured finger-joint effort plus live soft-object dimensions; "
                "compression/deformation severity can inhibit the admittance motor schema"
            ),
        }