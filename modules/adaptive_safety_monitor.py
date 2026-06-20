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
from typing import Any, Dict, Optional


def _ffloat(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _safe_min(vals):
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    return min(clean) if clean else None


def _safe_max(vals):
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    return max(clean) if clean else None


class AdaptiveSafetyMonitor:
    """Fuse effort and deformation into a safety/reaction decision."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

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
        # Soft-object deformation thresholds.  Ratios come from SoftObjectObserver:
        #   deformation_ratio_z = current height / nominal height
        #   width_ratio_x/y     = current bbox width / nominal width
        # Compression estimate is 1 - min(width_ratio_x, width_ratio_y, height_ratio).
        max_compression = _ffloat(cfg.get("adaptive_safety_max_compression_ratio", 0.25), 0.25)
        warn_compression = _ffloat(cfg.get("adaptive_safety_warn_compression_ratio", 0.18), 0.18)
        min_height_ratio = _ffloat(cfg.get("adaptive_safety_min_height_ratio", 0.70), 0.70)
        max_lateral_expansion = _ffloat(cfg.get("adaptive_safety_max_lateral_expansion_ratio", 1.40), 1.40)
        table_penetration_limit = _ffloat(cfg.get("adaptive_safety_max_table_penetration_m", 0.006), 0.006)

        effort = None
        contact_like = False
        if effort_obs:
            effort = _ffloat(effort_obs.get("grip_effort_sim"), None)
            contact_like = bool(effort_obs.get("contact_like_effort", False))

        height_ratio = None
        width_ratio_x = None
        width_ratio_y = None
        min_shape_ratio = None
        max_width_ratio = None
        compression_ratio = None
        table_gap = None
        pose_source = None
        soft_available = False

        if soft_obs and isinstance(soft_obs, dict):
            # The caller may pass a real SoftObjectObserver observation or an error dict.
            soft_available = soft_obs.get("center") is not None or soft_obs.get("pose_source") is not None
            pose_source = soft_obs.get("pose_source")
            height_ratio = _ffloat(soft_obs.get("deformation_ratio_z"), None)
            width_ratio_x = _ffloat(soft_obs.get("width_ratio_x"), None)
            width_ratio_y = _ffloat(soft_obs.get("width_ratio_y"), None)
            table_gap = _ffloat(soft_obs.get("table_gap_m"), None)
            min_shape_ratio = _safe_min([height_ratio, width_ratio_x, width_ratio_y])
            max_width_ratio = _safe_max([width_ratio_x, width_ratio_y])
            if min_shape_ratio is not None:
                compression_ratio = max(0.0, 1.0 - min_shape_ratio)

        warnings = []
        unsafe_reasons = []

        if effort is not None:
            if max_effort is not None and effort >= max_effort:
                unsafe_reasons.append(f"effort {effort:.4f} >= max {max_effort:.4f}")
            elif warn_effort is not None and effort >= warn_effort:
                warnings.append(f"effort {effort:.4f} >= warn {warn_effort:.4f}")

        if compression_ratio is not None:
            if compression_ratio >= max_compression:
                unsafe_reasons.append(
                    f"compression_ratio {compression_ratio:.4f} >= max {max_compression:.4f}"
                )
            elif compression_ratio >= warn_compression:
                warnings.append(
                    f"compression_ratio {compression_ratio:.4f} >= warn {warn_compression:.4f}"
                )

        if height_ratio is not None and height_ratio <= min_height_ratio:
            unsafe_reasons.append(
                f"height_ratio {height_ratio:.4f} <= min {min_height_ratio:.4f}"
            )

        if max_width_ratio is not None and max_width_ratio >= max_lateral_expansion:
            warnings.append(
                f"lateral_expansion_ratio {max_width_ratio:.4f} >= warn {max_lateral_expansion:.4f}"
            )

        # table_gap < 0 means object bbox bottom is below table top.  Some tiny
        # penetration is expected in PhysX.  Too much is a physical plausibility issue.
        if table_gap is not None and table_gap < -abs(table_penetration_limit):
            warnings.append(
                f"table penetration {table_gap:.4f} m exceeds {table_penetration_limit:.4f} m"
            )

        unsafe = len(unsafe_reasons) > 0
        warn = (not unsafe) and len(warnings) > 0

        if unsafe:
            recommended_action = "relax_open_deformation" if any("compression" in r or "height" in r for r in unsafe_reasons) else "relax_open_effort"
            safety_state = "unsafe"
            safe_to_continue = False
        elif warn:
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
            "safe_to_continue": safe_to_continue,
            "unsafe": unsafe,
            "warning": warn,
            "safety_state": safety_state,
            "recommended_action": recommended_action,
            "reasons": unsafe_reasons,
            "warnings": warnings,
            "effort_sim": effort,
            "contact_like_effort": contact_like,
            "pose_source": pose_source,
            "soft_observation_available": soft_available,
            "height_ratio": height_ratio,
            "width_ratio_x": width_ratio_x,
            "width_ratio_y": width_ratio_y,
            "min_shape_ratio": min_shape_ratio,
            "max_width_ratio": max_width_ratio,
            "compression_ratio_est": compression_ratio,
            "table_gap_m": table_gap,
            "thresholds": {
                "max_effort_sim": max_effort,
                "warn_effort_sim": warn_effort,
                "max_compression_ratio": max_compression,
                "warn_compression_ratio": warn_compression,
                "min_height_ratio": min_height_ratio,
                "max_lateral_expansion_ratio": max_lateral_expansion,
                "max_table_penetration_m": table_penetration_limit,
            },
            "context": context,
            "interpretation": (
                "monitor-first deformation/effort safety decision; requests gentle relaxation "
                "when effort or soft-object compression becomes unsafe"
            ),
        }
