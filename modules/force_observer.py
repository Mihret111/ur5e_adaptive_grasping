"""force_observer.py

Scientifically honest force-feedback observer for the B2B soft-grasp pipeline.

Current validated backend:
  - measured joint effort from the UR5e articulation that also contains the
    2FG7 prismatic finger joints.

  - This is NOT direct fingertip ContactSensor force.
  - This is measured articulation/joint effort. For the 2FG7 prismatic finger
    joints it is force-like and useful as a simulation-side squeeze-feedback
    signal,
  - It will be logged as ``measured_joint_effort_sim`` until a Newton calibration 
    is performed.

COGAR mapping:
  - ForceObserver = proprioceptive/tactile perceptual schema.
  - The controller/action-selection layer must consume this interface rather
    than directly reading Isaac APIs. This keeps sim and real robot backends
    replaceable.
"""

from __future__ import annotations

import math
import traceback
from typing import Any, Dict, List, Optional, Tuple


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _to_builtin(x: Any, max_items: int = 100) -> Any:
    """Convert numpy/torch/Isaac values into JSON-safe Python values."""
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return _to_builtin(x.tolist(), max_items=max_items)
    except Exception:
        pass

    if hasattr(x, "detach") and hasattr(x, "cpu"):
        try:
            return _to_builtin(x.detach().cpu().numpy(), max_items=max_items)
        except Exception:
            pass

    if isinstance(x, (list, tuple)):
        out = [_to_builtin(v, max_items=max_items) for v in list(x)[:max_items]]
        if len(x) > max_items:
            out.append(f"... truncated {len(x) - max_items} items")
        return out

    if isinstance(x, dict):
        return {str(k): _to_builtin(v, max_items=max_items) for k, v in x.items()}

    if isinstance(x, (str, int, bool)) or x is None:
        return x

    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)

    try:
        return float(x)
    except Exception:
        return str(x)


def _flat_float_list(x: Any) -> Tuple[List[float], List[int]]:
    """Return a flattened float list and the original array-like shape."""
    try:
        import numpy as np
        a = np.array(_to_builtin(x), dtype=float)
        return a.reshape(-1).tolist(), list(a.shape)
    except Exception:
        pass

    try:
        if isinstance(x, list):
            # Common Isaac case: [[...]]
            if len(x) == 1 and isinstance(x[0], list):
                return [float(v) for v in x[0]], [1, len(x[0])]
            return [float(v) for v in x], [len(x)]
    except Exception:
        pass

    return [], []


def _get_nested_attr(obj: Any, attr_path: str) -> Any:
    cur = obj
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


class ArticulationEffortForceObserver:
    """Read 2FG7 finger efforts from an Isaac articulation.

    This observer is intentionally conservative:
      - It validates that the configured left/right finger joint names appear
        in the articulation DOF/joint order.
      - It reports units as simulation joint-effort units, not calibrated Newtons.
      - It exposes ``contact_like`` only as an effort threshold, not a proof of
        contact by itself. The executor should still combine it with gripper
        state, contact geometry, and soft-object motion validation.
    """

    BACKEND_NAME = "measured_joint_effort_sim"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("force_observer_enabled", True))
        self.root_path = str(
            self.config.get(
                "force_observer_articulation_root",
                "/mir/base_link_cabinet/cabinet/ur_mount/ur5e_physics",
            )
        )
        roots_cfg = self.config.get("force_observer_articulation_roots", None)
        if isinstance(roots_cfg, (list, tuple)):
            self.root_candidates = [str(r) for r in roots_cfg if str(r).strip()]
        else:
            self.root_candidates = [self.root_path]
        # The 2FG7 may appear either inside the UR5e articulation or as its own
        # articulation depending on how the USD is loaded/composed.  Try both.
        for fallback in ["/onrobot_2fg7", "/World/onrobot_2fg7"]:
            if fallback not in self.root_candidates:
                self.root_candidates.append(fallback)
        self.left_joint_name = str(
            self.config.get("force_observer_left_finger_joint_name", "left_finger_joint")
        )
        self.right_joint_name = str(
            self.config.get("force_observer_right_finger_joint_name", "right_finger_joint")
        )
        self.contact_threshold = float(
            self.config.get("force_observer_contact_effort_threshold_sim", 0.05)
        )
        self.baseline_deadband = float(
            self.config.get("force_observer_baseline_abs_deadband", 1.0e-4)
        )
        self.target_effort_sim = float(
            self.config.get(
                "force_observer_target_effort_sim",
                self.config.get("adaptive_effort_target_sim", 0.35),
            )
        )
        self.max_effort_sim = float(
            self.config.get("force_observer_max_effort_sim", self.config.get("adaptive_effort_max_sim", 1.20))
        )
        self.target_band_sim = float(
            self.config.get(
                "force_observer_target_band_sim",
                self.config.get("adaptive_effort_target_band_sim", 0.08),
            )
        )
        self.audit_rich_joint_signals = bool(
            self.config.get("force_observer_audit_rich_joint_signals", True)
        )

        self._art = None
        self._art_module = None
        self._name_source = None
        self._dof_names: List[str] = []
        self._left_idx: Optional[int] = None
        self._right_idx: Optional[int] = None
        self._init_error: Optional[str] = None
        self._init_trace: Optional[str] = None
        self._attempted_init = False

        self._baseline_left = float(self.config.get("force_observer_baseline_left", 0.0))
        self._baseline_right = float(self.config.get("force_observer_baseline_right", 0.0))
        self._last_observation: Optional[dict] = None

        print(
            "[ForceObserver] configured: "
            f"enabled={self.enabled}, root={self.root_path}, "
            f"left={self.left_joint_name}, right={self.right_joint_name}, "
            f"target_effort_sim={self.target_effort_sim}, max_effort_sim={self.max_effort_sim}"
        )

    # ─────────────────────────────────────────────────────────────
    # Articulation creation and DOF mapping
    # ─────────────────────────────────────────────────────────────

    def _invalidate_articulation(self, reason: str = "unknown"):
        """Forget a bad Isaac articulation wrapper so the next observe can retry.

        In some Isaac/Kit sessions the first Articulation wrapper can be created
        before the full DOF metadata is populated, returning a scalar dummy
        effort with generated name ['dof_0'].  Treat that as not-ready, not as a
        valid force signal.
        """
        self._art = None
        self._art_module = None
        self._name_source = None
        self._dof_names = []
        self._left_idx = None
        self._right_idx = None
        self._init_error = reason

    def _try_create_articulation(self):
        errors = []
        roots = list(dict.fromkeys([self.root_path] + list(getattr(self, "root_candidates", []))))
        for root_path in roots:
            for module_name, class_name in [
                ("isaacsim.core.prims", "Articulation"),
                ("omni.isaac.core.articulations", "Articulation"),
            ]:
                try:
                    mod = __import__(module_name, fromlist=[class_name])
                    Cls = getattr(mod, class_name)
                    art = Cls(root_path)

                    for meth in ("initialize", "post_reset"):
                        if hasattr(art, meth):
                            try:
                                getattr(art, meth)()
                            except Exception:
                                pass

                    if not hasattr(art, "get_measured_joint_efforts"):
                        errors.append(f"{root_path} {module_name}.{class_name}: missing get_measured_joint_efforts")
                        continue

                    self._art = art
                    self._art_module = f"{module_name}.{class_name}"
                    self.root_path = root_path

                    flat, shape, err = self._read_efforts_flat()
                    if err or not flat or len(flat) < 2:
                        errors.append(
                            f"{root_path} {module_name}.{class_name}: unusable effort read err={err}, shape={shape}, len={len(flat)}"
                        )
                        self._art = None
                        continue

                    names, source = self._select_dof_names(len(flat))
                    if self.left_joint_name in names and self.right_joint_name in names:
                        self._dof_names = names
                        self._name_source = source
                        self._left_idx = names.index(self.left_joint_name)
                        self._right_idx = names.index(self.right_joint_name)
                        print(
                            "[ForceObserver] selected articulation: "
                            f"root={root_path}, module={self._art_module}, "
                            f"source={source}, left_idx={self._left_idx}, right_idx={self._right_idx}"
                        )
                        return True

                    errors.append(
                        f"{root_path} {module_name}.{class_name}: finger names missing in efforts; "
                        f"names={names}, source={source}, shape={shape}, len={len(flat)}"
                    )
                    self._art = None
                except Exception as e:
                    errors.append(f"{root_path} {module_name}.{class_name}: {repr(e)}")

        self._init_error = " | ".join(errors)
        self._init_trace = traceback.format_exc()
        self._invalidate_articulation(self._init_error)
        return False

    def _name_sources(self) -> Dict[str, List[str]]:
        if self._art is None:
            return {}
        out: Dict[str, List[str]] = {}
        for expr in [
            "dof_names",
            "joint_names",
            "_metadata.dof_names",
            "_metadata.joint_names",
            "_articulation_view.dof_names",
            "_articulation_view.joint_names",
        ]:
            try:
                val = _get_nested_attr(self._art, expr)
                if val is None:
                    continue
                names = list(val) if not isinstance(val, str) else [val]
                out[expr] = [str(n).split("/")[-1] for n in names]
            except Exception:
                continue
        return out

    def _select_dof_names(self, effort_len: int) -> Tuple[List[str], str]:
        sources = self._name_sources()
        wanted = {self.left_joint_name, self.right_joint_name}

        # Best: exact length and both finger names present.
        for source, names in sources.items():
            if len(names) == effort_len and wanted.issubset(set(names)):
                return names, source

        # Common Isaac case: joint_names includes fixed joints; efforts do not.
        for source, names in sources.items():
            movable = [n for n in names if "fixed" not in n.lower()]
            if len(movable) == effort_len and wanted.issubset(set(movable)):
                return movable, source + " minus fixed-like joints"

        # Fallback: exact length even if finger names are missing.
        for source, names in sources.items():
            if len(names) == effort_len:
                return names, source

        return [f"dof_{i}" for i in range(effort_len)], "generated_dof_names"

    def _read_efforts_flat(self) -> Tuple[List[float], List[int], Optional[str]]:
        if self._art is None:
            return [], [], "articulation_not_initialized"
        try:
            val = self._art.get_measured_joint_efforts()
            flat, shape = _flat_float_list(val)
            if not flat:
                return [], shape, "empty_effort_array"
            return flat, shape, None
        except Exception as e:
            return [], [], repr(e)

    def _safe_articulation_call(self, method_name: str):
        """Best-effort call for optional articulation force APIs.

        The exact Isaac wrapper differs by version. We do not use these optional
        signals for control yet; they are logged to explain the difference
        between measured projected efforts, measured 6D joint forces, and
        applied/commanded efforts.
        """
        if self._art is None or not hasattr(self._art, method_name):
            return None, f"method_missing:{method_name}"
        try:
            return getattr(self._art, method_name)(), None
        except Exception as e:
            return None, repr(e)

    def _summarize_optional_signal(self, method_name: str, max_items: int = 12) -> dict:
        val, err = self._safe_articulation_call(method_name)
        out = {
            "method": method_name,
            "available": err is None,
            "error": err,
        }
        if err is not None:
            return out

        safe = _to_builtin(val, max_items=200)
        flat, shape = _flat_float_list(val)
        out.update({
            "shape": shape,
            "flat_count": len(flat),
            "max_abs": max([abs(v) for v in flat], default=0.0),
            "sample": _to_builtin(safe, max_items=max_items),
        })

        # For measured_joint_forces the rows may be 6D spatial force/torque
        # vectors. The exact row-to-joint mapping can include base/fixed rows,
        # so we only provide candidate rows near the finger effort indices.
        if flat and shape and len(shape) >= 2 and shape[-1] in (6,):
            rows = _to_builtin(val, max_items=200)
            try:
                candidates = {}
                for label, idx in [("left_candidate_row", self._left_idx), ("right_candidate_row", self._right_idx)]:
                    if idx is None:
                        continue
                    for row_idx in [idx, idx + 1]:
                        if 0 <= row_idx < len(rows):
                            candidates[f"{label}_{row_idx}"] = rows[row_idx]
                out["candidate_spatial_rows_near_finger_indices"] = candidates
                out["spatial_force_note"] = (
                    "Rows are logged for audit only; row-to-finger mapping may include base/fixed-body offsets. "
                    "Do not use these as calibrated fingertip forces without a dedicated mapping/calibration test."
                )
            except Exception as e:
                out["candidate_rows_error"] = repr(e)

        return out

    def _ensure_ready(self) -> bool:
        if not self.enabled:
            return False
        if self._art is not None and self._left_idx is not None and self._right_idx is not None:
            return True

        self._attempted_init = True
        if self._art is None:
            if not self._try_create_articulation():
                return False
            if self._art is not None and self._left_idx is not None and self._right_idx is not None:
                return True

        flat, shape, err = self._read_efforts_flat()
        if err or not flat:
            self._init_error = f"could_not_read_efforts: {err}"
            return False

        names, source = self._select_dof_names(len(flat))
        self._dof_names = names
        self._name_source = source

        try:
            self._left_idx = names.index(self.left_joint_name)
            self._right_idx = names.index(self.right_joint_name)
        except ValueError:
            self._init_error = (
                "finger_joint_names_not_found_in_effort_order: "
                f"left={self.left_joint_name}, right={self.right_joint_name}, "
                f"names={names}, source={source}, effort_shape={shape}"
            )
            self._left_idx = None
            self._right_idx = None
            return False

        print(
            "[ForceObserver] ready: "
            f"module={self._art_module}, name_source={self._name_source}, "
            f"left_idx={self._left_idx}, right_idx={self._right_idx}"
        )
        return True

    # ─────────────────────────────────────────────────────────────
    # Public observer API
    # ─────────────────────────────────────────────────────────────

    def observe(self, stage_name: str = "unspecified") -> dict:
        if not self.enabled:
            obs = {
                "available": False,
                "stage": stage_name,
                "force_source": "disabled",
                "reason": "force_observer_disabled",
            }
            self._last_observation = obs
            return obs

        if not self._ensure_ready():
            obs = {
                "available": False,
                "stage": stage_name,
                "force_source": self.BACKEND_NAME,
                "reason": self._init_error or "not_ready",
                "root_path": self.root_path,
            }
            self._last_observation = obs
            return obs

        flat, shape, err = self._read_efforts_flat()
        bad_indices = (
            self._left_idx is None or self._right_idx is None or
            not flat or self._left_idx >= len(flat) or self._right_idx >= len(flat)
        )
        if err or bad_indices:
            # The Isaac articulation wrapper can occasionally become stale or
            # return a dummy scalar effort.  Reacquire once before giving up.
            old_reason = err or f"bad_effort_indices_or_empty: shape={shape}, len={len(flat)}"
            self._invalidate_articulation(old_reason)
            if self._try_create_articulation():
                flat, shape, err = self._read_efforts_flat()
                bad_indices = (
                    self._left_idx is None or self._right_idx is None or
                    not flat or self._left_idx >= len(flat) or self._right_idx >= len(flat)
                )
            if err or bad_indices:
                obs = {
                    "available": False,
                    "stage": stage_name,
                    "force_source": self.BACKEND_NAME,
                    "reason": err or old_reason or "empty_or_invalid_effort_array_after_reacquire",
                    "root_path": self.root_path,
                    "candidate_roots": list(getattr(self, "root_candidates", [])),
                    "effort_shape": shape,
                    "dof_names": list(self._dof_names),
                }
                self._last_observation = obs
                return obs

        left_raw = _finite_float(flat[self._left_idx], 0.0) if self._left_idx is not None else 0.0
        right_raw = _finite_float(flat[self._right_idx], 0.0) if self._right_idx is not None else 0.0

        left_net = float(left_raw - self._baseline_left)
        right_net = float(right_raw - self._baseline_right)

        # Suppress tiny numerical noise around the no-contact baseline.
        if abs(left_net) < self.baseline_deadband:
            left_net = 0.0
        if abs(right_net) < self.baseline_deadband:
            right_net = 0.0

        left_abs = abs(left_net)
        right_abs = abs(right_net)
        grip_effort = 0.5 * (left_abs + right_abs)
        effort_balance = abs(left_abs - right_abs)
        effort_sum = left_abs + right_abs

        # This is an effort-based releaser, not a proof of contact by itself.
        contact_like = grip_effort >= self.contact_threshold
        over_max = grip_effort >= self.max_effort_sim
        near_target = abs(grip_effort - self.target_effort_sim) <= self.target_band_sim

        optional_joint_signal_audit = None
        if self.audit_rich_joint_signals:
            optional_joint_signal_audit = {
                "measured_joint_efforts_used_for_control": {
                    "method": "get_measured_joint_efforts",
                    "left_index": self._left_idx,
                    "right_index": self._right_idx,
                    "left_raw": left_raw,
                    "right_raw": right_raw,
                    "note": "Projected active force/torque along each DOF; this is the control feedback signal for prismatic finger joints.",
                },
                "measured_joint_forces_spatial_audit": self._summarize_optional_signal("get_measured_joint_forces"),
                "applied_joint_efforts_command_audit": self._summarize_optional_signal("get_applied_joint_efforts"),
            }

        obs = {
            "available": True,
            "stage": stage_name,
            "force_source": self.BACKEND_NAME,
            "units": "measured_prismatic_joint_effort_N_not_fingertip_calibrated",
            "root_path": self.root_path,
            "articulation_class": self._art_module,
            "name_source": self._name_source,
            "effort_shape": shape,
            "left_joint_name": self.left_joint_name,
            "right_joint_name": self.right_joint_name,
            "left_index": self._left_idx,
            "right_index": self._right_idx,
            "left_effort_raw": left_raw,
            "right_effort_raw": right_raw,
            "left_effort_baseline": self._baseline_left,
            "right_effort_baseline": self._baseline_right,
            "left_effort_net_abs": left_abs,
            "right_effort_net_abs": right_abs,
            "grip_effort_sim": grip_effort,
            "effort_sum_sim": effort_sum,
            "effort_balance_sim": effort_balance,
            "contact_like_effort": contact_like,
            "target_effort_sim": self.target_effort_sim,
            "target_band_sim": self.target_band_sim,
            "max_effort_sim": self.max_effort_sim,
            "near_target_effort": near_target,
            "over_max_effort": over_max,
            "direct_contact_sensor_force_available": False,
            "optional_joint_signal_audit": optional_joint_signal_audit,
            "signal_taxonomy": {
                "used_for_control": "get_measured_joint_efforts / projected prismatic finger effort",
                "not_used_for_control_yet": "get_measured_joint_forces / 6D spatial joint wrench",
                "not_feedback": "get_applied_joint_efforts / commanded or applied effort",
                "not_available_for_soft_contact": "ContactSensor.force gave zero for deformable foam contact in our tests",
            },
            "interpretation": (
                "Measured articulation effort from 2FG7 prismatic finger DOFs; "
                "valid as sim-side force-like feedback after baseline and calibration checks."
            ),
        }
        self._last_observation = obs
        return obs

    def set_baseline_from_current(self, stage_name: str = "baseline") -> dict:
        """Capture current finger efforts as the no-contact baseline."""
        obs = self.observe(stage_name=stage_name)
        if obs.get("available"):
            self._baseline_left = float(obs.get("left_effort_raw", 0.0) or 0.0)
            self._baseline_right = float(obs.get("right_effort_raw", 0.0) or 0.0)
            obs = self.observe(stage_name=stage_name + "_after_set")
            obs["baseline_updated"] = True
        else:
            obs["baseline_updated"] = False
        return obs

    def get_diagnostics(self) -> dict:
        return {
            "enabled": self.enabled,
            "backend": self.BACKEND_NAME,
            "root_path": self.root_path,
            "articulation_class": self._art_module,
            "attempted_init": self._attempted_init,
            "ready": bool(self._art is not None and self._left_idx is not None and self._right_idx is not None),
            "init_error": self._init_error,
            "name_source": self._name_source,
            "dof_names": list(self._dof_names),
            "left_joint_name": self.left_joint_name,
            "right_joint_name": self.right_joint_name,
            "left_index": self._left_idx,
            "right_index": self._right_idx,
            "baseline_left": self._baseline_left,
            "baseline_right": self._baseline_right,
            "target_effort_sim": self.target_effort_sim,
            "target_band_sim": self.target_band_sim,
            "max_effort_sim": self.max_effort_sim,
            "contact_threshold": self.contact_threshold,
            "last_observation": self._last_observation,
        }