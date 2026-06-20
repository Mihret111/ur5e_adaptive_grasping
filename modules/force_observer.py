"""force_observer.py

force-feedback observer for the soft-grasp pipeline

Current validated backend:
  - Measured joint effort from the UR5e articulation that also contains the
    2FG7 prismatic finger joints.
  - NOT direct fingertip ContactSensor force.
  - Measured articulation/joint effort. For the 2FG7 prismatic finger
    joints it is force-like and useful as a simulation-side squeeze-feedback
    signal
  - Logged as "measured_joint_effort_sim" until a Newton calibration is performed

  - ForceObserver = proprioceptive/tactile perceptual schema.
  - The controller/action-selection layer will consume this interface rather
    than directly reading Isaac APIs.
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
            self.config.get("force_observer_target_effort_sim", 0.45)
        )
        self.max_effort_sim = float(
            self.config.get("force_observer_max_effort_sim", 1.20)
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

    def _try_create_articulation(self):
        errors = []
        for module_name, class_name in [
            ("isaacsim.core.prims", "Articulation"),
            ("omni.isaac.core.articulations", "Articulation"),
        ]:
            try:
                mod = __import__(module_name, fromlist=[class_name])
                Cls = getattr(mod, class_name)
                art = Cls(self.root_path)

                for meth in ("initialize", "post_reset"):
                    if hasattr(art, meth):
                        try:
                            getattr(art, meth)()
                        except Exception:
                            # Some Isaac versions require Play/World context.
                            # Do not fail immediately; effort read will decide.
                            pass

                # Test that at least an effort-like call exists.
                if not hasattr(art, "get_measured_joint_efforts"):
                    errors.append(f"{module_name}.{class_name}: missing get_measured_joint_efforts")
                    continue

                self._art = art
                self._art_module = f"{module_name}.{class_name}"
                return True
            except Exception as e:
                errors.append(f"{module_name}.{class_name}: {repr(e)}")

        self._init_error = " | ".join(errors)
        self._init_trace = traceback.format_exc()
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

        # Best case: exact length and both finger names present.
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

    def _ensure_ready(self) -> bool:
        if not self.enabled:
            return False
        if self._art is not None and self._left_idx is not None and self._right_idx is not None:
            return True

        self._attempted_init = True
        if self._art is None:
            if not self._try_create_articulation():
                return False

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
        if err or not flat:
            obs = {
                "available": False,
                "stage": stage_name,
                "force_source": self.BACKEND_NAME,
                "reason": err or "empty_effort_array",
                "root_path": self.root_path,
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
        near_target = abs(grip_effort - self.target_effort_sim) <= float(
            self.config.get("force_observer_target_band_sim", 0.10)
        )

        obs = {
            "available": True,
            "stage": stage_name,
            "force_source": self.BACKEND_NAME,
            "units": "sim_prismatic_joint_effort_not_calibrated_newtons",
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
            "max_effort_sim": self.max_effort_sim,
            "near_target_effort": near_target,
            "over_max_effort": over_max,
            "direct_contact_sensor_force_available": False,
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
            "max_effort_sim": self.max_effort_sim,
            "contact_threshold": self.contact_threshold,
            "last_observation": self._last_observation,
        }
