"""
manipulation_primitive_executor.py
──────────────────────────────────
manipulation primitives 

Design goal
-----------
This module adds *separate* soft-contact primitives without touching the
currently working pick-place benchmark.  It reuses the same perceptual and
motor schemas that the pick-and-place executor already owns:

  SoftObjectObserver      → object pose / table contact state
  ForceObserver           → finger effort proxy
  UR5EController          → Cartesian waypoint IK + smooth joint motion
  Gripper2FG7             → open / gentle hold / contact posture

The primitives are profile/config driven.  They do not hard-code a particular
object label; a config sequence chooses which target label to use for a demo,
and object metadata/affordances describe why that object is suitable.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import omni.kit.app
import omni.usd
from pxr import UsdGeom, Sdf, Gf

from modules.trial_diagnostics import json_safe


class ManipulationPrimitiveExecutor:
    """Run soft-push, soft-slide, and gentle-pull primitives.

    The class is deliberately thin: it uses the already-initialized controllers
    inside PickAndPlaceExecutor so the gripper/arm calibration remains identical
    to the pick-place pipeline.
    """

    SCHEMA_VERSION = "phase8_manipulation_primitives_v1"

    def __init__(self, config: dict, pick_executor: Any):
        self.config = config or {}
        self.pick_executor = pick_executor
        self.arm = pick_executor.arm
        self.gripper = pick_executor.gripper
        self.soft_observer = pick_executor.soft_observer
        self.force_observer = pick_executor.force_observer
        self.stage = omni.usd.get_context().get_stage()
        self._phase8_marker_counter = 0

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _safe_path_token(self, value: Any) -> str:
        token = "".join(c if str(c).isalnum() or c in ("_", "-") else "_" for c in str(value))
        return token.strip("_")[:64] or "item"

    def _phase8_marker_root(self) -> str:
        return (
            self.config
            .get("paths", {})
            .get("scene_spawn", {})
            .get("root", "/World/Trial")
        ) + "/Phase8PrimitiveMarkers"

    def _intended_destination_center(self, before: Optional[dict], direction_xy: Sequence[float], distance_m: float, table_h: float) -> Optional[List[float]]:
        if not before or not before.get("center"):
            return None
        center = before["center"]
        d = self._normalise_xy(direction_xy)
        z_offset = float(self.config.get("phase8_destination_marker_z_offset_m", 0.0020))
        return [
            float(center[0]) + d[0] * float(distance_m),
            float(center[1]) + d[1] * float(distance_m),
            float(table_h) + z_offset,
        ]

    def _add_phase8_destination_marker(
        self,
        *,
        primitive: str,
        target_label: str,
        destination_center: Optional[Sequence[float]],
        direction_xy: Sequence[float],
        distance_m: float,
    ) -> Optional[dict]:
        """Create a visual marker at the intended primitive destination.

        The marker is a visual-only USD cube, like tape on the table.  It does
        not add collision or rigid-body physics, so it cannot influence the soft
        object result.
        """
        if not bool(self.config.get("phase8_destination_markers_enabled", True)):
            return None
        if destination_center is None:
            return None

        root = self._phase8_marker_root()
        if not self.stage.GetPrimAtPath(Sdf.Path(root)).IsValid():
            UsdGeom.Xform.Define(self.stage, Sdf.Path(root))

        self._phase8_marker_counter += 1
        path = (
            f"{root}/"
            f"{self._safe_path_token(primitive)}_"
            f"{self._safe_path_token(target_label)}_"
            f"{self._phase8_marker_counter:02d}_IntendedDestination"
        )

        size = float(self.config.get("phase8_destination_marker_size_m", 0.045))
        thickness = float(self.config.get("phase8_destination_marker_thickness_m", 0.0012))
        color = self.config.get("phase8_destination_marker_color", [1.0, 0.65, 0.05])
        try:
            color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            color_vec = Gf.Vec3f(1.0, 0.65, 0.05)

        cube = UsdGeom.Cube.Define(self.stage, Sdf.Path(path))
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(float(destination_center[0]), float(destination_center[1]), float(destination_center[2])))
        cube.AddScaleOp().Set(Gf.Vec3f(size, size, thickness))
        cube.CreateDisplayColorAttr([color_vec])

        info = {
            "enabled": True,
            "marker_path": path,
            "world_center": [float(destination_center[0]), float(destination_center[1]), float(destination_center[2])],
            "size_xy_m": [size, size],
            "thickness_m": thickness,
            "direction_xy_m": [float(direction_xy[0]), float(direction_xy[1])],
            "intended_distance_m": float(distance_m),
            "note": "Visual-only marker for the intended final object center of the primitive.",
        }
        print(
            f"[PrimitiveExecutor] destination marker: {path} "
            f"at ({destination_center[0]:.3f}, {destination_center[1]:.3f}, {destination_center[2]:.4f})"
        )
        return info

    def _table_height(self, scene_info: dict) -> float:
        return float(scene_info.get("table_height", self.config.get("table_height", 0.750)))

    def _target_label(self, target: dict) -> str:
        return str((target or {}).get("label", "unknown_target"))

    def _shape(self, target: dict) -> str:
        return str((target or {}).get("shape", "unknown"))

    def _normalise_xy(self, v: Sequence[float], fallback: Sequence[float] = (1.0, 0.0)) -> List[float]:
        try:
            x = float(v[0])
            y = float(v[1])
        except Exception:
            x, y = float(fallback[0]), float(fallback[1])
        n = math.sqrt(x * x + y * y)
        if n < 1e-9:
            return [float(fallback[0]), float(fallback[1])]
        return [x / n, y / n]

    def _target_by_label(self, scene_info: dict, label: str) -> Optional[dict]:
        for target in scene_info.get("all_objects", []) or []:
            if str(target.get("label")) == str(label):
                return target
        return None

    def _select_target(self, scene_info: dict, spec: dict) -> Optional[dict]:
        label = spec.get("target_label") or spec.get("label")
        if label:
            target = self._target_by_label(scene_info, str(label))
            if target:
                return target
            print(f"[PrimitiveExecutor] ⚠ target_label not found: {label}")
            return None

        # Fallback: select the first object that declares this primitive in its
        # affordance list.  This keeps the mechanism property/profile based.
        primitive = str(spec.get("primitive", "")).lower()
        for target in scene_info.get("all_objects", []) or []:
            affordances = target.get("primitive_affordances") or target.get("affordances") or []
            if isinstance(affordances, str):
                affordances = [affordances]
            if primitive in [str(a).lower() for a in affordances]:
                return target

        # Last fallback: use scene pick target.
        return scene_info.get("pick_target")

    async def _step_seconds(self, seconds: float):
        app = omni.kit.app.get_app()
        frames = max(1, int(float(seconds) * 60.0))
        for _ in range(frames):
            try:
                self.gripper.update()
            except Exception:
                pass
            await app.next_update_async()

    def _observe(self, target: dict, stage_name: str, table_height: float) -> Optional[dict]:
        obs = None
        try:
            obs = self.soft_observer.observe(target, table_top_z=table_height)
        except Exception as e:
            print(f"[PrimitiveExecutor] ⚠ Soft observation failed at {stage_name}: {e}")
            obs = None
        if obs:
            c = obs.get("center") or [None, None, None]
            print(
                f"[PrimitiveExecutor:{stage_name}] "
                f"source={obs.get('pose_source')} center=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f}) "
                f"bottom={obs.get('bottom_z'):.4f} top={obs.get('top_z'):.4f}"
            )
        return obs

    def _force(self, stage_name: str) -> dict:
        try:
            return self.force_observer.observe(stage_name=stage_name)
        except Exception as e:
            return {"available": False, "stage": stage_name, "reason": str(e)}

    def _object_radius_for_contact(self, target: dict, obs: Optional[dict]) -> float:
        widths = []
        if obs:
            for k in ("width_x_m", "width_y_m"):
                try:
                    widths.append(float(obs.get(k)))
                except Exception:
                    pass
        for key in ("grip_dim_mm", "diameter_mm", "width_mm", "size_mm"):
            try:
                val = target.get(key)
                if val is not None:
                    widths.append(float(val) / 1000.0)
            except Exception:
                pass
        if not widths:
            return float(self.config.get("phase8_default_object_radius_m", 0.025))
        return 0.5 * max(widths)

    def _tool_z_for_table_contact(self, target: dict, obs: Optional[dict], table_height: float, fraction: float) -> float:
        # Put the gripper's nominal grasp/contact centre at a chosen fraction of
        # the object height.  This mirrors the pick planner but remains generic.
        h = None
        if obs and obs.get("height_m") is not None:
            h = float(obs.get("height_m"))
        if h is None:
            h = float(target.get("height_mm", target.get("grip_dim_mm", 40.0))) / 1000.0
        contact_center_z = table_height + max(0.0, min(1.0, float(fraction))) * h
        flange_to_grasp_centre = float(getattr(self.arm, "_flange_to_grasp_centre", 0.095))
        tool_z = contact_center_z + flange_to_grasp_centre

        # Never let the fingertips scrape deeply into the table.  Phase-8
        # primitives are surface-contact demonstrations, not table collision tests.
        min_tip_clearance = float(self.config.get("phase8_min_fingertip_clearance_m", 0.0025))
        flange_to_fingertips = float(getattr(self.arm, "_flange_to_fingertips", 0.117))
        min_tool_z = table_height + flange_to_fingertips + min_tip_clearance
        return max(tool_z, min_tool_z)

    async def _move_world(self, pos: Sequence[float], *, duration: float, steps: int, check_table_collision: bool = True) -> bool:
        seed = None
        try:
            seed = self.arm.get_joint_targets_deg()
        except Exception:
            seed = None
        joints = self.arm._solve_ik_for_world_pos(pos, seed_deg=seed)
        if joints is None:
            print(f"[PrimitiveExecutor] ❌ IK failed for world pos {pos}")
            return False
        return await self.arm.move_to(
            joints,
            duration=float(duration),
            steps=int(steps),
            check_table_collision=bool(check_table_collision),
        )

    async def _open_gripper(self, seconds: Optional[float] = None):
        self.gripper.open()
        await self._step_seconds(seconds if seconds is not None else self.config.get("phase8_open_settle_seconds", 0.35))

    async def _closed_pusher_posture(self, force_n: float):
        # Use the gripper as a compact soft-contact paddle. expected_grip_dim=0
        # makes the expected-contact floor equal to the fully closed end.
        self.gripper.close(
            force_n=float(force_n),
            expected_grip_dim_m=0.0,
            hold_extra_close_m=0.0,
            use_expected_contact_floor=False,
        )
        await self._step_seconds(self.config.get("phase8_pusher_close_seconds", 0.55))

    async def _gentle_object_close(self, target: dict, force_n: float, extra_close_m: float):
        grip_dim = target.get("close_expected_grip_dim_mm", target.get("grip_dim_mm"))
        expected = None
        if grip_dim is not None:
            expected = float(grip_dim) / 1000.0
        self.gripper.close(
            force_n=float(force_n),
            expected_grip_dim_m=expected,
            hold_extra_close_m=float(extra_close_m),
            use_expected_contact_floor=bool(target.get("gripper_use_expected_contact_floor", False)),
        )
        await self._step_seconds(self.config.get("phase8_grasp_close_seconds", 0.75))

    def _make_result(self, primitive: str, target: dict, spec: dict) -> dict:
        return {
            "primitive": primitive,
            "target_label": self._target_label(target),
            "target_shape": self._shape(target),
            "spec": dict(spec),
            "success": False,
            "reason": None,
            "events": [],
        }

    def _displacement_metrics(self, before: Optional[dict], after: Optional[dict], direction_xy: Sequence[float]) -> dict:
        if not before or not after:
            return {"available": False}
        b = before.get("center") or [0.0, 0.0, 0.0]
        a = after.get("center") or [0.0, 0.0, 0.0]
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        dxy = math.sqrt(dx * dx + dy * dy)
        direction = self._normalise_xy(direction_xy)
        along = dx * direction[0] + dy * direction[1]
        lateral = math.sqrt(max(0.0, dxy * dxy - along * along))
        return {
            "available": True,
            "delta_xyz_m": [dx, dy, dz],
            "delta_xy_m": dxy,
            "along_command_m": along,
            "lateral_error_m": lateral,
            "before_center": b,
            "after_center": a,
        }

    # ------------------------------------------------------------------
    # Primitive implementations
    # ------------------------------------------------------------------
    async def soft_push(self, target: dict, spec: dict, scene_info: dict) -> dict:
        """Table-supported open-loop soft push with force/pose audit."""
        primitive = "soft_push"
        result = self._make_result(primitive, target, spec)
        table_h = self._table_height(scene_info)
        direction = self._normalise_xy(spec.get("direction_xy_m", spec.get("direction_xy", (0.0, 1.0))))
        distance = float(spec.get("distance_m", self.config.get("phase8_soft_push_distance_m", 0.045)))
        approach_margin = float(spec.get("approach_margin_m", self.config.get("phase8_push_approach_margin_m", 0.030)))
        force_n = float(spec.get("force_n", self.config.get("phase8_soft_push_force_n", 38.0)))
        height_fraction = float(spec.get("contact_height_fraction", self.config.get("phase8_push_contact_height_fraction", 0.55)))
        min_along = float(spec.get("min_success_displacement_m", self.config.get("phase8_push_min_success_displacement_m", 0.010)))

        before = self._observe(target, f"{primitive}_before", table_h)
        if not before:
            result["reason"] = "missing_before_observation"
            return result

        center = before["center"]
        radius = self._object_radius_for_contact(target, before)
        tool_z = self._tool_z_for_table_contact(target, before, table_h, height_fraction)
        start = [center[0] - direction[0] * (radius + approach_margin), center[1] - direction[1] * (radius + approach_margin), tool_z]
        end = [center[0] + direction[0] * (distance), center[1] + direction[1] * (distance), tool_z]
        pre = list(start)
        pre[2] += float(spec.get("pre_contact_above_m", self.config.get("phase8_pre_contact_above_m", 0.080)))

        result["events"].append({"event": "planned", "before": before, "start": start, "end": end, "radius_m": radius})

        await self._closed_pusher_posture(force_n)
        ok = await self._move_world(pre, duration=self.config.get("phase8_approach_duration", 1.15), steps=self.config.get("phase8_approach_steps", 69))
        ok = ok and await self._move_world(start, duration=self.config.get("phase8_contact_approach_duration", 0.85), steps=self.config.get("phase8_contact_approach_steps", 51))
        result["events"].append({"event": "contact_pose", "ok": ok, "force": self._force(f"{primitive}_contact")})
        ok = ok and await self._move_world(end, duration=spec.get("duration", self.config.get("phase8_soft_push_duration", 1.60)), steps=spec.get("steps", self.config.get("phase8_soft_push_steps", 96)))
        await self._step_seconds(self.config.get("phase8_post_contact_settle_seconds", 0.25))
        after = self._observe(target, f"{primitive}_after", table_h)
        metrics = self._displacement_metrics(before, after, direction)
        marker_primitive = str(spec.get("_phase8_marker_primitive_name", primitive))
        intended_destination = self._intended_destination_center(before, direction, distance, table_h)
        destination_marker = self._add_phase8_destination_marker(
            primitive=marker_primitive,
            target_label=self._target_label(target),
            destination_center=intended_destination,
            direction_xy=direction,
            distance_m=distance,
        )
        if intended_destination is not None:
            result["intended_destination_center"] = intended_destination
        if destination_marker is not None:
            result["destination_marker"] = destination_marker
        result["events"].append({"event": "after_motion", "ok": ok, "after": after, "metrics": metrics, "intended_destination_center": intended_destination, "destination_marker": destination_marker, "force": self._force(f"{primitive}_after")})
        await self._move_world(pre, duration=self.config.get("phase8_retreat_duration", 0.90), steps=self.config.get("phase8_retreat_steps", 54), check_table_collision=True)
        await self._open_gripper()

        result["success"] = bool(ok and metrics.get("available") and metrics.get("along_command_m", 0.0) >= min_along)
        result["reason"] = "push_displacement_passed" if result["success"] else "push_displacement_too_small_or_motion_failed"
        result["metrics"] = metrics
        return result

    async def soft_slide(self, target: dict, spec: dict, scene_info: dict) -> dict:
        """Maintain low table contact while sliding an object along the table."""
        # Same contact mechanics as push, but longer/slow and stricter table-contact audit.
        primitive = "soft_slide"
        slide_spec = dict(spec)
        slide_spec.setdefault("distance_m", self.config.get("phase8_soft_slide_distance_m", 0.060))
        slide_spec.setdefault("duration", self.config.get("phase8_soft_slide_duration", 2.10))
        slide_spec.setdefault("steps", self.config.get("phase8_soft_slide_steps", 126))
        slide_spec.setdefault("contact_height_fraction", self.config.get("phase8_slide_contact_height_fraction", 0.45))
        slide_spec.setdefault("min_success_displacement_m", self.config.get("phase8_slide_min_success_displacement_m", 0.018))
        slide_spec.setdefault("_phase8_marker_primitive_name", "soft_slide")
        result = await self.soft_push(target, slide_spec, scene_info)
        result["primitive"] = primitive
        if result.get("success"):
            after = None
            for ev in reversed(result.get("events", [])):
                after = ev.get("after") if isinstance(ev, dict) else None
                if after:
                    break
            table_gap = after.get("table_gap_m") if after else None
            max_gap = float(self.config.get("phase8_slide_max_table_gap_m", 0.012))
            if table_gap is not None and table_gap > max_gap:
                result["success"] = False
                result["reason"] = "slide_object_lifted_too_far_from_table"
            else:
                result["reason"] = "slide_displacement_passed"
        return result

    async def gentle_pull(self, target: dict, spec: dict, scene_info: dict) -> dict:
        """Gripper-assisted table drag/pull without lifting."""
        primitive = "gentle_pull"
        result = self._make_result(primitive, target, spec)
        table_h = self._table_height(scene_info)
        direction = self._normalise_xy(spec.get("direction_xy_m", spec.get("direction_xy", (-1.0, 0.0))))
        distance = float(spec.get("distance_m", self.config.get("phase8_gentle_pull_distance_m", 0.040)))
        force_n = float(spec.get("force_n", self.config.get("phase8_gentle_pull_force_n", 45.0)))
        extra_close = float(spec.get("hold_extra_m", self.config.get("phase8_gentle_pull_hold_extra_m", 0.0010)))
        height_fraction = float(spec.get("contact_height_fraction", self.config.get("phase8_pull_contact_height_fraction", 0.50)))
        min_along = float(spec.get("min_success_displacement_m", self.config.get("phase8_pull_min_success_displacement_m", 0.010)))

        before = self._observe(target, f"{primitive}_before", table_h)
        if not before:
            result["reason"] = "missing_before_observation"
            return result

        center = before["center"]
        tool_z = self._tool_z_for_table_contact(target, before, table_h, height_fraction)
        grasp = [center[0], center[1], tool_z]
        pre = list(grasp)
        pre[2] += float(spec.get("pre_contact_above_m", self.config.get("phase8_pre_contact_above_m", 0.080)))
        end = [center[0] + direction[0] * distance, center[1] + direction[1] * distance, tool_z]
        post = list(end)
        post[2] += float(spec.get("post_pull_retreat_up_m", self.config.get("phase8_pre_contact_above_m", 0.080)))

        result["events"].append({"event": "planned", "before": before, "grasp": grasp, "end": end})

        await self._open_gripper(seconds=self.config.get("phase8_open_settle_seconds", 0.35))
        ok = await self._move_world(pre, duration=self.config.get("phase8_approach_duration", 1.15), steps=self.config.get("phase8_approach_steps", 69))
        ok = ok and await self._move_world(grasp, duration=self.config.get("phase8_contact_approach_duration", 0.85), steps=self.config.get("phase8_contact_approach_steps", 51))
        if ok:
            await self._gentle_object_close(target, force_n, extra_close)
        result["events"].append({"event": "closed_for_pull", "ok": ok, "force": self._force(f"{primitive}_after_close")})
        ok = ok and await self._move_world(end, duration=spec.get("duration", self.config.get("phase8_gentle_pull_duration", 1.65)), steps=spec.get("steps", self.config.get("phase8_gentle_pull_steps", 99)))
        await self._step_seconds(self.config.get("phase8_post_contact_settle_seconds", 0.25))
        after = self._observe(target, f"{primitive}_after", table_h)
        metrics = self._displacement_metrics(before, after, direction)
        intended_destination = self._intended_destination_center(before, direction, distance, table_h)
        gentle_pull_marker = self._add_phase8_destination_marker(
            primitive=primitive,
            target_label=self._target_label(target),
            destination_center=intended_destination,
            direction_xy=direction,
            distance_m=distance,
        )
        if intended_destination is not None:
            result["intended_destination_center"] = intended_destination
        if gentle_pull_marker is not None:
            result["destination_marker"] = gentle_pull_marker
        result["events"].append({"event": "after_pull", "ok": ok, "after": after, "metrics": metrics, "intended_destination_center": intended_destination, "destination_marker": gentle_pull_marker, "force": self._force(f"{primitive}_after")})
        await self._open_gripper(seconds=self.config.get("phase8_open_settle_seconds", 0.35))
        await self._move_world(post, duration=self.config.get("phase8_retreat_duration", 0.90), steps=self.config.get("phase8_retreat_steps", 54), check_table_collision=True)

        result["success"] = bool(ok and metrics.get("available") and metrics.get("along_command_m", 0.0) >= min_along)
        result["reason"] = "pull_displacement_passed" if result["success"] else "pull_displacement_too_small_or_motion_failed"
        result["metrics"] = metrics
        return result

    # ------------------------------------------------------------------
    # Batch entry point
    # ------------------------------------------------------------------
    async def run_from_scene(self, scene_info: dict) -> dict:
        sequence = self.config.get("phase8_primitive_sequence", []) or []
        if not sequence:
            print("[PrimitiveExecutor] No phase8_primitive_sequence configured.")
            return {
                "schema_version": self.SCHEMA_VERSION,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "enabled": True,
                "results": [],
                "success_count": 0,
                "failure_count": 0,
                "reason": "empty_sequence",
            }

        log = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "enabled": True,
            "trial_index": scene_info.get("trial_index"),
            "results": [],
            "success_count": 0,
            "failure_count": 0,
            "notes": [
                "Phase 8 primitives are separate from the frozen Phase 7 pick-place benchmark.",
                "They are selected by config/object affordances, not by hard-coded Python label rules.",
            ],
        }

        dispatch = {
            "soft_push": self.soft_push,
            "push": self.soft_push,
            "soft_slide": self.soft_slide,
            "slide": self.soft_slide,
            "gentle_pull": self.gentle_pull,
            "pull": self.gentle_pull,
        }

        print("\n[PrimitiveExecutor] Phase 8 manipulation primitives enabled")
        for i, spec in enumerate(sequence):
            spec = dict(spec or {})
            primitive = str(spec.get("primitive", "")).lower()
            fn = dispatch.get(primitive)
            if fn is None:
                res = {
                    "primitive": primitive,
                    "success": False,
                    "reason": "unknown_primitive",
                    "spec": spec,
                }
                log["failure_count"] += 1
                log["results"].append(res)
                continue

            target = self._select_target(scene_info, spec)
            if target is None:
                res = {
                    "primitive": primitive,
                    "success": False,
                    "reason": "target_not_found",
                    "spec": spec,
                }
                log["failure_count"] += 1
                log["results"].append(res)
                continue

            print("\n" + "·" * 60)
            print(f"[PrimitiveExecutor] {i + 1}/{len(sequence)} {primitive} → {self._target_label(target)}")
            print("·" * 60)
            try:
                res = await fn(target, spec, scene_info)
            except Exception as e:
                res = {
                    "primitive": primitive,
                    "target_label": self._target_label(target),
                    "success": False,
                    "reason": f"exception: {e}",
                    "spec": spec,
                }
            if res.get("success"):
                log["success_count"] += 1
            else:
                log["failure_count"] += 1
            log["results"].append(json_safe(res))
            print(f"[PrimitiveExecutor] result: success={res.get('success')} reason={res.get('reason')}")

            await self._step_seconds(self.config.get("phase8_between_primitives_settle_seconds", 0.35))

        return log