"""Soft/deformable object observation utilities

Rigid-object code often treats an object's USD prim transform as the object
pose.  That is not reliable for deformable USD assets: the asset root can be a
bottom frame, while the grasp-relevant body is represented by visible/collision
meshes that may move/deform.  This observer estimates the manipulation-relevant
state from live USD bounding boxes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

import omni.usd
from pxr import Usd, UsdGeom, Sdf


Vec3 = List[float]


class SoftObjectObserver:
    """Observe soft/deformable objects using live bbox geometry.

    The default pose returned by :meth:`observe` is the visible/render mesh bbox
    centre, not the wrapper/root transform.  For the foam USD, this means using
    ``CubeSoft_10`` rather than ``foam_cube_green_0`` root.
    """

    def __init__(self, config: dict):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_soft_target(self, target: Any) -> bool:
        """Return True when target metadata indicates soft/deformable logic."""
        if not isinstance(target, dict):
            return False
        asset_type = str(target.get("asset_type", "")).lower()
        compliance = str(target.get("compliance", "")).lower()
        material = str(target.get("material", target.get("material_name", ""))).lower()
        return bool(
            target.get("deformable")
            or target.get("soft_object")
            or asset_type in ("deformable_usd", "usd_reference")
            or compliance in ("soft", "deformable", "compliant")
            or "foam" in material
        )

    def observe(self, target_or_path: Any, *, stage=None, table_top_z: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Return a live soft-object observation dictionary.

        Parameters
        ----------
        target_or_path:
            Either a target dictionary containing ``prim_path`` and metadata, or
            a prim path string.
        table_top_z:
            Optional known table top height.  When omitted, the observer tries
            to measure ``/World/Trial/Table/Top``.
        """
        stage = stage or omni.usd.get_context().get_stage()
        if stage is None:
            return None

        target = target_or_path if isinstance(target_or_path, dict) else {}
        prim_path = target.get("prim_path") if isinstance(target, dict) else str(target_or_path)
        if not prim_path:
            return None

        wrapper_prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
        if not wrapper_prim or not wrapper_prim.IsValid():
            return None

        # Candidate render/visible prim is the best pose source for grasping.
        visible_path = self._find_visible_mesh_path(stage, prim_path)
        collision_path = self._find_child_path_by_name(stage, prim_path, ("collision_mesh",))
        simulation_path = self._find_child_path_by_name(stage, prim_path, ("simulation_mesh",))

        wrapper_bbox = self._bbox(stage, prim_path)
        visible_bbox = self._bbox(stage, visible_path) if visible_path else None
        collision_bbox = self._bbox(stage, collision_path) if collision_path else None

        # For grasp planning/validation, prefer visible geometry.  It is the
        # grasp-relevant object in the viewport and stays stable for our foam USD.
        pose_bbox = visible_bbox or collision_bbox or wrapper_bbox
        if pose_bbox is None:
            return None

        if table_top_z is None:
            table_top_z = self._table_top_z(stage)

        nominal_height = self._nominal_height_m(target)
        nominal_width = self._nominal_width_m(target)
        height = pose_bbox["size"][2]
        width_x = pose_bbox["size"][0]
        width_y = pose_bbox["size"][1]

        table_gap = None
        bottom_clearance = None
        if table_top_z is not None:
            table_gap = pose_bbox["min"][2] - float(table_top_z)
            bottom_clearance = table_gap

        deformation_ratio_z = None
        if nominal_height and nominal_height > 1e-9:
            deformation_ratio_z = height / nominal_height

        width_ratio_x = None
        width_ratio_y = None
        if nominal_width and nominal_width > 1e-9:
            width_ratio_x = width_x / nominal_width
            width_ratio_y = width_y / nominal_width

        obs = {
            "observer": "SoftObjectObserver",
            "prim_path": prim_path,
            "pose_source": "visible_bbox" if visible_bbox else ("collision_bbox" if collision_bbox else "wrapper_bbox"),
            "visible_mesh_path": visible_path,
            "collision_mesh_path": collision_path,
            "simulation_mesh_path": simulation_path,
            "center": pose_bbox["center"],
            "bottom_z": pose_bbox["min"][2],
            "top_z": pose_bbox["max"][2],
            "height_m": height,
            "width_x_m": width_x,
            "width_y_m": width_y,
            "table_top_z": table_top_z,
            "table_gap_m": table_gap,
            "bottom_clearance_m": bottom_clearance,
            "nominal_height_m": nominal_height,
            "nominal_width_m": nominal_width,
            "deformation_ratio_z": deformation_ratio_z,
            "width_ratio_x": width_ratio_x,
            "width_ratio_y": width_ratio_y,
            "visible_bbox": visible_bbox,
            "collision_bbox": collision_bbox,
            "wrapper_bbox": wrapper_bbox,
        }

        # Useful warning flags for logs/research analysis.
        warnings: List[str] = []
        if visible_bbox and collision_bbox:
            z_offset = collision_bbox["min"][2] - visible_bbox["min"][2]
            obs["collision_visible_bottom_offset_m"] = z_offset
            if abs(z_offset) > float(self.config.get("max_soft_visible_collision_bottom_offset_m", 0.004)):
                warnings.append(f"visible/collision bottom offset {z_offset:.4f} m")
        if table_gap is not None and table_gap > float(self.config.get("max_soft_table_gap_m", 0.003)):
            warnings.append(f"object above table by {table_gap:.4f} m")
        obs["warnings"] = warnings
        return obs

    def observe_position(self, target_or_path: Any, **kwargs) -> Optional[Vec3]:
        obs = self.observe(target_or_path, **kwargs)
        if not obs:
            return None
        return list(obs["center"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bbox(self, stage, path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        prim = stage.GetPrimAtPath(Sdf.Path(path))
        if not prim or not prim.IsValid():
            return None
        try:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=False,
            )
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn = box.GetMin()
            mx = box.GetMax()
            vals = [float(v) for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])]
            if any((not math.isfinite(v)) for v in vals):
                return None
            if any(abs(v) > 1.0e6 for v in vals):
                return None
            min_v = [float(mn[0]), float(mn[1]), float(mn[2])]
            max_v = [float(mx[0]), float(mx[1]), float(mx[2])]
            size = [max_v[i] - min_v[i] for i in range(3)]
            center = [(min_v[i] + max_v[i]) / 2.0 for i in range(3)]
            return {"path": path, "min": min_v, "max": max_v, "size": size, "center": center}
        except Exception:
            return None

    def _find_visible_mesh_path(self, stage, root_path: str) -> Optional[str]:
        root = stage.GetPrimAtPath(Sdf.Path(root_path))
        if not root or not root.IsValid():
            return None

        candidates: List[str] = []
        for prim in Usd.PrimRange(root):
            if prim == root:
                continue
            path = str(prim.GetPath())
            name = prim.GetName().lower()
            if prim.GetTypeName() == "Mesh":
                # Exclude generated/non-render physics helper meshes by name.
                if any(bad in name for bad in ("collision", "simulation", "tet")):
                    continue
                candidates.append(path)

        # Prefer the explicitly named foam cube mesh when present.
        for p in candidates:
            if "cubesoft" in p.lower() or "render" in p.lower() or "visual" in p.lower():
                return p
        return candidates[0] if candidates else None

    def _find_child_path_by_name(self, stage, root_path: str, names: Iterable[str]) -> Optional[str]:
        names = tuple(n.lower() for n in names)
        root = stage.GetPrimAtPath(Sdf.Path(root_path))
        if not root or not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if prim == root:
                continue
            if prim.GetName().lower() in names:
                return str(prim.GetPath())
        return None

    def _table_top_z(self, stage) -> Optional[float]:
        table_path = str(self.config.get("soft_observer_table_top_path", "/World/Trial/Table/Top"))
        info = self._bbox(stage, table_path)
        if info:
            return float(info["max"][2])
        if self.config.get("table_height") is not None:
            try:
                return float(self.config.get("table_height"))
            except Exception:
                pass
        return None

    def _nominal_height_m(self, target: dict) -> Optional[float]:
        if not isinstance(target, dict):
            return None
        for key in ("height", "height_m"):
            if target.get(key) is not None:
                try:
                    return float(target[key])
                except Exception:
                    pass
        if target.get("height_mm") is not None:
            try:
                return float(target["height_mm"]) / 1000.0
            except Exception:
                pass
        if target.get("size_mm") is not None:
            try:
                return float(target["size_mm"]) / 1000.0
            except Exception:
                pass
        return None

    def _nominal_width_m(self, target: dict) -> Optional[float]:
        if not isinstance(target, dict):
            return None
        if target.get("grip_dim_mm") is not None:
            try:
                return float(target["grip_dim_mm"]) / 1000.0
            except Exception:
                pass
        for key in ("width", "size"):
            if target.get(key) is not None:
                try:
                    return float(target[key])
                except Exception:
                    pass
        return None
