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

try:
    import numpy as np
except Exception:  # pragma: no cover - Isaac normally ships numpy
    np = None

import omni.usd
from pxr import Usd, UsdGeom, Sdf, Gf


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

        # IMPORTANT FOR DEFORMABLES:
        # UsdGeom.BBoxCache can remain static for deformable bodies, while the
        # runtime point state on simulation_mesh / collision_mesh / render mesh
        # actually moves.  Therefore we compute point-based world bboxes first
        # and only fall back to BBoxCache when points are unavailable.
        simulation_points_bbox = self._point_bbox(stage, simulation_path) if simulation_path else None
        collision_points_bbox = self._point_bbox(stage, collision_path) if collision_path else None
        visible_points_bbox = self._point_bbox(stage, visible_path) if visible_path else None

        visible_bbox = visible_points_bbox or (self._bbox(stage, visible_path) if visible_path else None)
        collision_bbox = collision_points_bbox or (self._bbox(stage, collision_path) if collision_path else None)
        simulation_bbox = simulation_points_bbox

        # Source priority for soft/deformable motion:
        #   1) simulation_mesh.points: actual deformable state cloud when available
        #   2) collision_mesh.points: contact geometry point state
        #   3) visible mesh points: render point state
        #   4) BBoxCache fallback: useful for static spawn checks only
        if simulation_points_bbox is not None:
            pose_bbox = simulation_points_bbox
            pose_source = "simulation_mesh_points"
        elif collision_points_bbox is not None:
            pose_bbox = collision_points_bbox
            pose_source = "collision_mesh_points"
        elif visible_points_bbox is not None:
            pose_bbox = visible_points_bbox
            pose_source = "visible_mesh_points"
        elif visible_bbox is not None:
            pose_bbox = visible_bbox
            pose_source = "visible_bbox"
        elif collision_bbox is not None:
            pose_bbox = collision_bbox
            pose_source = "collision_bbox"
        else:
            pose_bbox = wrapper_bbox
            pose_source = "wrapper_bbox"

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

        oriented_bbox = pose_bbox.get("oriented_bbox") if isinstance(pose_bbox, dict) else None
        oriented_size_m = None
        oriented_size_sorted_m = None
        oriented_max_ratio = None
        oriented_min_ratio = None
        oriented_ratio_sorted = None
        if isinstance(oriented_bbox, dict) and oriented_bbox.get("size") is not None:
            try:
                oriented_size_m = [float(v) for v in oriented_bbox.get("size", [])]
                oriented_size_sorted_m = [float(v) for v in oriented_bbox.get("size_sorted", sorted(oriented_size_m))]
                if nominal_width and nominal_width > 1e-9 and oriented_size_m:
                    ratios = [float(v) / nominal_width for v in oriented_size_m]
                    oriented_ratio_sorted = [float(v) / nominal_width for v in oriented_size_sorted_m]
                    oriented_max_ratio = max(ratios)
                    oriented_min_ratio = min(ratios)
            except Exception:
                oriented_size_m = None
                oriented_size_sorted_m = None
                oriented_max_ratio = None
                oriented_min_ratio = None
                oriented_ratio_sorted = None

        obs = {
            "observer": "SoftObjectObserver",
            "prim_path": prim_path,
            "pose_source": pose_source,
            "visible_mesh_path": visible_path,
            "collision_mesh_path": collision_path,
            "simulation_mesh_path": simulation_path,
            "simulation_bbox": simulation_bbox,
            "simulation_points_bbox": simulation_points_bbox,
            "collision_points_bbox": collision_points_bbox,
            "visible_points_bbox": visible_points_bbox,
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
            "oriented_bbox": oriented_bbox,
            "oriented_size_m": oriented_size_m,
            "oriented_size_sorted_m": oriented_size_sorted_m,
            "oriented_ratio_sorted": oriented_ratio_sorted,
            "oriented_max_ratio": oriented_max_ratio,
            "oriented_min_ratio": oriented_min_ratio,
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
    def _point_bbox(self, stage, path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Compute a world-space bbox directly from a prim's live points attr.

        This is the key observer for PhysX deformables.  For deformable bodies,
        the live state is often visible in the point arrays of simulation_mesh,
        collision_mesh, or render mesh even when UsdGeom.BBoxCache gives a stale
        authored/static bbox.
        """
        if not path:
            return None
        prim = stage.GetPrimAtPath(Sdf.Path(path))
        if not prim or not prim.IsValid():
            return None
        try:
            attr = prim.GetAttribute("points")
            if not attr:
                return None
            pts = attr.Get()
            if not pts:
                return None

            # Points are local to the prim. Convert them to world coordinates.
            try:
                xform = UsdGeom.Xformable(prim)
                M = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            except Exception:
                M = Gf.Matrix4d(1.0)

            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []

            for p in pts:
                wp = M.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                xs.append(float(wp[0]))
                ys.append(float(wp[1]))
                zs.append(float(wp[2]))

            if not xs:
                return None

            min_v = [min(xs), min(ys), min(zs)]
            max_v = [max(xs), max(ys), max(zs)]
            vals = min_v + max_v
            if any((not math.isfinite(v)) for v in vals):
                return None
            if any(abs(v) > 1.0e6 for v in vals):
                return None

            size = [max_v[i] - min_v[i] for i in range(3)]
            if any(v < 0.0 for v in size):
                return None
            center = [(min_v[i] + max_v[i]) / 2.0 for i in range(3)]

            # Also store centroid; bbox center is better for table/top/bottom tests,
            # centroid is useful later for deformation statistics.
            centroid = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]

            # Rotation-aware geometry audit.
            # A world-axis aligned bbox grows when a cube rotates, even if the
            # material has not stretched.  A PCA-oriented bbox is not perfect for
            # a deformable body, but it separates "rotated cube" from "actually
            # expanded point cloud" much better than AABB alone.  We keep this as
            # an audit signal and let higher-level validation decide whether to
            # use it as a pass/fail metric.
            oriented_bbox = None
            try:
                if np is not None and len(xs) >= 4:
                    P = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)])
                    C = np.mean(P, axis=0)
                    Pc = P - C
                    cov = np.cov(Pc, rowvar=False)
                    vals_e, vecs = np.linalg.eigh(cov)
                    order = np.argsort(vals_e)[::-1]
                    vals_e = vals_e[order]
                    vecs = vecs[:, order]
                    # Normalize/sign-stabilize axes for readable logs.
                    for j in range(3):
                        axis = vecs[:, j]
                        n = np.linalg.norm(axis)
                        if n > 1.0e-12:
                            axis = axis / n
                        # make dominant component positive for less log jitter
                        k = int(np.argmax(np.abs(axis)))
                        if axis[k] < 0.0:
                            axis = -axis
                        vecs[:, j] = axis
                    Q = Pc @ vecs
                    qmin = np.min(Q, axis=0)
                    qmax = np.max(Q, axis=0)
                    obb_size = qmax - qmin
                    obb_center_local = (qmin + qmax) / 2.0
                    obb_center = C + vecs @ obb_center_local
                    oriented_bbox = {
                        "source": "pca_oriented_points",
                        "center": [float(v) for v in obb_center],
                        "axes_world": [[float(vecs[i, j]) for i in range(3)] for j in range(3)],
                        "eigenvalues": [float(v) for v in vals_e],
                        "size": [float(v) for v in obb_size],
                        "size_sorted": [float(v) for v in sorted(obb_size.tolist())],
                        "min_local": [float(v) for v in qmin],
                        "max_local": [float(v) for v in qmax],
                        "note": "PCA oriented bbox reduces false deformation caused by rigid rotation of the object in world axes.",
                    }
            except Exception as exc:
                oriented_bbox = {"available": False, "reason": f"pca_oriented_bbox_failed: {exc}"}

            return {
                "path": path,
                "source": "points",
                "prim_type": prim.GetTypeName(),
                "num_points": len(xs),
                "min": min_v,
                "max": max_v,
                "size": size,
                "center": center,
                "centroid": centroid,
                "oriented_bbox": oriented_bbox,
            }
        except Exception as exc:
            if self.config.get("soft_observer_verbose_errors", False):
                print(f"[SoftObserver] point-bbox failed for {path}: {exc}")
            return None

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
        # Sphere records often carry radius but no explicit height.
        # The nominal height is the diameter.
        if target.get("radius") is not None:
            try:
                return 2.0 * float(target["radius"])
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