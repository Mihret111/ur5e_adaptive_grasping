"""
scene_builder.py
────────────────
Spawns room, table, and objects into the existing USD stage for each trial.

Owns:
  - Trial root prim management  (clear, ensure)
  - Physics scene creation
  - Gripper friction material
  - Room geometry
  - Table geometry + material
  - Object spawning + reach validation

Does NOT own:
  - Robot/arm control
  - Gripper actuation
  - Pick sequencing
  - Config loading
"""
from logging import root
import math
import os
import random
from typing import Optional

import omni.usd
from pxr import (
    Usd, UsdGeom, UsdPhysics, UsdShade,
    Sdf, Gf
)


class SceneBuilder:
    def __init__(
        self,
        config:           dict,
        table_materials:  list,
        table_seat_slots: list,
    ):
        self.config           = config
        self.stage            = omni.usd.get_context().get_stage()
        self.table_materials  = table_materials
        self.table_seat_slots = table_seat_slots
        self._spawned_objects = []
        self._pick_target     = None

        self._trial_root = (
            config
            .get("paths", {})
            .get("scene_spawn", {})
            .get("root", "/World/Trial")
        )

    # ══════════════════════════════════════════════════════════════════
    # PUBLIC
    # ══════════════════════════════════════════════════════════════════

    def build_trial(self, trial_index: int) -> dict:
        print(f"\n{'═' * 60}")
        print(f"  BUILDING TRIAL {trial_index}")
        print(f"{'═' * 60}")

        self._clear_trial()
        self._ensure_world_root()
        self._create_physics_scene()
        self._apply_gripper_friction()

        if self.config["build_room"]:
            self._build_room()

        table_info   = self._place_table()
        table_center = table_info["center"]
        table_slot   = table_info["slot"]

        # Phase 4.0: semantic table zones for reproducible pick-and-place.
        # These are visual/semantic markers only; by default they have no
        # collision so they behave like stickers on the table, not obstacles.
        table_zones = self._create_table_zones(table_info)
        if table_zones:
            table_info["zones"] = table_zones

        objects = self._spawn_objects(
            table_center=table_center,
            table_info=table_info,
        )

        self._pick_target = random.choice(objects)
        self._pick_target["is_target"] = True

        # --- FRAME CHECK ---
        ur5e_base_pos = self._get_ur5e_base_world_pos()
        obj_pos       = self._pick_target["world_pos"]
        # Get flange prim
        flange_path = (
            self.config
            .get("paths", {})
            .get("robot", {})
            .get(
                "flange_prim",
                "/mir/base_link_cabinet/cabinet/ur_mount/ur5e_physics/wrist_3_link/flange"
            )
        )

        flange_pos = self._get_prim_world_pos(flange_path)

        rel_obj_pos = [
            obj_pos[0] - ur5e_base_pos[0],
            obj_pos[1] - ur5e_base_pos[1],
            obj_pos[2] - ur5e_base_pos[2],
        ]

        print("\n  [FrameCheck]")
        print(f"    UR5e base world:  ({ur5e_base_pos[0]:.3f}, {ur5e_base_pos[1]:.3f}, {ur5e_base_pos[2]:.3f})")
        print(f"    Flange world:     ({flange_pos[0]:.3f}, {flange_pos[1]:.3f}, {flange_pos[2]:.3f})")
        print(f"    Target world:     ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        print(f"    Target rel base:  ({rel_obj_pos[0]:.3f}, {rel_obj_pos[1]:.3f}, {rel_obj_pos[2]:.3f})")

        ## Adding more detailed frame check - start
        robot_paths = self.config.get("paths", {}).get("robot", {})

        base_path = robot_paths.get(
            "ur5e_base_link",
            "/mir/base_link_cabinet/cabinet/ur_mount/ur5e_physics/base_link",
        )

        target_local_base = self._world_point_to_prim_local(
            reference_prim_path=base_path,
            world_point=obj_pos,
        )

        flange_local_base = self._world_point_to_prim_local(
            reference_prim_path=base_path,
            world_point=flange_pos,
        )

        print("\n  [FrameCheck - full transform]")
        print(
            f"    Target in UR5e base frame: "
            f"({target_local_base[0]:.3f}, {target_local_base[1]:.3f}, {target_local_base[2]:.3f})"
        )
        print(
            f"    Flange in UR5e base frame: "
            f"({flange_local_base[0]:.3f}, {flange_local_base[1]:.3f}, {flange_local_base[2]:.3f})"
        )
        ##
        dx = obj_pos[0] - ur5e_base_pos[0]
        dy = obj_pos[1] - ur5e_base_pos[1]
        pan_to_object = math.degrees(math.atan2(dy, dx))

        table_mat_dict = table_info["table_material"]
        table_mat_name = table_mat_dict["name"]       # e.g. "oak_light", "brushed_steel"

        print(f"\n  [Objects] Summary:")
        print(f"  ▶ PICK TARGET: {self._pick_target['label']}")
        print(f"    Object: ({obj_pos[0]:.3f}, "
              f"{obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        print(f"    Seat: {table_slot['name']}  "
              f"facing={table_slot['facing']}")
        print(f"    Table material: {table_mat_name}")    # FIX: log it

        return {
            "trial_index":      trial_index,
            "pick_target":      self._pick_target,
            "all_objects":      objects,
            "table_center":     table_center,
            "table_slot":       table_slot,
            "table_info":       table_info,
            "table_zones":      table_info.get("zones", {}),
            "table_material":   table_mat_name,           
            "table_height":     table_info["table_size"][2],
            "pan_to_table_deg": pan_to_object,
            "ur5e_base_pos":    ur5e_base_pos,
            "flange_world_pos": flange_pos,
            "target_relative_to_ur5e_base": rel_obj_pos,
            "target_local_base": target_local_base,
            "flange_local_base": flange_local_base,
        }

    def get_pick_target(self) -> Optional[dict]:
        return self._pick_target

    # ══════════════════════════════════════════════════════════════════
    # INTERNALS — scene lifecycle
    # ══════════════════════════════════════════════════════════════════

    def _ensure_world_root(self):
        for path in ["/World", self._trial_root]:
            prim = self.stage.GetPrimAtPath(Sdf.Path(path))
            if not prim.IsValid():
                UsdGeom.Xform.Define(self.stage, Sdf.Path(path))

    def _clear_trial(self):
        prim = self.stage.GetPrimAtPath(Sdf.Path(self._trial_root))
        if prim.IsValid():
            self.stage.RemovePrim(Sdf.Path(self._trial_root))
        self._spawned_objects = []
        self._pick_target     = None
        print("  [Scene] Cleared previous trial")

    def _create_physics_scene(self):
        """Ensure a PhysicsScene exists on stage. Never duplicates.
        Always applies anti-explosion PhysX settings.

        Attribute names confirmed for this Isaac Sim version:
            physxScene:solverType
            physxScene:minPositionIterationCount
            physxScene:minVelocityIterationCount
            physxScene:bounceThreshold
            physxScene:enableCCD
            physxScene:frictionOffsetThreshold
        """
        try:
            from pxr import PhysxSchema
        except ImportError:
            print("  [Physics] ⚠️  PhysxSchema not available — skipping anti-explosion settings")
            PhysxSchema = None

        # ── Find or create the physics scene prim ─────────────────────
        scene_prim = None

        known_paths = [
            "/World/PhysicsScene",
            "/physicsScene",
            f"{self._trial_root}/PhysicsScene",
        ]
        for path in known_paths:
            prim = self.stage.GetPrimAtPath(Sdf.Path(path))
            if prim.IsValid():
                print(f"  [Physics] Scene exists at {path}")
                scene_prim = prim
                break

        if scene_prim is None:
            for prim in self.stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    print(f"  [Physics] Scene exists at {prim.GetPath()}")
                    scene_prim = prim
                    break

        if scene_prim is None:
            scene_path = f"{self._trial_root}/PhysicsScene"
            scene = UsdPhysics.Scene.Define(
                self.stage, Sdf.Path(scene_path))
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_prim = scene.GetPrim()
            print(f"  [Physics] Created at {scene_path}")

        # ── Apply anti-explosion PhysX settings ───────────────────────
        if PhysxSchema is None or scene_prim is None:
            return

        # Confirmed attribute names for this Isaac Sim version
        settings = {
            "physxScene:solverType":                  "TGS",
            "physxScene:minPositionIterationCount":    32,
            "physxScene:minVelocityIterationCount":    8,
            "physxScene:bounceThreshold":              0.5,
            "physxScene:enableCCD":                    True,
            "physxScene:frictionOffsetThreshold":      0.001,
            "physxScene:enableStabilization":          True,   # extra jitter reduction
        }

        try:
            PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

            failed = []
            for attr_name, value in settings.items():
                attr = scene_prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(value)
                else:
                    failed.append(attr_name)

            if failed:
                print(f"  [Physics] ⚠️  Attrs not found: {failed}")

            print(
                f"  [Physics] Anti-explosion settings applied  "
                f"solver=TGS  pos_iter=32  vel_iter=8  "
                f"CCD=True  stabilization=True"
            )

        except Exception as e:
            print(f"  [Physics] ⚠️  Could not apply PhysxSceneAPI: {e}")

    def _apply_gripper_friction(self):
        """Apply friction material and anti-explosion solver settings to gripper finger prims."""
        mat_path  = f"{self._trial_root}/PhysicsMaterials/GripperMat"
        mats_root = f"{self._trial_root}/PhysicsMaterials"   

        fric = self.config.get("gripper_friction", {
            "static_friction":  2.0,
            "dynamic_friction": 2.0,
            "restitution":      0.0,
        })

        # ── Ensure parent prim exists ──────────────────────────────────
        if not self.stage.GetPrimAtPath(Sdf.Path(mats_root)).IsValid():
            UsdGeom.Xform.Define(self.stage, mats_root)          

        material = UsdShade.Material.Define(
            self.stage, Sdf.Path(mat_path))
        mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat_api.CreateStaticFrictionAttr().Set(fric["static_friction"])
        mat_api.CreateDynamicFrictionAttr().Set(fric["dynamic_friction"])
        mat_api.CreateRestitutionAttr().Set(fric["restitution"])

        # ── Read finger paths from paths config ────────────────────────
        gripper_paths = self.config.get("paths", {}).get("gripper", {})
        finger_paths  = [
            gripper_paths.get(
                "left_finger_link",  "/onrobot_2fg7/left_finger_link"),
            gripper_paths.get(
                "right_finger_link", "/onrobot_2fg7/right_finger_link"),
        ]

        count = 0
        for path in finger_paths:
            prim = self.stage.GetPrimAtPath(Sdf.Path(path))
            if prim.IsValid():
                prim.CreateRelationship(
                    "material:binding:physics"
                ).SetTargets([Sdf.Path(mat_path)])
                count += 1
                for child in prim.GetAllChildren():
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        child.GetPrim().CreateRelationship(
                            "material:binding:physics"
                        ).SetTargets([Sdf.Path(mat_path)])
                        count += 1

        print(f"  [Friction] Applied to {count} gripper prims")

        # ── Per-body solver iterations for gripper fingers ─────────────
        # Overrides the scene-level defaults specifically for finger bodies.
        # High K drives + small finger mass = constraint explosion without this.
        # 32 position iterations resolves the contact force in one timestep
        # instead of accumulating across multiple steps.
        try:
            from pxr import PhysxSchema

            pos_iter = self.config.get("gripper_solver_position_iterations", 32)
            vel_iter = self.config.get("gripper_solver_velocity_iterations", 8)

            iter_count = 0
            for path in finger_paths:
                prim = self.stage.GetPrimAtPath(Sdf.Path(path))
                if not prim.IsValid():
                    continue

                # Apply to the finger link itself
                rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb_api.CreateSolverPositionIterationCountAttr().Set(pos_iter)
                rb_api.CreateSolverVelocityIterationCountAttr().Set(vel_iter)
                iter_count += 1

                # Apply to all collision children too
                for child in prim.GetAllChildren():
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        rb_api_child = PhysxSchema.PhysxRigidBodyAPI.Apply(
                            child.GetPrim())
                        rb_api_child.CreateSolverPositionIterationCountAttr().Set(
                            pos_iter)
                        rb_api_child.CreateSolverVelocityIterationCountAttr().Set(
                            vel_iter)
                        iter_count += 1

            print(
                f"  [Friction] Solver iterations set on {iter_count} "
                f"gripper prims  pos={pos_iter}  vel={vel_iter}"
            )

        except Exception as e:
            print(f"  [Friction] ⚠️  Could not set solver iterations: {e}")
            print(f"             Explosion risk from gripper contacts remains")

    ## A helper to retrieve the world position of the ur5e base link
    def _get_ur5e_base_world_pos(self) -> list:
        """Return UR5E base_link world position as [x, y, z]."""
        robot_paths = self.config.get("paths", {}).get("robot", {})
        base_path   = robot_paths.get(
            "ur5e_base_link",
            "/mir/base_link_cabinet/cabinet/ur_mount/ur5e_physics/base_link",
        )

        prim = self.stage.GetPrimAtPath(Sdf.Path(base_path))
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            mtx   = xform.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            pos   = mtx.ExtractTranslation()
            return [pos[0], pos[1], pos[2]]

        print("  ⚠️  UR5E base_link not found — using fallback position")
        return [0.0, 0.0, self.config.get("ur5e_base_height", 0.8593)]
    
    ## A helper to retrieve the world position of any prim
    def _get_prim_world_pos(self, prim_path: str) -> list:
        """Return world position of a USD prim as [x, y, z]."""
        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))

        if not prim.IsValid():
            print(f"  ⚠️  Prim not found: {prim_path}")
            return [0.0, 0.0, 0.0]

        xform = UsdGeom.Xformable(prim)
        mtx = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = mtx.ExtractTranslation()

        return [pos[0], pos[1], pos[2]]

    ## A helper to convert a world-space point into the local frame of a given USD prim
    def _world_point_to_prim_local(self, reference_prim_path: str, world_point: tuple) -> list:
        """
        Convert a world-space point into the local coordinate frame of a USD prim.

        This is better than simple subtraction because it also accounts for
        the base prim's rotation.
        """
        prim = self.stage.GetPrimAtPath(Sdf.Path(reference_prim_path))

        if not prim.IsValid():
            print(f"  ⚠️  Reference prim not found: {reference_prim_path}")
            return [0.0, 0.0, 0.0]

        xform = UsdGeom.Xformable(prim)
        world_from_local = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local_from_world = world_from_local.GetInverse()

        p_world = Gf.Vec3d(
            float(world_point[0]),
            float(world_point[1]),
            float(world_point[2]),
        )

        p_local = local_from_world.Transform(p_world)

        return [p_local[0], p_local[1], p_local[2]]

    # ══════════════════════════════════════════════════════════════════
    # ROOM
    # ══════════════════════════════════════════════════════════════════

    def _build_room(self):
        root = f"{self._trial_root}/Room"
        UsdGeom.Xform.Define(self.stage, root)

        rx, ry, rz = self.config["room_size"]
        t           = self.config["room_wall_thickness"]

        self._make_box(
            path=f"{root}/Floor",
            size=(rx, ry, t),
            position=(0, 0, -t / 2),
            color=self.config["floor_color"],
            is_static=True,
        )

        walls = [
            ("Wall_PosX", (t, ry, rz),
             (rx / 2 + t / 2, 0, rz / 2)),
            ("Wall_NegX", (t, ry, rz),
             (-rx / 2 - t / 2, 0, rz / 2)),
            ("Wall_PosY", (rx + 2 * t, t, rz),
             (0,  ry / 2 + t / 2, rz / 2)),
            ("Wall_NegY", (rx + 2 * t, t, rz),
             (0, -ry / 2 - t / 2, rz / 2)),
        ]
        for name, size, pos in walls:
            self._make_box(
                path=f"{root}/{name}",
                size=size, position=pos,
                color=self.config["room_color"],
                is_static=True,
            )

        print("  [Room] Built")

    # ══════════════════════════════════════════════════════════════════
    # TABLE
    # ══════════════════════════════════════════════════════════════════

    def _place_table(self) -> dict:
        root = f"{self._trial_root}/Table"
        UsdGeom.Xform.Define(self.stage, root)

        # ── Table size ─────────────────────────────────────────────────
        size_range = self.config.get("table_size_range", None)
        if size_range:
            tw = random.uniform(*size_range["width"])
            td = random.uniform(*size_range["depth"])
            th = random.uniform(*size_range["height"])
        else:
            tw, td, th = self.config["table_size"]
            
        ## ── Fixed debug table override ─────────────────────────────────
        debug_fixed_table = self.config.get("debug_fixed_table", False)

        if debug_fixed_table:
            fixed_size = self.config.get("debug_fixed_table_size", [1.00, 0.60, 0.75])
            tw, td, th = fixed_size

            # IMPORTANT:
            # "long" means the long table edge is the facing/sideways edge,
            # and the short table dimension becomes the approach depth.
            facing = "long"
            approach_deg = float(self.config.get("debug_fixed_table_approach_deg", 0.0))
            approach_rad = math.radians(approach_deg)
            seat_offset = 0.0
            slot = {
                "name": "debug_fixed_front_short_depth",
                "approach_deg": approach_deg,
                "facing": facing,
                "seat_offset": seat_offset,
            }

            table_mat = random.choice(self.table_materials)

        else:
            # ── Material + seat slot ───────────────────────────────────────
            table_mat = random.choice(self.table_materials)
            slot = random.choice(self.table_seat_slots)
            approach_deg = slot["approach_deg"]
            approach_rad = math.radians(approach_deg)
            facing = slot["facing"]
            seat_offset = slot["seat_offset"]

        # ── Geometry based on facing direction ─────────────────────────
        if facing == "long":
            facing_edge  = tw
            depth_edge   = td
            box_approach = td
            box_perp     = tw
        else:
            facing_edge  = td
            depth_edge   = tw
            box_approach = tw
            box_perp     = td

        # ── Near-edge distance (reach-aware) ───────────────────────────
        arm_min = self.config.get("arm_min_reach", 0.40)
        arm_max = self.config.get("arm_max_reach", 0.80)
        pad     = self.config.get("near_edge_padding", 0.03)
        margin  = self.config.get("object_margin", 0.04)

        approach_max = min(depth_edge / 3.0, 0.25)
        near_min     = max(arm_min - margin + pad, 0.35)
        near_max     = arm_max - approach_max - pad

        if near_min >= near_max:
            near = (near_min + near_max) / 2.0
            print(
                f"  [Table] ⚠️  Tight reach: near forced to {near:.3f}m"
            )
        else:
            near = random.uniform(near_min, near_max)

        print(
            f"  [Table] Near edge: {near:.3f}m  "
            f"(range [{near_min:.3f}, {near_max:.3f}])  "
            f"approach_max={approach_max:.3f}m"
        )
        print(
            f"  [Table] Reach zone: {near:.3f}–{near + approach_max:.3f}m  "
            f"(arm: {arm_min:.2f}–{arm_max:.2f}m)"
        )

        # ── Table centre ───────────────────────────────────────────────
        ur5e_pos    = self._get_ur5e_base_world_pos()
        centre_dist = near + depth_edge / 2

        ax = math.cos(approach_rad)
        ay = math.sin(approach_rad)
        px = -math.sin(approach_rad)
        py = math.cos(approach_rad)


        # previous random table generation logic has been replaced with a fixed table generation logic
        # tcx = ur5e_pos[0] + centre_dist * ax + seat_offset * px
        # tcy = ur5e_pos[1] + centre_dist * ay + seat_offset * py
        # table_center  = (tcx, tcy, 0.0)
        # table_rot_deg = approach_deg
        # top_thickness = 0.04
        tcx = ur5e_pos[0] + centre_dist * ax + seat_offset * px
        tcy = ur5e_pos[1] + centre_dist * ay + seat_offset * py
        table_center = (tcx, tcy, 0.0)
        table_rot_deg = approach_deg

        if self.config.get("debug_fixed_table", False):
            fixed_center = self.config.get("debug_fixed_table_center", [1.10, 0.0, 0.0])
            fixed_approach_deg = self.config.get("debug_fixed_table_approach_deg", 0.0)
            fixed_facing = self.config.get("debug_fixed_table_facing", "long")

            tcx = float(fixed_center[0])
            tcy = float(fixed_center[1])
            table_center = (tcx, tcy, 0.0)

            approach_deg = float(fixed_approach_deg)
            approach_rad = math.radians(approach_deg)
            table_rot_deg = approach_deg
            facing = fixed_facing

            ax = math.cos(approach_rad)
            ay = math.sin(approach_rad)
            px = -math.sin(approach_rad)
            py = math.cos(approach_rad)

            print(
                f"  [Table] DEBUG fixed table enabled: "
                f"center=({tcx:.2f}, {tcy:.2f}) "
                f"approach={approach_deg:.1f}° facing={facing}"
            )

        top_thickness = 0.04

        # ── Tabletop ───────────────────────────────────────────────────
        top_path = f"{root}/Top"
        self._make_box(
            path=top_path,
            size=(box_approach, box_perp, top_thickness),
            position=(tcx, tcy, th - top_thickness / 2),
            color=table_mat["color"],
            is_static=True,
            rotation_z_deg=table_rot_deg,
        )

        top_prim = self.stage.GetPrimAtPath(Sdf.Path(top_path))
        if top_prim.IsValid():
            self._apply_table_material(top_prim, table_mat)
            self._apply_physx_contact_offsets_recursive(
                top_prim,
                contact_offset_m=float(self.config.get("table_contact_offset_m", 0.001)),
                rest_offset_m=float(self.config.get("table_rest_offset_m", 0.0)),
                label="table_top",
            )

        # ── Legs ───────────────────────────────────────────────────────
        leg_r   = self.config["table_leg_radius"]
        leg_h   = th - top_thickness
        leg_inset = leg_r * 2 + 0.02

        leg_offsets = [
            ( box_approach / 2 - leg_inset,  box_perp / 2 - leg_inset),
            (-box_approach / 2 + leg_inset,  box_perp / 2 - leg_inset),
            ( box_approach / 2 - leg_inset, -box_perp / 2 + leg_inset),
            (-box_approach / 2 + leg_inset, -box_perp / 2 + leg_inset),
        ]

        # 30% chance of metal legs on non-metal tables
        leg_mat = table_mat
        if table_mat["category"] in ("wood", "lacquer", "plastic"):
            if random.random() < 0.30:
                metal_legs = [
                    m for m in self.table_materials
                    if m["category"] == "metal"
                ]
                if metal_legs:
                    leg_mat = random.choice(metal_legs)

        for i, (la, lp) in enumerate(leg_offsets):
            wx = tcx + la * ax + lp * px
            wy = tcy + la * ay + lp * py
            leg_path = f"{root}/Leg_{i}"
            self._make_cylinder(
                path=leg_path, radius=leg_r, height=leg_h,
                position=(wx, wy, leg_h / 2),
                color=leg_mat["color"], is_static=True,
            )
            leg_prim = self.stage.GetPrimAtPath(Sdf.Path(leg_path))
            if leg_prim.IsValid():
                self._apply_table_material(leg_prim, leg_mat)

        print(
            f"  [Table] Size: {tw:.2f}×{td:.2f}×{th:.2f}m  "
            f"Material: {table_mat['name']} ({table_mat['category']})  "
            f"rough={table_mat['roughness']:.2f}  "
            f"metal={table_mat['metallic']:.2f}"
        )
        if leg_mat != table_mat:
            print(
                f"  [Table] Legs: {leg_mat['name']} ({leg_mat['category']})"
            )
        print(
            f"  [Table] Centre ({tcx:.2f}, {tcy:.2f})  h={th:.2f}m  "
            f"Facing: {facing} edge ({facing_edge:.2f}m)  "
            f"depth={depth_edge:.2f}m"
        )
        print(
            f"  [Table] Approach: {approach_deg:.0f}°  "
            f"seat: {slot['name']}  near={near:.2f}m"
        )

        return {
            "center":         table_center,
            "slot":           slot,
            "approach_rad":   approach_rad,
            "facing":         facing,
            "facing_edge":    facing_edge,
            "depth_edge":     depth_edge,
            "table_rot_deg":  table_rot_deg,
            "table_size":     (tw, td, th),
            "table_material": table_mat,
            "leg_material":   leg_mat,
        }

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4.0 — SEMANTIC TABLE ZONES / VISUAL MARKERS
    # ══════════════════════════════════════════════════════════════════

    def _create_table_zones(self, table_info: dict) -> dict:
        """Create visual pickup/place zones on the table.

        The zones are semantic/environment markers for the pick-and-place task.
        By default they are *visual only*: no collision, no rigid body, no mass.
        This keeps them equivalent to stickers/tape on a real table and avoids
        changing the contact physics of the soft object.

        Coordinates are expressed in table-local axes:
          local_x = approach/depth direction
          local_y = sideways/perpendicular direction
        and are converted to world using the same table axes used by object
        spawning.
        """
        if not bool(self.config.get("table_zones_enabled", True)):
            return {}

        zones_cfg = self.config.get("table_zones", {}) or {}
        if not zones_cfg:
            # Safe defaults for the current fixed table and 40 mm foam cube.
            zones_cfg = {
                "pickup_zone": {
                    "label": "pickup_zone",
                    "local_xy_m": [-0.07, 0.0],
                    "size_xy_m": [0.12, 0.12],
                    "color": [0.10, 0.35, 1.00],
                    "cross_color": [1.00, 1.00, 1.00],
                },
                "place_zone": {
                    "label": "place_zone",
                    "local_xy_m": [-0.07, 0.20],
                    "size_xy_m": [0.12, 0.12],
                    "color": [0.10, 0.80, 0.20],
                    "cross_color": [1.00, 1.00, 1.00],
                },
            }

        root = f"{self._trial_root}/TableZones"
        UsdGeom.Xform.Define(self.stage, Sdf.Path(root))

        tcx, tcy, _ = table_info["center"]
        approach_rad = float(table_info.get("approach_rad", 0.0))
        table_rot_deg = float(table_info.get("table_rot_deg", math.degrees(approach_rad)))
        ax = math.cos(approach_rad)
        ay = math.sin(approach_rad)
        px = -math.sin(approach_rad)
        py = math.cos(approach_rad)

        table_surface_z = float(table_info["table_size"][2])
        marker_thickness = float(self.config.get("table_zone_marker_thickness_m", 0.001))
        marker_z_offset = float(self.config.get("table_zone_marker_z_offset_m", 0.001))
        marker_z = table_surface_z + marker_z_offset
        collision_enabled = bool(self.config.get("table_zone_marker_collision_enabled", False))
        crosshair_enabled_default = bool(self.config.get("table_zone_crosshair_enabled", True))

        facing_edge = float(table_info.get("facing_edge", table_info["table_size"][0]))
        depth_edge = float(table_info.get("depth_edge", table_info["table_size"][1]))

        zone_records = {}
        for zone_key, zone in zones_cfg.items():
            local_xy = zone.get("local_xy_m", [0.0, 0.0])
            size_xy = zone.get("size_xy_m", [0.12, 0.12])
            color = tuple(zone.get("color", [0.2, 0.6, 1.0]))
            cross_color = tuple(zone.get("cross_color", [1.0, 1.0, 1.0]))
            label = zone.get("label", zone_key)

            la = float(local_xy[0])
            lp = float(local_xy[1])
            sx = float(size_xy[0])
            sy = float(size_xy[1])

            wx = tcx + la * ax + lp * px
            wy = tcy + la * ay + lp * py

            inside_table = (
                abs(la) + sx / 2.0 <= depth_edge / 2.0
                and abs(lp) + sy / 2.0 <= facing_edge / 2.0
            )

            zone_root = f"{root}/{zone_key}"
            UsdGeom.Xform.Define(self.stage, Sdf.Path(zone_root))

            marker_path = f"{zone_root}/Marker"
            self._make_visual_box(
                path=marker_path,
                size=(sx, sy, marker_thickness),
                position=(wx, wy, marker_z),
                color=color,
                rotation_z_deg=table_rot_deg,
                collision_enabled=collision_enabled,
            )

            if bool(zone.get("crosshair", crosshair_enabled_default)):
                strip = float(zone.get("crosshair_strip_width_m", 0.008))
                strip_z = marker_z + marker_thickness * 0.75
                self._make_visual_box(
                    path=f"{zone_root}/Cross_Long",
                    size=(sx * 0.86, strip, marker_thickness * 1.2),
                    position=(wx, wy, strip_z),
                    color=cross_color,
                    rotation_z_deg=table_rot_deg,
                    collision_enabled=False,
                )
                self._make_visual_box(
                    path=f"{zone_root}/Cross_Short",
                    size=(strip, sy * 0.86, marker_thickness * 1.2),
                    position=(wx, wy, strip_z),
                    color=cross_color,
                    rotation_z_deg=table_rot_deg,
                    collision_enabled=False,
                )

            zone_records[zone_key] = {
                "key": zone_key,
                "label": label,
                "local_xy_m": [la, lp],
                "world_center": [wx, wy, marker_z],
                "table_surface_z": table_surface_z,
                "marker_path": marker_path,
                "size_xy_m": [sx, sy],
                "marker_thickness_m": marker_thickness,
                "collision_enabled": collision_enabled,
                "inside_table": inside_table,
                "role": zone.get("role", zone_key),
            }

            print(
                f"  [TableZone] {label}: local=({la:.3f},{lp:.3f}) "
                f"world=({wx:.3f},{wy:.3f},{marker_z:.4f}) "
                f"size=({sx:.3f},{sy:.3f}) collision={collision_enabled} "
                f"inside_table={inside_table}"
            )

            if not inside_table:
                print(
                    f"  [TableZone] ⚠ {label} marker may extend outside table bounds; "
                    f"depth_edge={depth_edge:.3f}, facing_edge={facing_edge:.3f}"
                )

        return zone_records

    def _make_visual_box(
        self,
        path: str,
        size: tuple,
        position: tuple,
        color: tuple,
        rotation_z_deg: float = 0.0,
        collision_enabled: bool = False,
    ):
        """Make a visual cube marker, optionally with collision.

        Used for table stickers/semantic zones.  Unlike _make_box(), this does
        not automatically apply CollisionAPI, because markers should normally
        not change soft-object contact physics.
        """
        cube = UsdGeom.Cube.Define(self.stage, Sdf.Path(path))
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        if rotation_z_deg != 0.0:
            xf.AddRotateZOp().Set(rotation_z_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(*size))

        self._apply_display_color(cube.GetPrim(), color)
        if collision_enabled:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # ══════════════════════════════════════════════════════════════════
    # RANDOM OBJECT GENERATION
    # ══════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════
    # SOFT OBJECT CATALOGUE HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve_asset_path(self, asset_path: str) -> str:
        if not asset_path:
            return ""
        expanded = os.path.expanduser(asset_path)
        if os.path.isabs(expanded):
            return expanded
        return os.path.join(self._project_root(), expanded)

    def _select_weighted_entry(self, entries: list) -> dict:
        weights = [float(e.get("weight", 1.0)) for e in entries]
        return random.choices(entries, weights=weights, k=1)[0]

    def _material_from_name(self, name: str, fallback: dict = None) -> dict:
        """Return material dictionary by name from soft or normal material lists."""
        fallback = fallback or {
            "name": name or "soft_default",
            "static_friction": 0.8,
            "dynamic_friction": 0.7,
            "restitution": 0.05,
        }
        for key in ("soft_physics_materials", "object_physics_materials"):
            for mat in self.config.get(key, []) or []:
                if mat.get("name") == name:
                    return mat
        return fallback

    def _generate_soft_catalog_object(self, index: int, used_labels: set) -> dict:
        """Generate one soft/deformable object definition from catalogue.

        This supports two development modes:
        - primitive_proxy: uses Cube/Rectangle/Cylinder/Sphere primitive geometry.
        - deformable_usd/usd_reference: references an imported USD asset if it exists.

        The object record always carries compliance/fragility fields so the
        action-selection layer can choose delicate grasping.
        """
        catalog = self.config.get("soft_object_catalog", []) or []
        if not catalog:
            return None

        # Phase 7: for multi-object benchmark mode we need deterministic,
        # ordered catalogue spawning rather than weighted random sampling.
        # This lets one trial contain exactly the prepared object set
        # (cube, roller, ball, later disc) and makes repeatability tables
        # meaningful.
        if bool(self.config.get("soft_object_spawn_catalog_in_order", False)):
            tmpl = dict(catalog[index % len(catalog)])
        else:
            tmpl = dict(self._select_weighted_entry(catalog))
        shape = tmpl.get("shape", "Cube")
        obj_def = {"shape": shape}

        grip_dim = float(tmpl.get("grip_dim_mm", 50.0)) / 1000.0

        if shape == "Cube":
            side = float(tmpl.get("size_mm", tmpl.get("height_mm", tmpl.get("grip_dim_mm", 50.0)))) / 1000.0
            obj_def["size"] = round(side, 4)
            grip_mm = grip_dim * 1000

        elif shape == "Rectangle":
            width = float(tmpl.get("width_mm", tmpl.get("grip_dim_mm", 48.0))) / 1000.0
            length = float(tmpl.get("length_mm", width * 1000.0 * 1.4)) / 1000.0
            height = float(tmpl.get("height_mm", 35.0)) / 1000.0
            obj_def.update({
                "width": round(width, 4),
                "length": round(length, 4),
                "height": round(height, 4),
            })
            grip_mm = float(tmpl.get("grip_dim_mm", width * 1000.0))

        elif shape in ("Cylinder", "Disc"):
            radius = grip_dim / 2.0
            height = float(tmpl.get("height_mm", 50.0)) / 1000.0
            obj_def["radius"] = round(radius, 4)
            obj_def["height"] = round(height, 4)
            grip_mm = grip_dim * 1000

        elif shape == "Sphere":
            radius = grip_dim / 2.0
            obj_def["radius"] = round(radius, 4)
            # Keep a height/diameter field for all downstream modules.
            # Without this, the safety monitor may fall back to the global
            # 40 mm nominal height, which is wrong for larger soft balls.
            obj_def["height"] = round(2.0 * radius, 4)
            grip_mm = grip_dim * 1000

        else:
            raise ValueError(f"Unknown soft object shape: {shape}")

        if grip_mm < 45:
            size_cat = "small"
        elif grip_mm < 58:
            size_cat = "medium"
        else:
            size_cat = "large"

        color = tuple(tmpl.get("color", tmpl.get("visual_color", (0.9, 0.8, 0.4))))
        # Accept both the old scaffold key (material_name) and the clearer
        # research-config key (material).
        material_name = tmpl.get("material_name", tmpl.get("material", "foam"))

        material_fallback = {
            "name": material_name,
            "static_friction": float(tmpl.get("static_friction", 0.8)),
            "dynamic_friction": float(tmpl.get("dynamic_friction", 0.7)),
            "restitution": float(tmpl.get("restitution", 0.05)),
        }
        material = self._material_from_name(material_name, fallback=material_fallback)

        # Accept mass_kg from the scientifically documented soft object profile.
        mass = float(tmpl.get("mass_kg", tmpl.get("mass", 0.06)))

        base_label = tmpl.get("label", f"{size_cat}_{material_name}_{shape.lower()}")
        label = base_label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_labels.add(label)

        name = f"{tmpl.get('name', base_label)}_{index}"

        obj_def.update({
            "name": name,
            "label": label,
            "color": color,
            "mass": mass,
            "material": material,
            "material_name": material.get("name", material_name),
            "static_friction": float(material.get("static_friction", 0.8)),
            "dynamic_friction": float(material.get("dynamic_friction", 0.7)),
            "restitution": float(material.get("restitution", 0.05)),
            "grip_dim_mm": round(float(grip_mm), 1),
            "size_category": size_cat,
            "color_name": tmpl.get("color_name", material_name),
            "asset_type": tmpl.get("asset_type", "primitive_proxy"),
            "asset_path": tmpl.get("asset_path"),
            "resolved_asset_path": self._resolve_asset_path(tmpl.get("asset_path", "")),
            "asset_scale": tmpl.get("asset_scale", None),
            "asset_reference_height_m": tmpl.get("asset_reference_height_m", 1.0),
            "asset_origin_z": tmpl.get("asset_origin_z", "auto_bbox"),
            "spawn_pose_mode": tmpl.get("spawn_pose_mode"),
            "spawn_local_xy_m": tmpl.get("spawn_local_xy_m"),
            "spawn_yaw_deg_fixed": tmpl.get("spawn_yaw_deg"),
            # Preserve benchmark semantics from soft_objects.yaml.  These keys
            # were previously lost, so Phase 7 fell back to generic
            # place_zone_0/1/2 and logs looked like the old single-object path.
            "place_zone_key": tmpl.get("place_zone_key"),
            "pickup_zone_key": tmpl.get("pickup_zone_key"),
            "phase7_object_role": tmpl.get("phase7_object_role"),
            "batch_order": tmpl.get("batch_order"),
            "compliance": tmpl.get("compliance", "soft"),
            "deformable": bool(tmpl.get("deformable", True)),
            "fragile": bool(tmpl.get("fragile", True)),
            "fragility": tmpl.get("fragility", "medium"),
            "thin_object": bool(tmpl.get("thin_object", False)),
            "density_kg_m3": tmpl.get("density_kg_m3"),
            "youngs_modulus_pa": tmpl.get("youngs_modulus_pa"),
            "poissons_ratio": tmpl.get("poissons_ratio"),
            "preferred_force_n": tmpl.get("preferred_force_n"),
            "max_force_n": tmpl.get("max_force_n"),
            # Accept both the internal key and the YAML-facing key.
            "gripper_hold_extra_close_m": tmpl.get(
                "gripper_hold_extra_close_m", tmpl.get("hold_extra_m")
            ),
            "micro_lift_speed_scale": tmpl.get("micro_lift_speed_scale"),
            "close_speed_scale": tmpl.get("close_speed_scale"),
            # Phase 7.4: preserve per-object runtime control keys.  Earlier
            # patches wrote these in soft_objects.yaml, but they were silently
            # dropped here, so the rubber_ball still used global adaptive safety
            # and generic gripper behavior.
            "close_expected_grip_dim_mm": tmpl.get("close_expected_grip_dim_mm"),
            "close_contact_shell_extra_m": tmpl.get("close_contact_shell_extra_m"),
            "adaptive_effort_control_enabled": tmpl.get("adaptive_effort_control_enabled"),
            "adaptive_effort_target_sim": tmpl.get("adaptive_effort_target_sim"),
            "adaptive_effort_max_relax_open_m": tmpl.get("adaptive_effort_max_relax_open_m"),
            "adaptive_effort_max_extra_close_m": tmpl.get("adaptive_effort_max_extra_close_m"),
            "soft_pose_source": tmpl.get("soft_pose_source"),
            "grasp_height_mode": tmpl.get("grasp_height_mode"),
            "grasp_center_height_fraction": tmpl.get("grasp_center_height_fraction"),
            "grasp_z_offset_m": tmpl.get("grasp_z_offset_m"),
            "min_fingertip_clearance_m": tmpl.get("min_fingertip_clearance_m"),
            "allow_lower_than_global_fingertip_clearance": tmpl.get("allow_lower_than_global_fingertip_clearance"),
            "sphere_allow_probationary_partial_micro_lift": tmpl.get("sphere_allow_probationary_partial_micro_lift"),
            "sphere_partial_micro_lift_min_dz_m": tmpl.get("sphere_partial_micro_lift_min_dz_m"),
            "sphere_partial_micro_lift_min_following_ratio": tmpl.get("sphere_partial_micro_lift_min_following_ratio"),
            "sphere_partial_micro_lift_max_drift_m": tmpl.get("sphere_partial_micro_lift_max_drift_m"),
            "sphere_micro_lift_rescue_delta_m": tmpl.get("sphere_micro_lift_rescue_delta_m"),
            "sphere_micro_lift_rescue_settle_seconds": tmpl.get("sphere_micro_lift_rescue_settle_seconds"),
            "sphere_micro_lift_rescue_force_n": tmpl.get("sphere_micro_lift_rescue_force_n"),
            "gripper_use_expected_contact_floor": tmpl.get("gripper_use_expected_contact_floor"),
            "gripper_capture_floor_note": tmpl.get("gripper_capture_floor_note"),
            "sphere_micro_lift_preload_enabled": tmpl.get("sphere_micro_lift_preload_enabled"),
            "sphere_micro_lift_preload_delta_m": tmpl.get("sphere_micro_lift_preload_delta_m"),
            "sphere_micro_lift_preload_settle_seconds": tmpl.get("sphere_micro_lift_preload_settle_seconds"),
            "sphere_micro_lift_preload_force_n": tmpl.get("sphere_micro_lift_preload_force_n"),
            "target_stability_max_drift_m": tmpl.get("target_stability_max_drift_m"),
            "target_stability_max_xy_drift_m": tmpl.get("target_stability_max_xy_drift_m"),
            "target_stability_retry_wait_seconds": tmpl.get("target_stability_retry_wait_seconds"),
            # Phase 7.8: generic behavior-profile and transport policy keys.
            # These avoid hard-coding one label (for example rubber_ball) in the executor.
            "grasp_behavior_profile": tmpl.get("grasp_behavior_profile"),
            "transport_effort_policy": tmpl.get("transport_effort_policy"),
            "transport_abort_on_effort_over_max": tmpl.get("transport_abort_on_effort_over_max"),
            "place_transport_admittance_max_effort_sim": tmpl.get("place_transport_admittance_max_effort_sim"),
            "adaptive_safety_effort_enabled": tmpl.get("adaptive_safety_effort_enabled"),
            "use_target_nominal_for_safety": tmpl.get("use_target_nominal_for_safety"),
            "carry_bottom_clearance_m": tmpl.get("carry_bottom_clearance_m"),
            "full_lift_bottom_clearance_m": tmpl.get("full_lift_bottom_clearance_m"),
            "target_nominal_width_m": tmpl.get("target_nominal_width_m"),
            "target_nominal_depth_m": tmpl.get("target_nominal_depth_m"),
            "target_nominal_height_m": tmpl.get("target_nominal_height_m"),
            "strategy_hint": tmpl.get("strategy_hint", "delicate_pick"),
            "validation_profile": tmpl.get("validation_profile", "soft"),
            "fallback_primitive": tmpl.get("fallback_primitive"),
            "soft_object": True,
        })

        # Phase 7.6A hotfix: do not propagate optional keys with explicit None.
        # Python dict.get(key, fallback) returns None when key exists with None,
        # so these None values can later crash float(None) in the executor.
        for _optional_key in list(obj_def.keys()):
            if obj_def.get(_optional_key) is None:
                obj_def.pop(_optional_key, None)

        # If the USD is authored as a 1-unit object, derive a metres scale from
        # the scientific height declared in the catalogue.  This prevents a
        # 50 mm foam cube from being referenced as a 1 metre cube and colliding
        # with the table/robot during commissioning.
        if obj_def.get("asset_type") in ("usd_reference", "deformable_usd"):
            if obj_def.get("asset_scale") is None:
                declared_height_m = float(tmpl.get("height_mm", tmpl.get("grip_dim_mm", 50.0))) / 1000.0
                ref_height_m = max(float(obj_def.get("asset_reference_height_m", 1.0) or 1.0), 1e-6)
                obj_def["asset_scale"] = declared_height_m / ref_height_m
            obj_def["asset_scale"] = float(obj_def.get("asset_scale", 1.0))

        return obj_def

    def _generate_random_object(self, index: int, used_labels: set) -> dict:
        """
        Generate a single random object definition by combining:
        shape + dimensions + color + mass + material

        Returns a dict with all info needed to spawn and label the object.
        """
        # ── 0. Optional soft/deformable catalogue ─────────────────────
        if self.config.get("soft_object_catalog_enabled", False):
            prob = float(self.config.get("soft_object_spawn_probability", 1.0))
            if random.random() <= prob:
                soft_obj = self._generate_soft_catalog_object(index, used_labels)
                if soft_obj is not None:
                    return soft_obj

        # ── 1. Pick shape (weighted) ──────────────────────────────────
        shape_defs = self.config["shapes"]
        weights    = [s.get("weight", 1.0) for s in shape_defs]
        total_w    = sum(weights)
        probs      = [w / total_w for w in weights]

        shape_def = random.choices(shape_defs, weights=probs, k=1)[0]
        shape     = shape_def["name"]

        # ── 2. Grip dimension (constrained to gripper range) ──────────
        grip_cfg = self.config.get("grip_range_mm", {"min": 35, "max": 73})
        grip_min = grip_cfg["min"] / 1000.0
        grip_max = grip_cfg["max"] / 1000.0

        grip_dim = random.uniform(grip_min, grip_max)

        # ── 3. Build geometry from shape + grip dimension ─────────────
        obj_def = {"shape": shape}

        if shape == "Cube":
            side = grip_dim
            obj_def["size"] = round(side, 4)
            grip_mm = side * 1000

        elif shape == "Rectangle":
            width  = grip_dim
            ratio  = random.uniform(
                *shape_def.get("length_ratio", [1.3, 2.0]))
            length = width * ratio
            h_range = shape_def.get("height_range", [0.018, 0.040])
            height = random.uniform(*h_range)
            obj_def["width"]  = round(width, 4)
            obj_def["length"] = round(length, 4)
            obj_def["height"] = round(height, 4)
            grip_mm = width * 1000

        elif shape == "Cylinder":
            radius = grip_dim / 2.0
            h_range = shape_def.get("height_range", [0.035, 0.095])
            height = random.uniform(*h_range)
            obj_def["radius"] = round(radius, 4)
            obj_def["height"] = round(height, 4)
            grip_mm = grip_dim * 1000

        elif shape == "Disc":
            radius = grip_dim / 2.0
            h_range = shape_def.get("height_range", [0.008, 0.022])
            height = random.uniform(*h_range)
            obj_def["radius"] = round(radius, 4)
            obj_def["height"] = round(height, 4)
            grip_mm = grip_dim * 1000

        elif shape == "Sphere":
            radius = grip_dim / 2.0
            obj_def["radius"] = round(radius, 4)
            # Keep a height/diameter field for all downstream modules.
            # Without this, the safety monitor may fall back to the global
            # 40 mm nominal height, which is wrong for larger soft balls.
            obj_def["height"] = round(2.0 * radius, 4)
            grip_mm = grip_dim * 1000

        else:
            raise ValueError(f"Unknown shape: {shape}")

        # ── 4. Size category from grip dimension ──────────────────────
        if grip_mm < 45:
            size_cat = "small"
        elif grip_mm < 58:
            size_cat = "medium"
        else:
            size_cat = "large"

        # ── 5. Pick color ─────────────────────────────────────────────
        color_def  = random.choice(self.config["colors"])
        color_name = color_def["name"]
        color_rgb  = tuple(color_def["rgb"])

        # ── 6. Pick physics material ──────────────────────────────────
        materials = self.config["object_physics_materials"]
        material  = random.choice(materials)

        # ── 7. Randomize mass ─────────────────────────────────────────
        mass_cfg = self.config["object_mass"]
        mass     = round(random.uniform(
            mass_cfg["min_kg"], mass_cfg["max_kg"]), 3)

        # ── 8. Generate unique label ──────────────────────────────────
        shape_label = shape.lower()
        base_label  = f"{size_cat}_{color_name}_{shape_label}"

        label  = base_label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_labels.add(label)

        # ── 9. Internal name (for prim path) ──────────────────────────
        name = f"{color_name}_{shape_label}_{int(grip_mm)}_{index}"

        # ── 10. Assemble full definition ──────────────────────────────
        obj_def.update({
            "name":             name,
            "label":            label,
            "color":            color_rgb,
            "mass":             mass,
            "material":         material,
            "material_name":    material["name"],
            "static_friction":  float(material["static_friction"]),
            "dynamic_friction": float(material.get(
                "dynamic_friction", 0.3)),
            "restitution":      float(material.get("restitution", 0.1)),
            "grip_dim_mm":      round(grip_mm, 1),
            "size_category":    size_cat,
            "color_name":       color_name,
            "soft_object":      False,
            "compliance":       "rigid",
            "deformable":       False,
            "fragile":          material["name"] in ("glass", "ceramic"),
        })

        return obj_def

    def _get_half_height(self, obj_def: dict) -> float:
        """Return half-height in metres for Z positioning."""
        shape = obj_def["shape"]
        if shape == "Cube":
            return obj_def["size"] / 2.0
        elif shape in ("Rectangle",):
            return obj_def["height"] / 2.0
        elif shape in ("Cylinder", "Disc"):
            return obj_def["height"] / 2.0
        elif shape == "Sphere":
            return obj_def["radius"]
        return 0.02

    def _get_object_dims_str(self, obj_def: dict) -> str:
        """Human-readable dimension string for logging."""
        shape = obj_def["shape"]
        if shape == "Cube":
            s = obj_def["size"] * 1000
            return f"{s:.0f}mm³"
        elif shape == "Rectangle":
            w = obj_def["width"]  * 1000
            l = obj_def["length"] * 1000
            h = obj_def["height"] * 1000
            return f"{w:.0f}×{l:.0f}×{h:.0f}mm"
        elif shape in ("Cylinder", "Disc"):
            d = obj_def["radius"] * 2000
            h = obj_def["height"] * 1000
            return f"⌀{d:.0f}×{h:.0f}mm"
        elif shape == "Sphere":
            d = obj_def["radius"] * 2000
            return f"⌀{d:.0f}mm"
        return "?"

    # ══════════════════════════════════════════════════════════════════
    # OBJECTS — SPAWN
    # ══════════════════════════════════════════════════════════════════

    def _spawn_objects(
        self,
        table_center: tuple,
        table_info:   dict,
    ) -> list:
        root = f"{self._trial_root}/Objects"
        UsdGeom.Xform.Define(self.stage, root)

        # ── Generate N object definitions ───────────────────────────
        # During first deformable-object commissioning we force a single
        # catalogue object.  This removes clutter/randomness while we debug
        # scale, table contact, and stability of the imported USD.
        if (
            self.config.get("soft_object_catalog_enabled", False)
            and self.config.get("soft_object_commissioning_mode", False)
            and self.config.get("soft_object_override_num_objects") is not None
        ):
            n = int(self.config.get("soft_object_override_num_objects", 1))
            print(f"  [Objects] Soft commissioning mode: forcing n={n}")
        else:
            n_min, n_max = self.config["num_objects_range"]
            n = random.randint(n_min, n_max)

        used_labels = set()
        obj_defs = []
        for i in range(n):
            obj_def = self._generate_random_object(i, used_labels)
            obj_defs.append(obj_def)

        # ── Table geometry ──────────────────────────────────────────
        th           = table_info["table_size"][2]
        margin       = self.config["object_margin"]
        tcx, tcy, _  = table_center
        approach_rad = table_info["approach_rad"]
        facing_edge  = table_info["facing_edge"]
        depth_edge   = table_info["depth_edge"]

        ax = math.cos(approach_rad)
        ay = math.sin(approach_rad)
        px = -math.sin(approach_rad)
        py = math.cos(approach_rad)

        # ── Reach zone ──────────────────────────────────────────────
        ur5e_pos        = self._get_ur5e_base_world_pos()
        arm_reach       = self.config.get("arm_max_reach", 0.80)
        arm_min_reach   = self.config.get("arm_min_reach", 0.40)
        reach_safety    = self.config.get("reach_safety_margin", 0.05)
        effective_reach = arm_reach - reach_safety

        approach_min       = -depth_edge / 2 + margin
        approach_max_table =  depth_edge / 2 - margin
        perp_half          =  facing_edge / 2 - margin

        # Rigid primitive objects are usually spawned by their centre, so a
        # tiny positive clearance is useful.  Imported deformable USD assets
        # are different: the green foam asset is authored with its ROOT at
        # the bottom-centre, so its root must be placed directly at the
        # physical table surface (optionally with a tiny negative penetration
        # to compensate deformable contact-offset visual hovering).
        table_surface_z = th
        primitive_surface_z = th + 0.002
        max_reach  = effective_reach
        min_reach  = arm_min_reach + reach_safety

        print(
            f"  [Objects] Effective reach: {effective_reach:.3f}m "
            f"(arm={arm_reach:.2f} - safety={reach_safety:.2f})"
        )
        print(f"  [Objects] Min reach: {min_reach:.3f}m")

        placed: list[tuple[float, float, float]] = []
        result = []

        for i, obj_def in enumerate(obj_defs):
            shape    = obj_def["shape"]
            mass     = obj_def["mass"]
            material = obj_def["material"]
            color    = obj_def["color"]

            # ── Create unique physics material prim ─────────────────
            mat_path = (
                f"{self._trial_root}/PhysicsMaterials"
                f"/{obj_def['name']}"
            )
            self._create_object_material(mat_path, material)

            # ── Find valid placement position ───────────────────────
            # For the first imported deformable asset we use a deterministic
            # table-local pose. Random placement comes back after the asset is
            # scaled, table-aligned, and stable.
            fixed_local = obj_def.get("spawn_local_xy_m")
            if fixed_local is None and self.config.get("soft_object_commissioning_mode", False):
                fixed_local = self.config.get("soft_object_spawn_local_xy_m")

            new_radius = self._get_footprint_radius(obj_def)
            padding    = self.config.get("object_spacing_padding", 0.02)  # extra air gap

            if fixed_local is not None:
                la = float(fixed_local[0])
                lp = float(fixed_local[1])
                jitter = float(self.config.get("soft_object_spawn_jitter_m", 0.0) or 0.0)
                if jitter > 0.0:
                    la += random.uniform(-jitter, jitter)
                    lp += random.uniform(-jitter, jitter)

                # Clamp to table surface bounds with margin.  This prevents a
                # bad config from placing the soft object partly off the table.
                la = max(approach_min + new_radius, min(approach_max_table - new_radius, la))
                lp = max(-perp_half + new_radius, min(perp_half - new_radius, lp))

                wx = tcx + la * ax + lp * px
                wy = tcy + la * ay + lp * py
                obj_dist = math.sqrt((wx - ur5e_pos[0]) ** 2 + (wy - ur5e_pos[1]) ** 2)
                print(
                    f"    [SoftSpawn] deterministic local=({la:.3f},{lp:.3f}) "
                    f"world=({wx:.3f},{wy:.3f}) reach={obj_dist:.3f}m radius={new_radius:.3f}m"
                )
                if obj_dist > max_reach or obj_dist < min_reach:
                    print(
                        f"    ⚠ Deterministic soft spawn is outside reach "
                        f"[{min_reach:.3f}, {max_reach:.3f}]m; using random fallback"
                    )
                    fixed_local = None

            if fixed_local is None:
                for attempt in range(80):
                    la = random.uniform(approach_min, approach_max_table)
                    lp = random.uniform(-perp_half, perp_half)

                    too_close = False
                    for pa, pp, pr in placed:
                        required_dist = new_radius + pr + padding
                        if math.hypot(la - pa, lp - pp) < required_dist:
                            too_close = True
                            break

                    if too_close:
                        continue

                    wx = tcx + la * ax + lp * px
                    wy = tcy + la * ay + lp * py

                    obj_dist = math.sqrt(
                        (wx - ur5e_pos[0]) ** 2 +
                        (wy - ur5e_pos[1]) ** 2
                    )

                    if obj_dist > max_reach or obj_dist < min_reach:
                        continue

                    break
                else:
                    print(
                        f"    ⚠ Could not place {obj_def['label']} "
                        f"within reach")
                    continue

            placed.append((la, lp, new_radius))

            # ── Z positioning ───────────────────────────────────────
            half_h    = self._get_half_height(obj_def)
            asset_type = str(obj_def.get("asset_type", "primitive_proxy"))
            is_usd_asset = asset_type in ("usd_reference", "deformable_usd") and bool(obj_def.get("resolved_asset_path"))
            bottom_center_asset = (
                is_usd_asset
                and (
                    str(obj_def.get("spawn_pose_mode", "")).lower() == "bottom_on_table"
                    or str(obj_def.get("asset_origin_z", "")).lower() == "bottom_center"
                )
            )

            if bottom_center_asset:
                soft_clearance = float(self.config.get("soft_asset_table_clearance_m", 0.0) or 0.0)
                soft_penetration = float(self.config.get("soft_asset_table_penetration_m", 0.0) or 0.0)

                # Root is bottom-centre: place ROOT at table surface, not at
                # table + height/2.  The object centre used by grasp planning
                # is still table + height/2.
                spawn_root_z = table_surface_z + soft_clearance - soft_penetration
                wz = spawn_root_z + half_h
            else:
                spawn_root_z = primitive_surface_z + half_h
                wz = spawn_root_z
            print(
                f"    [SoftSpawnZ] label={obj_def.get('label')} "
                f"asset_type={obj_def.get('asset_type')} "
                f"bottom_center_asset={bottom_center_asset} "
                f"table_surface_z={table_surface_z:.4f} "
                f"half_h={half_h:.4f} "
                f"spawn_root_z={spawn_root_z:.4f} "
                f"world_center_z={wz:.4f} "
                f"penetration={float(self.config.get('soft_asset_table_penetration_m', 0.0) or 0.0):.4f}"
            )
            prim_path = f"{root}/{obj_def['name']}"

            # ── Random yaw rotation ─────────────────────────────────
            # Sphere: no rotation needed (fully symmetric)
            # Cylinder/Disc: no visible rotation (rotationally symmetric)
            # Cube: 90° increments + slight randomness (looks natural)
            # Rectangle: full random rotation (gripper reads prim yaw
            #            and always grasps across the short side)
            if obj_def.get("spawn_yaw_deg_fixed") is not None:
                yaw_deg = float(obj_def.get("spawn_yaw_deg_fixed"))
            elif shape == "Sphere":
                yaw_deg = 0.0
            elif shape in ("Cylinder", "Disc"):
                yaw_deg = 0.0
            elif shape == "Cube":
                yaw_deg = random.choice([0.0, 90.0, 180.0, 270.0])
                yaw_deg += random.uniform(-5.0, 5.0)
            elif shape == "Rectangle":
                yaw_deg = random.uniform(0.0, 360.0)
            else:
                yaw_deg = 0.0

            obj_def["spawn_yaw_deg"] = round(yaw_deg, 1)

            # ── Spawn geometry ──────────────────────────────────────
            spawned_from_asset = False
            asset_type = str(obj_def.get("asset_type", "primitive_proxy"))
            asset_path = obj_def.get("resolved_asset_path") or ""
            if asset_type in ("usd_reference", "deformable_usd") and asset_path:
                if os.path.exists(asset_path):
                    # For the commissioned green foam USD, the asset root is
                    # authored at the bottom centre.  Therefore we place the
                    # referenced root directly at spawn_root_z and DO NOT run
                    # bbox realignment.  BBox realignment can be unreliable for
                    # nested/defaultPrim referenced deformables and can leave the
                    # record pose inconsistent with the actual asset pose.
                    align_bottom = None
                    asset_position = (wx, wy, spawn_root_z if bottom_center_asset else wz)
                    if (not bottom_center_asset) and self.config.get("soft_asset_align_bottom_to_table", True):
                        align_bottom = (
                            table_surface_z
                            + float(self.config.get("soft_asset_table_clearance_m", 0.0))
                            - float(self.config.get("soft_asset_table_penetration_m", 0.0))
                        )

                    spawned_from_asset = self._make_usd_reference_object(
                        path=prim_path,
                        asset_path=asset_path,
                        position=asset_position,
                        scale=obj_def.get("asset_scale", 1.0),
                        rotation_z_deg=yaw_deg,
                        align_bottom_z=align_bottom,
                    )
                elif not self.config.get("soft_asset_missing_fallback_to_proxy", True):
                    print(f"    ❌ Missing soft asset: {asset_path}")
                    continue
                else:
                    print(
                        f"    ⚠ Missing soft asset for {obj_def['label']}; "
                        "using primitive proxy fallback"
                    )

            if spawned_from_asset:
                pass
            elif shape == "Cube":
                s = obj_def["size"]
                self._make_box(
                    path=prim_path, size=(s, s, s),
                    position=(wx, wy, wz), color=color,
                    is_static=False, mass=mass,
                    physics_mat_path=mat_path,
                    rotation_z_deg=yaw_deg,
                )

            elif shape == "Rectangle":
                sx = obj_def["width"]
                sy = obj_def["length"]
                sz = obj_def["height"]
                self._make_box(
                    path=prim_path, size=(sx, sy, sz),
                    position=(wx, wy, wz), color=color,
                    is_static=False, mass=mass,
                    physics_mat_path=mat_path,
                    rotation_z_deg=yaw_deg,
                )

            elif shape in ("Cylinder", "Disc"):
                self._make_cylinder(
                    path=prim_path,
                    radius=obj_def["radius"],
                    height=obj_def["height"],
                    position=(wx, wy, wz), color=color,
                    is_static=False, mass=mass,
                    physics_mat_path=mat_path,
                )

            elif shape == "Sphere":
                self._make_sphere(
                    path=prim_path,
                    radius=obj_def["radius"],
                    position=(wx, wy, wz), color=color,
                    is_static=False, mass=mass,
                    physics_mat_path=mat_path,
                )

            else:
                print(f"    ❌ Unknown shape '{shape}' — skipping")
                continue

            # ── Build object record ─────────────────────────────────
            obj_record = {
                "name":             obj_def["name"],
                "label":            obj_def["label"],
                "shape":            shape,
                "prim_path":        prim_path,
                # world_pos is the grasp-relevant object CENTRE, not
                # necessarily the USD root.  For bottom-centre deformable
                # assets the root is on/near the table, while the centre is
                # table + height/2.
                "world_pos":        (wx, wy, wz),
                "spawn_root_pos":   (wx, wy, spawn_root_z),
                "table_surface_z":  table_surface_z,
                "bottom_center_asset": bool(bottom_center_asset),
                "mass":             mass,
                "material_name":    obj_def["material_name"],
                "static_friction":  obj_def["static_friction"],
                "dynamic_friction": obj_def["dynamic_friction"],
                "color":            color,
                "color_name":       obj_def["color_name"],
                "size_category":    obj_def["size_category"],
                "grip_dim_mm":      obj_def["grip_dim_mm"],
                "spawn_yaw_deg":    yaw_deg,
                "is_target":        False,
                "horiz_dist":       obj_dist,
            }

            # Pass through all geometry and semantic/action-selection keys
            for key in (
                "size", "width", "length", "height", "radius", "size_xyz",
                "soft_object", "compliance", "deformable", "fragile",
                "fragility", "thin_object", "density_kg_m3",
                "youngs_modulus_pa", "poissons_ratio",
                "asset_type", "asset_path", "resolved_asset_path", "asset_scale",
                "preferred_force_n", "max_force_n",
                "gripper_hold_extra_close_m", "micro_lift_speed_scale",
                "close_speed_scale",
                "close_expected_grip_dim_mm", "close_contact_shell_extra_m",
                "adaptive_effort_control_enabled", "adaptive_effort_target_sim",
                "adaptive_effort_max_relax_open_m", "adaptive_effort_max_extra_close_m",
                "soft_pose_source", "grasp_height_mode", "grasp_center_height_fraction",
                "grasp_z_offset_m", "min_fingertip_clearance_m",
                "allow_lower_than_global_fingertip_clearance",
                "sphere_allow_probationary_partial_micro_lift",
                "sphere_partial_micro_lift_min_dz_m",
                "sphere_partial_micro_lift_min_following_ratio",
                "sphere_partial_micro_lift_max_drift_m",
                "sphere_micro_lift_rescue_delta_m",
                "sphere_micro_lift_rescue_settle_seconds",
                "sphere_micro_lift_rescue_force_n",
                "gripper_use_expected_contact_floor", "gripper_capture_floor_note",
                "sphere_micro_lift_preload_enabled",
                "sphere_micro_lift_preload_delta_m",
                "sphere_micro_lift_preload_settle_seconds",
                "sphere_micro_lift_preload_force_n",
                "target_stability_max_drift_m",
                "target_stability_max_xy_drift_m",
                "target_stability_retry_wait_seconds",
                "grasp_behavior_profile", "transport_effort_policy",
                "transport_abort_on_effort_over_max",
                "place_transport_admittance_max_effort_sim",
                "adaptive_safety_effort_enabled",
                "use_target_nominal_for_safety",
                "carry_bottom_clearance_m", "full_lift_bottom_clearance_m",
                "target_nominal_width_m", "target_nominal_depth_m",
                "target_nominal_height_m",
                "strategy_hint", "validation_profile",
                "fallback_primitive",
                "place_zone_key", "pickup_zone_key",
                "phase7_object_role", "batch_order",
            ):
                if key in obj_def and obj_def[key] is not None:
                    obj_record[key] = obj_def[key]

            self._spawned_objects.append((prim_path, obj_record))
            result.append(obj_record)

        # ── Summary log ─────────────────────────────────────────────
        print(f"  [Objects] Spawned {len(result)} on table")
        for r in result:
            dims_str  = self._get_object_dims_str(r)

            # Show rotation only for shapes where it matters
            if r["shape"] in ("Cube", "Rectangle"):
                rot_str = f"rot={r['spawn_yaw_deg']:>5.1f}°"
            else:
                rot_str = "          "

            print(
                f"{r['shape']:10s}  {dims_str:10s}  "
                f"{rot_str}  "
                f"mat={r['material_name']:10s}  "
                f"mass={r['mass']:.3f}kg  "
                f"grip={r['grip_dim_mm']:.0f}mm  "
                f"({r['world_pos'][0]:.3f}, "
                f"{r['world_pos'][1]:.3f}, "
                f"{r['world_pos'][2]:.3f})  "
                f"horiz={r['horiz_dist']:.3f}m "
            )

        return result
    
    def _get_footprint_radius(self, obj_def: dict) -> float:
        """
        Returns the radius of the object's 2D footprint on the table surface.
        Used for collision-aware spacing between spawned objects.
        """
        shape = obj_def["shape"]

        if shape == "Sphere":
            return obj_def["radius"]

        elif shape in ("Cylinder", "Disc"):
            return obj_def["radius"]

        elif shape == "Cube":
            # Half-diagonal of the square face
            half = obj_def["size"] / 2.0
            return math.sqrt(half ** 2 + half ** 2)

        elif shape == "Rectangle":
            # Half-diagonal of the rectangular face
            hw = obj_def["width"]  / 2.0
            hl = obj_def["length"] / 2.0
            return math.sqrt(hw ** 2 + hl ** 2)

        return 0.02  # Fallback

    # ══════════════════════════════════════════════════════════════════
    # MATERIALS
    # ══════════════════════════════════════════════════════════════════

    def _create_object_material(self, mat_path: str, material: dict):
        """..."""
        mats_root = f"{self._trial_root}/PhysicsMaterials"
        if not self.stage.GetPrimAtPath(Sdf.Path(mats_root)).IsValid():
            UsdGeom.Xform.Define(self.stage, mats_root)

        usd_mat = UsdShade.Material.Define(
            self.stage, Sdf.Path(mat_path))
        mat_api = UsdPhysics.MaterialAPI.Apply(usd_mat.GetPrim())
        mat_api.CreateStaticFrictionAttr().Set(
            float(material["static_friction"]))
        mat_api.CreateDynamicFrictionAttr().Set(
            float(material["dynamic_friction"]))
        mat_api.CreateRestitutionAttr().Set(
            float(material["restitution"]))


    def _apply_table_material(self, prim, material_def: dict):
        """
        Apply a PBR USD material to a table prim (top or legs).
        Creates the material prim once, reuses it on subsequent calls
        with the same material name.

        Args:
            prim:         USD prim to bind the material to.
            material_def: Material dict from table_materials list in table.yaml.
                        Keys: name, category, color, roughness, metallic, specular
        """
        mat_name  = material_def["name"]
        mat_path  = f"{self._trial_root}/Materials/Table_{mat_name}"
        mats_root = f"{self._trial_root}/Materials"

        # ── Create material prim once per unique material name ─────────
        mat_prim = self.stage.GetPrimAtPath(Sdf.Path(mat_path))
        if not mat_prim.IsValid():

            if not self.stage.GetPrimAtPath(Sdf.Path(mats_root)).IsValid():
                UsdGeom.Xform.Define(self.stage, mats_root)

            material    = UsdShade.Material.Define(
                self.stage, Sdf.Path(mat_path))
            shader_path = f"{mat_path}/PBRShader"
            shader      = UsdShade.Shader.Define(
                self.stage, Sdf.Path(shader_path))
            shader.CreateIdAttr("UsdPreviewSurface")

            r, g, b = material_def["color"]
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(r, g, b))
            shader.CreateInput(
                "roughness", Sdf.ValueTypeNames.Float
            ).Set(float(material_def["roughness"]))
            shader.CreateInput(
                "metallic", Sdf.ValueTypeNames.Float
            ).Set(float(material_def["metallic"]))
            shader.CreateInput(
                "specularLevel", Sdf.ValueTypeNames.Float
            ).Set(float(material_def.get("specular", 0.5)))

            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(
                UsdShade.ConnectableAPI(shader), "surface")

        # ── Bind material to prim ──────────────────────────────────────
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(
                self.stage.GetPrimAtPath(Sdf.Path(mat_path))))

        # ── Also set display color for viewport visibility ─────────────
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayColorAttr().Set(
                [Gf.Vec3f(*material_def["color"])])


    # ══════════════════════════════════════════════════════════════════
    # PRIMITIVE BUILDERS
    # ══════════════════════════════════════════════════════════════════

    def _make_box(
        self,
        path:             str,
        size:             tuple,
        position:         tuple,
        color:            tuple,
        is_static:        bool  = False,
        mass:             Optional[float] = None,
        physics_mat_path: Optional[str]   = None,
        rotation_z_deg:   float = 0.0,
    ):
        cube = UsdGeom.Cube.Define(self.stage, Sdf.Path(path))
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        if rotation_z_deg != 0.0:
            xf.AddRotateZOp().Set(rotation_z_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(*size))

        self._apply_display_color(cube.GetPrim(), color)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        if not is_static:
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            if mass is not None:
                mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
                mass_api.CreateMassAttr().Set(mass)

        if physics_mat_path:
            cube.GetPrim().CreateRelationship(
                "material:binding:physics"
            ).SetTargets([Sdf.Path(physics_mat_path)])

    def _make_cylinder(
        self,
        path:             str,
        radius:           float,
        height:           float,
        position:         tuple,
        color:            tuple,
        is_static:        bool  = False,
        mass:             Optional[float] = None,
        physics_mat_path: Optional[str]   = None,
    ):
        cyl = UsdGeom.Cylinder.Define(self.stage, Sdf.Path(path))
        cyl.GetRadiusAttr().Set(radius)
        cyl.GetHeightAttr().Set(height)
        cyl.GetAxisAttr().Set("Z")

        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))

        self._apply_display_color(cyl.GetPrim(), color)
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

        if not is_static:
            UsdPhysics.RigidBodyAPI.Apply(cyl.GetPrim())
            if mass is not None:
                mass_api = UsdPhysics.MassAPI.Apply(cyl.GetPrim())
                mass_api.CreateMassAttr().Set(mass)

        if physics_mat_path:
            cyl.GetPrim().CreateRelationship(
                "material:binding:physics"
            ).SetTargets([Sdf.Path(physics_mat_path)])

    def _make_sphere(
        self,
        path:             str,
        radius:           float,
        position:         tuple,
        color:            tuple,
        is_static:        bool  = False,
        mass:             Optional[float] = None,
        physics_mat_path: Optional[str]   = None,
    ):
        sph = UsdGeom.Sphere.Define(self.stage, Sdf.Path(path))
        sph.GetRadiusAttr().Set(radius)

        xf = UsdGeom.Xformable(sph.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))

        self._apply_display_color(sph.GetPrim(), color)
        UsdPhysics.CollisionAPI.Apply(sph.GetPrim())

        if not is_static:
            UsdPhysics.RigidBodyAPI.Apply(sph.GetPrim())
            if mass is not None:
                mass_api = UsdPhysics.MassAPI.Apply(sph.GetPrim())
                mass_api.CreateMassAttr().Set(mass)

        if physics_mat_path:
            sph.GetPrim().CreateRelationship(
                "material:binding:physics"
            ).SetTargets([Sdf.Path(physics_mat_path)])

    def _make_usd_reference_object(
        self,
        path: str,
        asset_path: str,
        position: tuple,
        scale=1.0,
        rotation_z_deg: float = 0.0,
        align_bottom_z=None,
    ) -> bool:
        """Reference an imported USD soft/deformable asset at a table pose.

        The referenced asset should already contain its own physics setup
        if it is truly deformable. This wrapper positions/scales it and,
        during commissioning, optionally uses the composed USD bounding box to
        align the lowest point just above the table. This is more robust than
        assuming the imported asset origin is at its centre or bottom.
        """
        prim = self.stage.DefinePrim(Sdf.Path(path), "Xform")
        if not prim.IsValid():
            return False

        prim.GetReferences().AddReference(asset_path)

        # Shrink the contact shell for the imported soft-object subtree.
        # Without this, default contact offsets can make the deformable
        # collision mesh rest ~20 mm above a 40 mm object/table contact.
        self._apply_physx_contact_offsets_recursive(
            prim,
            contact_offset_m=float(self.config.get("soft_asset_contact_offset_m", 0.001)),
            rest_offset_m=float(self.config.get("soft_asset_rest_offset_m", 0.0)),
            label="soft_asset_reference",
        )

        # B2B deformable/USD asset convention:
        # - The wrapper Xform pose is the asset root/spawn frame, often bottom-centre.
        # - The grasp-relevant object pose is the composed visual/collision extent.
        # Therefore downstream execution monitors should observe this object using
        # the wrapper bounding-box centre, not the wrapper translation.
        try:
            attr = prim.CreateAttribute("b2b:poseSource", Sdf.ValueTypeNames.String)
            attr.Set("bbox_center")
            prim.CreateAttribute("b2b:assetPath", Sdf.ValueTypeNames.String).Set(str(asset_path))
        except Exception as e:
            print(f"    ⚠ Could not annotate referenced asset pose source: {e}")

        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        translate_op = xf.AddTranslateOp()
        translate = Gf.Vec3d(*position)
        translate_op.Set(translate)
        if rotation_z_deg != 0.0:
            xf.AddRotateZOp().Set(rotation_z_deg)
        if isinstance(scale, (int, float)):
            scale_vec = Gf.Vec3f(float(scale), float(scale), float(scale))
            xf.AddScaleOp().Set(scale_vec)
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            scale_vec = Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2]))
            xf.AddScaleOp().Set(scale_vec)
        else:
            scale_vec = Gf.Vec3f(1.0, 1.0, 1.0)

        bbox_msg = ""
        if align_bottom_z is not None:
            try:
                bbox_cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    useExtentsHint=False,
                )
                world_bound = bbox_cache.ComputeWorldBound(prim)
                box = world_bound.ComputeAlignedBox()
                mn = box.GetMin()
                mx = box.GetMax()
                dz = float(align_bottom_z) - float(mn[2])
                translate = Gf.Vec3d(translate[0], translate[1], translate[2] + dz)
                translate_op.Set(translate)
                bbox_msg = (
                    f" bbox_before_min_z={float(mn[2]):.4f} "
                    f"bbox_before_max_z={float(mx[2]):.4f} align_bottom_z={float(align_bottom_z):.4f} dz_align={dz:.4f}"
                )
            except Exception as e:
                bbox_msg = f" bbox_align_failed={e}"

        print(
            f"    [SoftAsset] Referenced USD asset: {asset_path} "
            f"scale={tuple(round(float(v), 5) for v in scale_vec)}{bbox_msg}"
        )
        return True

    def _apply_physx_contact_offsets_recursive(
        self,
        prim,
        contact_offset_m: float,
        rest_offset_m: float,
        label: str = "",
    ):
        """Apply small PhysX contact/rest offsets to a prim subtree.

        Why this exists:
        Default PhysX contact offsets can be surprisingly large relative to a
        40 mm deformable object.  A ~20 mm inflated contact shell made the foam
        collision_mesh rest visibly above the table even when the render mesh
        was correctly placed.  For the B2B soft-object commissioning phase we
        explicitly shrink the contact shell on both the table collider and the
        imported soft object collision subtree.
        """
        if prim is None or not prim.IsValid():
            return

        try:
            from pxr import PhysxSchema
        except Exception as e:
            print(f"    ⚠ PhysxSchema unavailable; cannot set contact offsets for {label}: {e}")
            return

        contact_offset_m = float(contact_offset_m)
        rest_offset_m = float(rest_offset_m)
        applied = []
        errors = []

        def visit(p):
            # Apply only to geometry-ish prims plus wrapper; harmless failures
            # are collected and not fatal.
            type_name = p.GetTypeName()
            should_try = type_name in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "TetMesh", "Xform")
            if should_try:
                try:
                    api = PhysxSchema.PhysxCollisionAPI.Apply(p)
                    api.CreateContactOffsetAttr().Set(contact_offset_m)
                    api.CreateRestOffsetAttr().Set(rest_offset_m)
                    applied.append(str(p.GetPath()))
                except Exception as e:
                    errors.append((str(p.GetPath()), str(e)))
            for c in p.GetChildren():
                visit(c)

        visit(prim)

        if applied:
            print(
                f"    [ContactOffset] {label}: contact={contact_offset_m:.4f}m "
                f"rest={rest_offset_m:.4f}m applied_to={len(applied)} prims"
            )
            for path in applied[:8]:
                print(f"      - {path}")
            if len(applied) > 8:
                print(f"      ... {len(applied)-8} more")
        else:
            print(f"    ⚠ ContactOffset {label}: no prims accepted PhysxCollisionAPI")

        if errors and bool(self.config.get("print_contact_offset_errors", False)):
            for path, err in errors[:8]:
                print(f"      [ContactOffset error] {path}: {err}")

    def _apply_display_color(self, prim, color: tuple):
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])


            