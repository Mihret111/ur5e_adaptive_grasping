"""
arm_controller.py
─────────────────
UR5E arm controller with Lula IK, collision avoidance,
and height-aware grasping for Isaac Sim.

Owns:
  - IK solving (Lula / RMPflow / calibration fallback)
  - Joint trajectory execution (smoothstep interpolation)
  - Collision checking (table + cabinet + arm body)
  - Grasp height + force planning
  - Pick + place waypoint computation
"""

import os
import glob
import math
import random
import asyncio
import time
import numpy as np

import omni.usd
import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf


# ═══════════════════════════════════════════════════════════════════
# Isaac Sim IK imports
# ═══════════════════════════════════════════════════════════════════

_HAS_LULA = False
_HAS_CONFIG_LOADER = False
_RmpFlow = None

try:
    from omni.isaac.motion_generation import LulaKinematicsSolver
    _HAS_LULA = True
except ImportError:
    pass

try:
    from omni.isaac.motion_generation import interface_config_loader
    _HAS_CONFIG_LOADER = True
except ImportError:
    try:
        from isaacsim.robot.motion_generation import interface_config_loader
        _HAS_CONFIG_LOADER = True
    except ImportError:
        pass

try:
    from omni.isaac.motion_generation import RmpFlow as _RmpFlow
except ImportError:
    try:
        from omni.isaac.motion_generation.lula import RmpFlow as _RmpFlow
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════
# LULA CONFIG DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def _discover_lula_configs(verbose: bool = False):
    """Find UR5e Lula kinematics config files via config_loader or file search."""
    result = None

    # ── Method 1: Isaac Sim config loader ─────────────────────
    if _HAS_CONFIG_LOADER:
        for name in ("UR5e", "UR5", "ur5e", "ur5", "UR10e", "UR10"):
            try:
                cfg = (
                    interface_config_loader
                    .load_supported_lula_kinematics_solver_config(name)
                )
                urdf = cfg.get("urdf_path", cfg.get("urdf"))
                desc = cfg.get(
                    "robot_description_path",
                    cfg.get("robot_description"),
                )
                if urdf and desc:
                    result = {
                        "urdf_path":              urdf,
                        "robot_description_path": desc,
                        "rmpflow_config_path":    None,
                        "source":                 f"config_loader({name})",
                    }
                    break
            except Exception:
                pass

        if result is not None:
            for name in ("UR5e", "UR5", "ur5e", "UR10e"):
                try:
                    rmp_cfg = (
                        interface_config_loader
                        .load_supported_motion_policy_config(name, "RMPflow")
                    )
                    rmp_path = (
                        rmp_cfg.get("rmpflow_config_path")
                        or rmp_cfg.get("motion_policy_config_path")
                    )
                    if rmp_path:
                        result["rmpflow_config_path"] = rmp_path
                    break
                except Exception:
                    pass

    # ── Method 2: File system search ──────────────────────────
    if result is None and verbose:
        print("  [IK] Config loader failed, searching extensions...")
        search_dirs = set()
        try:
            import omni.isaac.motion_generation as mg
            mg_dir = os.path.dirname(os.path.abspath(mg.__file__))
            for depth in range(6):
                candidate = mg_dir
                for _ in range(depth):
                    candidate = os.path.dirname(candidate)
                data = os.path.join(candidate, "data")
                if os.path.isdir(data):
                    search_dirs.add(data)
        except Exception:
            pass

        home = os.path.expanduser("~")
        for extra in [
            os.path.join(home, "isaacsim"),
            os.path.join(home, "isaacsim", "exts"),
            os.path.join(home, "isaacsim", "extsDeprecated"),
        ]:
            if os.path.isdir(extra):
                search_dirs.add(extra)

        for sd in search_dirs:
            for pat in [
                "**/ur5e*robot_descriptor*.yaml",
                "**/ur5*robot_descriptor*.yaml",
                "**/universal_robots/**/robot_descriptor*.yaml",
            ]:
                for m in glob.glob(os.path.join(sd, pat), recursive=True):
                    d = os.path.dirname(m)
                    for up in [".", "..", "../.."]:
                        for u in glob.glob(os.path.join(d, up, "*.urdf")):
                            if "ur5" in u.lower():
                                result = {
                                    "urdf_path":              os.path.abspath(u),
                                    "robot_description_path": os.path.abspath(m),
                                    "rmpflow_config_path":    None,
                                    "source":                 "file_search",
                                }
                                break
                        if result:
                            break
                    if result:
                        break
                if result:
                    break
            if result:
                break

    return result


# ═══════════════════════════════════════════════════════════════════
# GRASP ORIENTATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def _quat_from_axis_angle(axis, angle_rad):
    """Quaternion [w,x,y,z] from axis-angle."""
    axis = np.array(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_multiply(q1, q2):
    """Hamilton product q1 * q2, both [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def compute_grasp_orientation(approach_angle_rad: float) -> np.ndarray:
    """Return tool0 quaternion [w,x,y,z] for a downward grasp at given yaw."""
    # 180° rotation about X → tool pointing straight down
    q_down = np.array([0.0, 1.0, 0.0, 0.0])
    # Yaw rotation about Z
    q_yaw = _quat_from_axis_angle([0, 0, 1], approach_angle_rad)
    q_final = _quat_multiply(q_yaw, q_down)
    q_final /= np.linalg.norm(q_final)
    return q_final


# ═══════════════════════════════════════════════════════════════════
# UR5E CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class UR5EController:
    """
    Joint-level arm controller for UR5E on MiR base in Isaac Sim.

    Public API used by PickAndPlaceExecutor:
        set_table_info(table_info)
        compute_pick_joints(obj_pos, pan_deg, table_height, object_metadata, prim_path)
            → {"joints": {wp: [6 floats]}, "flange_targets": {wp: [3 floats]}, ...}
        move_to(target_deg, duration, steps)   [async]
        get_flange_world_pos() → [x, y, z]
    """

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    JOINT_LIMITS_DEG = [
        (-360, 360), (-360, 360), (-360, 360),
        (-360, 360), (-360, 360), (-360, 360),
    ]

    # Empirical offset between "pan to object" angle and shoulder_pan joint
    PAN_OFFSET_DEG = 13.5

    # Calibration lookup: (shoulder_lift, elbow, reach_m, flange_z_m)
    # Used when Lula IK is unavailable
    CALIB_TABLE = [
        (-40.0, 50.0, 0.824, 1.107),
        (-30.0, 60.0, 0.815, 0.923),
        (-20.0, 70.0, 0.764, 0.769),
        (-10.0, 80.0, 0.668, 0.630),
        ( -5.0, 85.0, 0.601, 0.569),
        (  0.0, 89.0, 0.551, 0.532),
        ( 10.0, 95.0, 0.440, 0.470),
    ]

    # ──────────────────────────────────────────────────────────
    # __init__
    # ──────────────────────────────────────────────────────────

    def __init__(self, config: dict):
        self.config = config
        self.stage  = omni.usd.get_context().get_stage()

        # ── USD prim paths ────────────────────────────────────
        robot_paths   = config.get("paths", {}).get("robot", {})
        gripper_paths = config.get("paths", {}).get("gripper", {})

        self.joints_base = robot_paths.get(
            "ur5e_joints_base",
            "/mir/base_link_cabinet/cabinet/ur_mount/ur5e_physics/joints",
        )
        # 
        self.robot_model_root = robot_paths.get(
            "ur5e_model_root",
            self.joints_base.rsplit("/joints", 1)[0],
        )
        self.flange_path = robot_paths.get(
            "flange_prim",
            "/mir/base_link_cabinet/cabinet/ur_mount/"
            "ur5e_physics/wrist_3_link/flange",
        )
        self.base_link_path = robot_paths.get(
            "ur5e_base_link",
            "/mir/base_link_cabinet/cabinet/ur_mount/"
            "ur5e_physics/base_link",
        )
        self._gripper_base_path = gripper_paths.get(
            "base_link", "/onrobot_2fg7/base_link",
        )

        # ── IK engine state ───────────────────────────────────
        self._lula_solver = None
        self._rmpflow     = None
        self._ik_mode     = "calibration"
        self._ee_frame    = config.get("lula_end_effector_frame", "tool0")

        # ── Verbosity ─────────────────────────────────────────
        self._verbose = config.get("arm_verbose", False)

        # ── Grip geometry ─────────────────────────────────────
        self._grip_mode = config.get("grip_mode", "outwards")
        grip_cfg = config.get(f"{self._grip_mode}_grip", {})

        if self._grip_mode == "outwards":
            self._flange_to_finger_base  = grip_cfg.get(
                "flange_to_finger_base", 0.1225)
            self._flange_to_fingertips   = grip_cfg.get(
                "flange_to_fingertips", 0.1625)
            self._flange_to_grasp_centre = grip_cfg.get(
                "gripper_flange_to_fingertip", 0.1425)
            self._grip_range_min         = grip_cfg.get(
                "grip_range_min", 0.035)
            self._grip_range_max         = grip_cfg.get(
                "grip_range_max", 0.073)
        else:
            self._flange_to_finger_base  = grip_cfg.get(
                "flange_to_finger_base", 0.1175)
            self._flange_to_fingertips   = grip_cfg.get(
                "flange_to_fingertips", 0.1625)
            self._flange_to_grasp_centre = grip_cfg.get(
                "gripper_flange_to_fingertip", 0.14)
            self._grip_range_min         = grip_cfg.get(
                "grip_range_min", 0.045)
            self._grip_range_max         = grip_cfg.get(
                "grip_range_max", 0.083)

        self._grasp_z_offset      = config.get("grasp_z_offset", 0.0)
        self._finger_grasp_height = (
            self._flange_to_fingertips - self._flange_to_finger_base
        )

        # Calibrated from the pre-close diagnostic experiment. In the current mounting convention,
        # flange-local +X points from the flange toward the active finger capture region. 
        # TODO: configurable for sim-to-real calibration
        self._finger_capture_axis_name = config.get(
            "finger_capture_axis_local", "+X"
        )
        if self._finger_capture_axis_name not in {
            "+X", "-X", "+Y", "-Y", "+Z", "-Z"
        }:
            raise ValueError(
                "finger_capture_axis_local must be one of "
                "+X, -X, +Y, -Y, +Z, -Z; got "
                f"{self._finger_capture_axis_name!r}"
            )

        # TODO(tunable param) ── Force + grasp tuning ──────────────────────────────
        self._max_insertion_depth = config.get("max_insertion_depth", 0.040)
        self._grasp_clearance_mm = config.get("grasp_clearance_mm", 2.0)
        self._force_safety       = config.get("force_safety_factor", 2.5)
        self._min_grip_force     = config.get("min_grip_force", 20.0)
        self._max_grip_force     = config.get("max_grip_force", 140.0)

        # ── Clearance heights (from robot.yaml §4) ───────────
        self._pre_grasp_above_table = config.get(
            "pre_grasp_clearance_above_table", 0.15)
        self._lift_above_table = config.get(
            "lift_clearance_above_table", 0.20)
        self._settle_time = config.get(
            "post_move_settle_time", 0.8)
        self._finger_top_margin = config.get(
            "finger_grasp_top_margin", 0.005)

        # ── Collision avoidance (from robot.yaml §5) ─────────
        self._table_edge_safety = config.get(
            "table_edge_safety_margin", 0.10)
        self._cabinet_safety = config.get(
            "cabinet_safety_margin", 0.10)
        self._safe_travel_height = config.get(
            "safe_travel_height_above_table", 0.35)
        self._collision_check_margin = config.get(
            "collision_check_margin", 0.04)
        self.ARM_COLLISION_LINKS = config.get(
            "arm_collision_links", [
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "wrist_3_link",
            ],
        )
        self.ARM_BODY_CLEARANCE = config.get("arm_body_clearance", 0.08)

        # ── Table info (set at runtime by executor) ───────────
        self._table_info = None

        # ── Initialize IK engine ──────────────────────────────
        self._init_ik()

    # ──────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._verbose:
            print(msg)

    # ══════════════════════════════════════════════════════════
    # IK INITIALIZATION
    # ══════════════════════════════════════════════════════════

    def _init_ik(self):
        """Set up Lula IK → RMPflow → calibration fallback chain."""
        print("\n  [UR5E] Initializing IK...")
        lula_cfg = None

        # ── 1. Check user-supplied paths from robot.yaml ─────
        u = self.config.get("ur5e_urdf_path")
        d = self.config.get("ur5e_robot_descriptor_path")
        if (u and d
                and os.path.isfile(str(u))
                and os.path.isfile(str(d))):
            lula_cfg = {
                "urdf_path":              u,
                "robot_description_path": d,
                "rmpflow_config_path":    self.config.get(
                    "ur5e_rmpflow_config_path"),
                "source": "user",
            }

        # ── 2. Auto-discover ─────────────────────────────────
        if lula_cfg is None:
            lula_cfg = _discover_lula_configs(verbose=self._verbose)

        if lula_cfg is None:
            self._ik_mode = "calibration"
            print("  [UR5E] IK: calibration (no Lula config found)")
            return

        # ── 3. Lula kinematics solver ────────────────────────
        if _HAS_LULA:
            try:
                self._lula_solver = LulaKinematicsSolver(
                    robot_description_path=lula_cfg["robot_description_path"],
                    urdf_path=lula_cfg["urdf_path"],
                )
                self._ik_mode = "lula"

                # Validate end-effector frame
                for frame in [self._ee_frame, "tool0", "ee_link"]:
                    try:
                        self._lula_solver.compute_forward_kinematics(
                            frame, np.zeros(6))
                        self._ee_frame = frame
                        break
                    except Exception:
                        pass
                else:
                    self._lula_solver = None
                    self._ik_mode = "calibration"

            except Exception as e:
                self._log(f"  [UR5E] Lula init failed: {e}")
                self._ik_mode = "calibration"

        # ── 4. RMPflow (optional upgrade) ────────────────────
        if (self._lula_solver
                and lula_cfg.get("rmpflow_config_path")
                and _RmpFlow):
            try:
                self._rmpflow = _RmpFlow(
                    robot_description_path=lula_cfg["robot_description_path"],
                    urdf_path=lula_cfg["urdf_path"],
                    rmpflow_config_path=lula_cfg["rmpflow_config_path"],
                    end_effector_frame_name=self._ee_frame,
                    maximum_substep_size=0.00334,
                )
                self._ik_mode = "rmpflow"
            except Exception:
                pass

        print(
            f"  [UR5E] IK: {self._ik_mode}  "
            f"EE: {self._ee_frame}  "
            f"Grip: {self._grip_mode} "
            f"({self._finger_grasp_height * 1000:.0f}mm fingers)"
        )

    # ══════════════════════════════════════════════════════════
    # TABLE INFO
    # ══════════════════════════════════════════════════════════

    def set_table_info(self, table_info):
        """Store table geometry for collision avoidance.

        Expected keys:
            center:     (x, y, z) world position
            table_size: (length_x, width_y, height_z)
        """
        self._table_info = table_info

    # ══════════════════════════════════════════════════════════
    # JOINT ACCESS (USD Physics Drive API)
    # ══════════════════════════════════════════════════════════

    def set_joint_targets_deg(self, target_deg):
        """Write angular drive targets for all 6 joints (degrees)."""
        for name, deg in zip(self.JOINT_NAMES, target_deg):
            prim = self.stage.GetPrimAtPath(
                Sdf.Path(f"{self.joints_base}/{name}"))
            if prim.IsValid():
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    drive.GetTargetPositionAttr().Set(float(deg))

    def get_joint_targets_deg(self) -> np.ndarray:
        """Read current angular drive targets for all 6 joints (degrees)."""
        result = []
        for name in self.JOINT_NAMES:
            prim = self.stage.GetPrimAtPath(
                Sdf.Path(f"{self.joints_base}/{name}"))
            if prim.IsValid():
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    v = drive.GetTargetPositionAttr().Get()
                    result.append(v if v is not None else 0.0)
                else:
                    result.append(0.0)
            else:
                result.append(0.0)
        return np.array(result)

    # ══════════════════════════════════════════════════════════
    # WORLD POSITION QUERIES
    # ══════════════════════════════════════════════════════════

    def get_flange_world_pos(self) -> list:
        """Return [x, y, z] world position of the flange (tool0)."""
        prim = self.stage.GetPrimAtPath(Sdf.Path(self.flange_path))
        if not prim.IsValid():
            return [0.0, 0.0, 0.0]
        xf  = UsdGeom.Xformable(prim)
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p   = mtx.ExtractTranslation()
        return [p[0], p[1], p[2]]

    def _transform_flange_local_point_to_world(self, local_xyz) -> list:
        """Transform a point from flange-local coordinates into world coordinates.

        Diagnostic helper only.  At this stage we deliberately do not assume
        which local flange axis points toward the useful finger-capture region.
        """
        prim = self.stage.GetPrimAtPath(Sdf.Path(self.flange_path))
        if not prim.IsValid():
            raise RuntimeError(
                f"Cannot compute pre-close diagnostics: invalid flange prim "
                f"'{self.flange_path}'"
            )

        xf = UsdGeom.Xformable(prim)
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local = Gf.Vec3d(
            float(local_xyz[0]),
            float(local_xyz[1]),
            float(local_xyz[2]),
        )
        world = mtx.Transform(local)
        return [float(world[0]), float(world[1]), float(world[2])]

    @staticmethod
    def _euclidean_distance(a, b) -> float:
        return float(np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        ))

    def get_preclose_geometry_diagnostics(
        self,
        object_world_pos,
        object_metadata=None,
        planned_flange_world_pos=None,
    ) -> dict:
        """Return diagnostic evidence immediately before gripper closure.

        This method intentionally does *not* reject a grasp yet.  The final
        USD mounting convention must first be verified experimentally.

        It reports candidate useful-grasp-centre positions for all six local
        flange-axis directions (+/-X, +/-Y, +/-Z).  The closest candidate to
        the object gives us evidence about the real flange-to-finger axis in
        the professor-provided USD model.
        """
        obj = np.asarray(object_world_pos, dtype=np.float64)
        flange = np.asarray(self.get_flange_world_pos(), dtype=np.float64)

        planned_flange_error = None
        if planned_flange_world_pos is not None:
            planned = np.asarray(planned_flange_world_pos, dtype=np.float64)
            planned_flange_error = self._euclidean_distance(flange, planned)

        axis_vectors = {
            "+X": [1.0, 0.0, 0.0],
            "-X": [-1.0, 0.0, 0.0],
            "+Y": [0.0, 1.0, 0.0],
            "-Y": [0.0, -1.0, 0.0],
            "+Z": [0.0, 0.0, 1.0],
            "-Z": [0.0, 0.0, -1.0],
        }

        candidate_axes = {}
        centre_offset = float(self._flange_to_grasp_centre)
        base_offset = float(self._flange_to_finger_base)
        tip_offset = float(self._flange_to_fingertips)

        for axis_name, axis in axis_vectors.items():
            centre_local = [centre_offset * v for v in axis]
            base_local = [base_offset * v for v in axis]
            tip_local = [tip_offset * v for v in axis]

            centre_world = self._transform_flange_local_point_to_world(
                centre_local
            )
            finger_base_world = self._transform_flange_local_point_to_world(
                base_local
            )
            fingertips_world = self._transform_flange_local_point_to_world(
                tip_local
            )

            candidate_axes[axis_name] = {
                "grasp_centre_world_pos": centre_world,
                "finger_base_world_pos": finger_base_world,
                "fingertips_world_pos": fingertips_world,
                "grasp_centre_object_distance_m": self._euclidean_distance(
                    centre_world, obj
                ),
                "grasp_centre_object_delta_m": (
                    np.asarray(centre_world, dtype=np.float64) - obj
                ).tolist(),
            }

        best_axis = min(
            candidate_axes,
            key=lambda name: candidate_axes[name][
                "grasp_centre_object_distance_m"
            ],
        )

        gripper_base_pos = None
        gripper_base_mtx = self._get_gripper_base_transform()
        if gripper_base_mtx is not None:
            p = gripper_base_mtx.ExtractTranslation()
            gripper_base_pos = [float(p[0]), float(p[1]), float(p[2])]

        object_height = float(self._get_object_height(object_metadata))

        return {
            "mode": "diagnostic_only_no_rejection",
            "object_world_pos": obj.tolist(),
            "object_height_m": object_height,
            "actual_flange_world_pos": flange.tolist(),
            "planned_flange_world_pos": (
                None
                if planned_flange_world_pos is None
                else np.asarray(
                    planned_flange_world_pos, dtype=np.float64
                ).tolist()
            ),
            "planned_flange_tracking_error_m": planned_flange_error,
            "flange_object_delta_m": (flange - obj).tolist(),
            "flange_object_distance_m": self._euclidean_distance(
                flange, obj
            ),
            "gripper_base_world_pos": gripper_base_pos,
            "flange_to_grasp_centre_m": centre_offset,
            "flange_to_finger_base_m": base_offset,
            "flange_to_fingertips_m": tip_offset,
            "candidate_axes": candidate_axes,
            "closest_axis_candidate": best_axis,
            "closest_axis_distance_m": candidate_axes[best_axis][
                "grasp_centre_object_distance_m"
            ],
        }

    def evaluate_preclose_geometry_gate(self, diagnostics: dict) -> dict:
        """Evaluate whether closing the gripper is geometrically plausible.

        Uses the calibrated flange-local finger axis rather than the
        closest of six diagnostic hypotheses.  It checks three quantities:

          1. Cartesian flange tracking error;
          2. horizontal grasp-centre alignment with the object;
          3. vertical overlap between the object and the active finger span.

        It intentionally does not require the grasp centre to coincide with
        the object's 3-D geometric centre.  A valid grasp may just engage an upper
        portion of a tall object
        """
        axis_name = self._finger_capture_axis_name
        axis_info = diagnostics["candidate_axes"][axis_name]

        obj = np.asarray(
            diagnostics["object_world_pos"], dtype=np.float64
        )
        centre = np.asarray(
            axis_info["grasp_centre_world_pos"], dtype=np.float64
        )
        finger_base = np.asarray(
            axis_info["finger_base_world_pos"], dtype=np.float64
        )
        fingertips = np.asarray(
            axis_info["fingertips_world_pos"], dtype=np.float64
        )

        object_height = float(diagnostics["object_height_m"])
        object_low_z = float(obj[2] - object_height / 2.0)
        object_high_z = float(obj[2] + object_height / 2.0)

        capture_low_z = float(min(finger_base[2], fingertips[2]))
        capture_high_z = float(max(finger_base[2], fingertips[2]))

        vertical_overlap = max(
            0.0,
            min(object_high_z, capture_high_z)
            - max(object_low_z, capture_low_z),
        )

        grasp_centre_xy_error = float(np.linalg.norm(centre[:2] - obj[:2]))
        tracking_error = diagnostics.get("planned_flange_tracking_error_m")

        max_xy_error = float(
            self.config.get("max_preclose_grasp_centre_xy_error_m", 0.015)
        )
        min_overlap = float(
            self.config.get("min_preclose_vertical_overlap_m", 0.010)
        )
        max_tracking_error = float(
            self.config.get("max_preclose_flange_tracking_error_m", 0.010)
        )

        reasons = []

        if tracking_error is None:
            reasons.append("planned flange tracking error unavailable")
        elif float(tracking_error) > max_tracking_error:
            reasons.append(
                "flange tracking error too large: "
                f"{float(tracking_error):.4f} m > "
                f"{max_tracking_error:.4f} m"
            )

        if grasp_centre_xy_error > max_xy_error:
            reasons.append(
                "grasp-centre XY error too large: "
                f"{grasp_centre_xy_error:.4f} m > "
                f"{max_xy_error:.4f} m"
            )

        if vertical_overlap < min_overlap:
            reasons.append(
                "insufficient finger/object vertical overlap: "
                f"{vertical_overlap:.4f} m < {min_overlap:.4f} m"
            )

        require_axis_match = bool(
            self.config.get("require_preclose_closest_axis_match", False)
        )
        closest_axis = diagnostics.get("closest_axis_candidate")
        axis_match = closest_axis == axis_name
        if require_axis_match and not axis_match:
            reasons.append(
                "closest diagnostic axis differs from calibrated finger axis: "
                f"closest={closest_axis}, calibrated={axis_name}"
            )

        return {
            "mode": "calibrated_preclose_gate",
            "calibrated_finger_axis_local": axis_name,
            "closest_diagnostic_axis": closest_axis,
            "axis_match": axis_match,
            "grasp_centre_world_pos": centre.tolist(),
            "finger_base_world_pos": finger_base.tolist(),
            "fingertips_world_pos": fingertips.tolist(),
            "object_world_pos": obj.tolist(),
            "object_low_z": object_low_z,
            "object_high_z": object_high_z,
            "capture_low_z": capture_low_z,
            "capture_high_z": capture_high_z,
            "grasp_centre_xy_error_m": grasp_centre_xy_error,
            "vertical_overlap_m": vertical_overlap,
            "planned_flange_tracking_error_m": tracking_error,
            "max_preclose_grasp_centre_xy_error_m": max_xy_error,
            "min_preclose_vertical_overlap_m": min_overlap,
            "max_preclose_flange_tracking_error_m": max_tracking_error,
            "geometry_ok": len(reasons) == 0,
            "reasons": reasons,
        }

    def get_base_world_pos(self) -> list:
        """Return [x, y, z] world position of the UR5E base_link."""
        prim = self.stage.GetPrimAtPath(Sdf.Path(self.base_link_path))
        if not prim.IsValid():
            return [
                0.0, 0.0,
                self.config.get("ur5e_base_height", 0.8593),
            ]
        xf  = UsdGeom.Xformable(prim)
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p   = mtx.ExtractTranslation()
        return [p[0], p[1], p[2]]

    def _get_base_world_matrix(self):
        """Return full 4×4 local-to-world matrix for base_link, or None."""
        prim = self.stage.GetPrimAtPath(Sdf.Path(self.base_link_path))
        if not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        return xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    def _get_gripper_base_transform(self):
        """Return gripper base_link world transform, or None."""
        prim = self.stage.GetPrimAtPath(
            Sdf.Path(self._gripper_base_path))
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            return xform.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
        return None

    # def _get_link_world_pos(self, link_name: str):
    #     """Return [x, y, z] of a named arm link, or None."""
    #     path = f"{self.joints_base}/{link_name}"
    #     prim = self.stage.GetPrimAtPath(Sdf.Path(path))
    #     if not prim.IsValid():
    #         return None
    #     xf  = UsdGeom.Xformable(prim)
    #     mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    #     return [float(mtx[3][0]), float(mtx[3][1]), float(mtx[3][2])]
    
    def _get_link_world_pos(self, link_name: str):
        """Return [x, y, z] of a named arm link, or None."""
        path = f"{self.robot_model_root}/{link_name}"
        prim = self.stage.GetPrimAtPath(Sdf.Path(path))
       
        # self._log(f"  [Collision] Checking link: {path}")

        if not prim.IsValid():
            self._log(f"  [Collision] Link prim not found: {path}")
            return None

        xf = UsdGeom.Xformable(prim)
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p = mtx.ExtractTranslation()

        return [float(p[0]), float(p[1]), float(p[2])]
    # ══════════════════════════════════════════════════════════
    # COORDINATE TRANSFORMS
    # ══════════════════════════════════════════════════════════

    def _world_to_robot_base(self, world_pos) -> np.ndarray:
        """Transform a world position into the robot base_link frame."""
        mtx = self._get_base_world_matrix()
        if mtx is not None:
            base = np.array([mtx[3][0], mtx[3][1], mtx[3][2]])
            rot  = np.array([
                [mtx[0][0], mtx[0][1], mtx[0][2]],
                [mtx[1][0], mtx[1][1], mtx[1][2]],
                [mtx[2][0], mtx[2][1], mtx[2][2]],
            ])
            return rot.T @ (np.array(world_pos) - base)
        return np.array(world_pos) - np.array(self.get_base_world_pos())

    def _robot_base_to_world(self, local_pos) -> np.ndarray:
        """Transform a robot base_link position into world frame."""
        mtx = self._get_base_world_matrix()
        if mtx is not None:
            base = np.array([mtx[3][0], mtx[3][1], mtx[3][2]])
            rot  = np.array([
                [mtx[0][0], mtx[0][1], mtx[0][2]],
                [mtx[1][0], mtx[1][1], mtx[1][2]],
                [mtx[2][0], mtx[2][1], mtx[2][2]],
            ])
            return rot @ np.array(local_pos) + base
        return np.array(local_pos) + np.array(self.get_base_world_pos())

    # ══════════════════════════════════════════════════════════
    # COLLISION CHECKING
    # ══════════════════════════════════════════════════════════

    def _is_over_table(self, world_pos) -> bool:
        """True if (x, y) is within the table footprint (+ margin)."""
        if self._table_info is None:
            return False
        tc     = self._table_info.get("center", (0, 0, 0))
        ts     = self._table_info.get("table_size", (1.0, 0.6, 0.75))
        half_x = ts[0] / 2.0 + 0.05
        half_y = ts[1] / 2.0 + 0.05
        dx     = abs(world_pos[0] - tc[0])
        dy     = abs(world_pos[1] - tc[1])
        return dx < half_x and dy < half_y

    def _get_table_surface_z(self) -> float:
        """Return the table surface Z in world coordinates."""
        if self._table_info is None:
            return 0.75
        return self._table_info.get("table_size", (1, 0.6, 0.75))[2]

    def _safe_z_above_table(self) -> float:
        """Return the minimum safe flange Z for transit over the table."""
        if self._table_info is None:
            return 1.2
        th = self._get_table_surface_z()
        return th + self._safe_travel_height + self._flange_to_fingertips

    def _clamp_flange_z_above_table(
        self, flange_pos, label: str = "",
    ) -> np.ndarray:
        """Raise flange Z if it would collide with the table surface."""
        pos = np.array(flange_pos, dtype=np.float64)
        if not self._is_over_table(pos):
            return pos
        safe_z = self._safe_z_above_table()
        if pos[2] < safe_z:
            self._log(
                f"  [Clamp] {label}: Z {pos[2]:.4f} → {safe_z:.4f}")
            pos[2] = safe_z
        return pos

    def _is_near_cabinet(self, world_pos) -> bool:
        """True if the position is dangerously close to the MiR cabinet."""
        base        = self.get_base_world_pos()
        cab_front_x = base[0] - 0.15
        cab_left_y  = base[1] - 0.35
        cab_right_y = base[1] + 0.35
        cab_top_z   = base[2] + 0.10
        return (
            world_pos[0] < cab_front_x + self._cabinet_safety
            and cab_left_y - self._cabinet_safety
                < world_pos[1]
                < cab_right_y + self._cabinet_safety
            and world_pos[2] < cab_top_z + self._cabinet_safety
        )

    # ── Point / link collision helpers ────────────────────────

    def _get_link_endpoints(self, link_name_a: str, link_name_b: str):
        """Return (pos_a, pos_b) for two arm links."""
        return (
            self._get_link_world_pos(link_name_a),
            self._get_link_world_pos(link_name_b),
        )

    def _sample_points_along_link(
        self, pos_a, pos_b, n_samples: int = 5,
    ) -> list:
        """Return evenly-spaced points between two link positions."""
        if pos_a is None or pos_b is None:
            return []
        a = np.array(pos_a)
        b = np.array(pos_b)
        return [
            (a + (i / n_samples) * (b - a)).tolist()
            for i in range(n_samples + 1)
        ]

    def _check_point_table_collision(
        self, world_pos, table_z: float, margin: float,
    ) -> bool:
        """True if a single point is below table surface + margin."""
        return (
            world_pos[2] < table_z + margin
            and self._is_over_table(world_pos)
        )

    def _check_flange_table_collision(self) -> bool:
        """True if the flange is inside the table volume."""
        if self._table_info is None:
            return False
        flange   = self.get_flange_world_pos()
        table_z  = self._get_table_surface_z()
        min_safe = table_z + self._collision_check_margin
        return self._is_over_table(flange) and flange[2] < min_safe

    def _check_arm_body_table_collision(self) -> bool:
        """
        Check configured arm links for table penetration.

        Uses dense sampling for long links and point checks for
        remaining arm-link frame positions.
        """
        if self._table_info is None:
            return False

        table_z = self._get_table_surface_z()
        margin = self._collision_check_margin

        n_samples = self.config.get(
            "forearm_sample_count",
            5,
        )

        long_link_pairs = self.config.get(
            "long_link_pairs",
            [
                ["forearm_link", "wrist_1_link"],
                ["upper_arm_link", "forearm_link"],
            ],
        )

        debug = self.config.get("collision_debug", False)

        covered_links = set()

        # ── Dense sampling along long links ──────────────────
        for pair in long_link_pairs:
            if len(pair) != 2:
                continue

            link_a, link_b = pair

            covered_links.add(link_a)
            covered_links.add(link_b)

            pos_a, pos_b = self._get_link_endpoints(
                link_a,
                link_b,
            )

            if debug:
                self._log(
                    f"  [CollisionDebug] Segment {link_a} → {link_b}\n"
                    f"      {link_a}: {pos_a}\n"
                    f"      {link_b}: {pos_b}"
                )

            if pos_a is None or pos_b is None:
                pos = pos_a or pos_b

                if (
                    pos is not None
                    and self._check_point_table_collision(
                        pos,
                        table_z,
                        margin,
                    )
                ):
                    self._log(
                        f"  [Collision] ❌ Table collision near "
                        f"{link_a} → {link_b}: point={pos}"
                    )
                    return True

                continue

            sampled_points = self._sample_points_along_link(
                pos_a,
                pos_b,
                n_samples,
            )

            for pt in sampled_points:
                if self._check_point_table_collision(
                    pt,
                    table_z,
                    margin,
                ):
                    self._log(
                        f"  [Collision] ❌ Table collision on segment "
                        f"{link_a} → {link_b}: "
                        f"point={pt}, "
                        f"table_z={table_z:.3f}, "
                        f"margin={margin:.3f}"
                    )
                    return True

        # ── Point checks for links not already sampled ───────
        for link_name in self.ARM_COLLISION_LINKS:
            if link_name in covered_links:
                continue

            link_pos = self._get_link_world_pos(link_name)

            if debug:
                self._log(
                    f"  [CollisionDebug] Point link "
                    f"{link_name}: {link_pos}"
                )

            if link_pos is None:
                continue

            if self._check_point_table_collision(
                link_pos,
                table_z,
                margin,
            ):
                self._log(
                    f"  [Collision] ❌ Table collision at "
                    f"{link_name}: "
                    f"point={link_pos}, "
                    f"table_z={table_z:.3f}, "
                    f"margin={margin:.3f}"
                )
                return True

        return False

    def _check_full_collision(self) -> bool:
        """Check both flange and arm body for table collision."""
        return (
            self._check_flange_table_collision()
            or self._check_arm_body_table_collision()
        )

    # ══════════════════════════════════════════════════════════
    # OBJECT HEIGHT ANALYSIS
    # ══════════════════════════════════════════════════════════

    def _get_object_height(self, object_metadata) -> float:
        if object_metadata is None:
            return 0.035
        shape = object_metadata.get("shape", "Cube")
        if shape == "Cube":
            return object_metadata.get("size", 0.035)
        if shape == "Rectangle":
            return object_metadata.get("height", 0.025)
        if shape in ("Cylinder", "Disc"):
            return object_metadata.get("height", 0.050)
        if shape == "Sphere":
            return object_metadata.get("radius", 0.020) * 2.0
        return 0.035

    def _compute_grasp_height(
        self, obj_meta, table_height: float,
    ) -> dict:
        """
        Compute tool0 Z for external grasping.

        full_wrap:    object shorter than fingers → fingers reach table
        partial_wrap: object taller than fingers → fingers wrap top portion
        """
        obj_h = self._get_object_height(obj_meta)

        if obj_h <= self._finger_grasp_height:
            # Fingers can fully wrap the object
            grasp_target_z = table_height
            tool0_z        = grasp_target_z + self._flange_to_fingertips
            strategy       = "full_wrap"
        else:
            # Fingers wrap the top portion
            grasp_target_z = table_height + obj_h
            tool0_z        = grasp_target_z + self._flange_to_finger_base
            strategy       = "partial_wrap"

        tool0_z += self._grasp_z_offset

        self._log(
            f"  [Height] {strategy}  obj={obj_h * 1000:.0f}mm  "
            f"tool0_z={tool0_z:.4f}"
        )

        return {
            "tool0_z":         tool0_z,
            "strategy":        strategy,
            "object_height":   obj_h,
            "grasp_target_z":  grasp_target_z,
            "finger_top_z":    tool0_z - self._flange_to_finger_base,
            "finger_bottom_z": tool0_z - self._flange_to_fingertips,
            "grasp_ok":        True,
        }

    # ══════════════════════════════════════════════════════════
    # MOTION — SMOOTH JOINT INTERPOLATION
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Hermite smoothstep: zero velocity at t=0 and t=1."""
        return t * t * (3.0 - 2.0 * t)

    async def move_to(
        self,
        target_deg,
        duration:              float = 3.0,
        steps:                 int   = 150,
        check_table_collision: bool  = True,
        step_callback          = None,
    ) -> bool:
        """
        Smoothly interpolate all 6 joints to target_deg.

        Args:
            step_callback:
                Optional callable invoked once after every simulation update.
                Can be sync or async. Useful for camera capture / logging.

        Returns:
            False if a collision was detected and the move was aborted.
        """
        app      = omni.kit.app.get_app()
        current  = self.get_joint_targets_deg()
        target   = np.array(target_deg, dtype=float)
        last_safe = current.copy()

        for i in range(steps + 1):
            t      = self._smoothstep(i / steps)
            interp = current + t * (target - current)
            self.set_joint_targets_deg(list(interp))
            await app.next_update_async()

            if step_callback is not None:
                if asyncio.iscoroutinefunction(step_callback):
                    await step_callback()
                else:
                    step_callback()

            if check_table_collision and self._table_info is not None:
                if self._check_full_collision():
                    print(f"  [COLLISION] Halting at step {i}/{steps}")
                    self.set_joint_targets_deg(list(last_safe))
                    for _ in range(60):
                        await app.next_update_async()
                        if step_callback is not None:
                            if asyncio.iscoroutinefunction(step_callback):
                                await step_callback()
                            else:
                                step_callback()
                    return False

            last_safe = interp.copy()

        # Post-move settle
        settle_frames = max(1, int(self._settle_time * 60))
        for _ in range(settle_frames):
            await app.next_update_async()
            if step_callback is not None:
                if asyncio.iscoroutinefunction(step_callback):
                    await step_callback()
                else:
                    step_callback()

        return True

    async def move_via_safe_height(
        self,
        target_deg,
        duration: float = 3.0,
        steps:    int   = 150,
        step_callback   = None,
    ) -> bool:
        """
        If currently over the table below safe height, lift first,
        then move to target.
        """
        current_flange = self.get_flange_world_pos()
        current_joints = self.get_joint_targets_deg()
        safe_z         = self._safe_z_above_table()

        # Insert an intermediate lift before moving to the requested target
        #  if it thinks the flange is over the table and below safe height
        if (self._is_over_table(current_flange)
                and current_flange[2] < safe_z - 0.02):
            lifted_pos    = np.array(current_flange)
            lifted_pos[2] = safe_z
            lift_joints   = self._solve_ik_for_world_pos(
                lifted_pos, seed_deg=current_joints)
            if lift_joints is not None:
                ok = await self.move_to(
                    lift_joints,
                    duration=duration * 0.4,
                    steps=max(30, steps // 3),
                    check_table_collision=True,
                    step_callback=step_callback,
                )
                if not ok:
                    return False

        return await self.move_to(
            target_deg,
            duration=duration,
            steps=steps,
            check_table_collision=True,
            step_callback=step_callback,
        )

    def _solve_ik_for_world_pos(
        self, world_pos, orient=None, seed_deg=None,
    ):
        # """Convenience: world position → IK solve in robot base frame."""
        # if self._lula_solver is None:
        #     return None
        # local  = self._world_to_robot_base(world_pos)
        # orient = (
        #     orient if orient is not None
        #     else np.array([0.0, 1.0, 0.0, 0.0])
        # )
        # return self._solve_ik_retries(local, orient, seed_deg=seed_deg)
        """Convenience: world position → IK solve in robot base frame."""
        if self._lula_solver is None:
            return None

        local = self._world_to_robot_base(world_pos)

        orient = (
            orient if orient is not None
            else np.array([0.0, 1.0, 0.0, 0.0])
        )

        joints, meta = self._solve_ik_retries(
            local,
            orient,
            seed_deg=seed_deg,
        )

        if joints is None:
            self._log(f"  [IK] _solve_ik_for_world_pos failed: {meta}")
            return None

        return joints

    async def _emergency_retreat(self):
        """Last-resort return to home, ignoring collision checks."""
        print("  [EMERGENCY] Retreating to home")
        home = self.config.get(
            "arm_home_deg", [0.0, 90.0, 90.0, 0.0, 0.0, 0.0])
        await self.move_to(
            home, duration=5.0, steps=300,
            check_table_collision=False,
        )

    # ══════════════════════════════════════════════════════════
    # SEQUENCE EXECUTION (used by executor for structured moves)
    # ══════════════════════════════════════════════════════════

    async def execute_pick_sequence(
        self,
        pick_result,
        gripper_open_fn  = None,
        gripper_close_fn = None,
        settle_fn        = None,
    ) -> bool:
        """
        Execute a pre-computed pick sequence through all waypoints.

        Waypoints: safe_above → pre_grasp → grasp → lift →
                   safe_retreat → retract
        """
        joints   = pick_result["joints"]
        sequence = [
            ("safe_above",   3.0, 150, True,  None),  # (safe_above, 3 = duration in seconds, 150 = number of steps, True = use IK, None = no gripper action)
            ("pre_grasp",    3.0, 150, False, "open_gripper"),
            ("grasp",        4.0, 200, False, "close_gripper"),
            ("lift",         3.0, 150, False, None),  
            ("safe_retreat", 3.0, 150, False, None),
            ("retract",      3.0, 150, True,  None),
        ]
        for name, dur, steps, via_safe, action in sequence:
            if name not in joints:
                continue
            j = joints[name]
            if via_safe:
                ok = await self.move_via_safe_height(
                    j, duration=dur, steps=steps)
            else:
                ok = await self.move_to(
                    j, duration=dur, steps=steps,
                    check_table_collision=True)
            if not ok:
                print(f"  [Pick] ❌ COLLISION at '{name}'")
                await self._emergency_retreat()
                return False
            if action == "open_gripper" and gripper_open_fn:
                await gripper_open_fn()
            elif action == "close_gripper" and gripper_close_fn:
                await gripper_close_fn()
            if settle_fn:
                await settle_fn()
        return True

    async def execute_place_sequence(
        self,
        place_joints,
        gripper_open_fn = None,
        settle_fn       = None,
    ) -> bool:
        """Execute a pre-computed place sequence through all waypoints."""
        sequence = [
            ("safe_above",   3.0, 150, True,  None),
            ("pre_place",    3.0, 150, False, None),
            ("place",        4.0, 200, False, "open_gripper"),
            ("lift",         3.0, 150, False, None),
            ("safe_retreat", 3.0, 150, False, None),
            ("retract",      3.0, 150, True,  None),
        ]
        for name, dur, steps, via_safe, action in sequence:
            if name not in place_joints:
                continue
            j = place_joints[name]
            if via_safe:
                ok = await self.move_via_safe_height(
                    j, duration=dur, steps=steps)
            else:
                ok = await self.move_to(
                    j, duration=dur, steps=steps,
                    check_table_collision=True)
            if not ok:
                print(f"  [Place] ❌ COLLISION at '{name}'")
                await self._emergency_retreat()
                return False
            if action == "open_gripper" and gripper_open_fn:
                await gripper_open_fn()
            if settle_fn:
                await settle_fn()
        return True

    # ══════════════════════════════════════════════════════════
    # APPROACH DIRECTION
    # ══════════════════════════════════════════════════════════

    def _compute_approach_direction(self, object_world_pos):
        """
        Return (dir_x, dir_y, angle_rad) from robot base toward the object.
        """
        base    = self.get_base_world_pos()
        dx      = object_world_pos[0] - base[0]
        dy      = object_world_pos[1] - base[1]
        dist_xy = math.sqrt(dx ** 2 + dy ** 2)
        if dist_xy > 0.001:
            dir_x = dx / dist_xy
            dir_y = dy / dist_xy
        else:
            dir_x, dir_y = 1.0, 0.0
        return dir_x, dir_y, math.atan2(dy, dx)

    # ══════════════════════════════════════════════════════════
    # FLANGE TARGET COMPUTATION
    # ══════════════════════════════════════════════════════════

    def _compute_flange_targets(
        self,
        object_world_pos,
        table_height:    float,
        object_metadata = None,
    ) -> dict:

        obj  = np.array(object_world_pos, dtype=np.float64)
        base = self.get_base_world_pos()

        dir_x, dir_y, approach_angle = self._compute_approach_direction(
            object_world_pos)

        # ── Grasp plan from object metadata ───────────────────
        if object_metadata:
            grasp_plan        = self._plan_grasp_by_object(object_metadata)
            approach_offset_m = grasp_plan["approach_offset_m"]
            target_force_n    = grasp_plan["target_force_n"]
        else:
            approach_offset_m = 0.025
            target_force_n    = 80.0

        # ── Grasp height ──────────────────────────────────────
        height_info    = self._compute_grasp_height(
            object_metadata, table_height)
        grasp_strategy = height_info["strategy"]
        obj_height     = height_info["object_height"]

        # ── Grasp position ────────────────────────────────────
        flange_grasp    = obj.copy()
        flange_grasp[2] = height_info["tool0_z"]

        safe_z = self._safe_z_above_table()

        # ── Safe above (transit height) ───────────────────────
        flange_safe_above    = obj.copy()
        flange_safe_above[2] = safe_z
        flange_safe_above    = self._clamp_flange_z_above_table(
            flange_safe_above, "safe_above")

        # ── Pre-grasp: DIRECTLY ABOVE grasp (same XY) ─────────
        # XY is locked to the grasp XY, only Z is raised.
        pre_grasp_z_offset = self.config.get("pre_grasp_above_m", 0.15)

        flange_pre    = flange_grasp.copy()          
        flange_pre[2] = flange_grasp[2] + pre_grasp_z_offset  

        # ── Lift (after grasp) ────────────────────────────────
        flange_lift    = flange_grasp.copy()
        flange_lift[2] = (
            table_height
            + self._lift_above_table
            + self._flange_to_finger_base
            + self.ARM_BODY_CLEARANCE
        )

        # ── Safe retreat (back to transit height) ─────────────
        flange_safe_retreat    = flange_lift.copy()
        flange_safe_retreat[2] = safe_z

        # ── Retract (pull back toward base) ───────────────────
        to_base = np.array(base[:2]) - flange_lift[:2]
        norm    = np.linalg.norm(to_base)
        if norm > 1e-6:
            to_base /= norm
        flange_retract    = flange_safe_retreat.copy()
        flange_retract[0] += to_base[0] * 0.20
        flange_retract[1] += to_base[1] * 0.20
        flange_retract[2]  = safe_z

        if self._is_near_cabinet(flange_retract):
            flange_retract[2] = max(
                flange_retract[2], base[2] + 0.30)

        return {
            "safe_above":         flange_safe_above,
            "pre_grasp":          flange_pre,
            "grasp":              flange_grasp,
            "lift":               flange_lift,
            "safe_retreat":       flange_safe_retreat,
            "retract":            flange_retract,
            "approach_offset_m":  approach_offset_m,
            "target_force_n":     target_force_n,
            "approach_dir":       (dir_x, dir_y),
            "approach_angle_rad": approach_angle,
            "grasp_strategy":     grasp_strategy,
            "object_height":      obj_height,
            "height_info":        height_info,
        }

    # ══════════════════════════════════════════════════════════
    # GRASP PLANNING BY OBJECT TYPE
    # ══════════════════════════════════════════════════════════

    def _plan_grasp_by_object(self, obj_metadata) -> dict:
        shape = obj_metadata.get("shape", "Cube")
        mass  = obj_metadata.get("mass", 0.1)

        plan = {
            "strategy":          "external",
            "approach_offset_m": 0.0,
            "target_force_n":    80.0,
            "object_shape":      shape,
        }

        # ── Approach offset ────────────────────────────────────────
        if shape == "Cube":
            size      = obj_metadata.get("size", 0.035)
            offset_mm = (size / 2.0) * 1000 + self._grasp_clearance_mm

        elif shape == "Rectangle":
            # Grip across WIDTH (short side)
            width     = obj_metadata.get("width", 0.050)
            offset_mm = (width / 2.0) * 1000 + self._grasp_clearance_mm

        elif shape in ("Box",):
            sz        = obj_metadata.get(
                "size_xyz", (0.060, 0.030, 0.020))
            grasp_dim = min(sz[0], sz[1])
            offset_mm = (grasp_dim / 2.0) * 1000 + self._grasp_clearance_mm

        elif shape in ("Cylinder", "Disc"):
            radius    = obj_metadata.get("radius", 0.018)
            offset_mm = radius * 1000 + self._grasp_clearance_mm

        elif shape == "Sphere":
            radius    = obj_metadata.get("radius", 0.020)
            offset_mm = radius * 1000 + self._grasp_clearance_mm + 1.0

        else:
            offset_mm = 25.0

        plan["approach_offset_m"] = min(
            offset_mm / 1000.0, self._max_insertion_depth)

        # ── Grip force (unchanged) ─────────────────────────────────
        mu            = 0.6
        physics_force = (mass * 9.81 * self._force_safety) / (2 * mu)
        sim_min_force = max(40.0, mass * 200.0)
        plan["target_force_n"] = max(
            self._min_grip_force,
            min(self._max_grip_force,
                max(physics_force, sim_min_force)),
        )

        if shape == "Sphere":
            plan["target_force_n"] = min(
                self._max_grip_force,
                plan["target_force_n"] * 1.5)
        elif shape in ("Cylinder", "Disc"):
            plan["target_force_n"] = min(
                self._max_grip_force,
                plan["target_force_n"] * 1.2)

        obj_height = self._get_object_height(obj_metadata)
        if obj_height > self._finger_grasp_height:
            plan["target_force_n"] = min(
                self._max_grip_force,
                plan["target_force_n"] * 1.3)

        return plan

    # ══════════════════════════════════════════════════════════
    # IK SOLVERS
    # ══════════════════════════════════════════════════════════

    def _normalise_joint_name(self, name: str) -> str:
        """Return a short joint name suitable for USD/Lula order matching."""
        return str(name).split("/")[-1]

    def _get_lula_joint_names(self) -> list:
        """
        Return the joint order expected by Lula.

        The USD articulation and the Lula descriptor often use the same UR5e
        joint names, but a safe controller should not rely on that accidentally
        being true.  FK and IK inputs must use Lula's own c-space order.
        """
        if self._lula_solver is None:
            return list(self.JOINT_NAMES)

        try:
            return [
                self._normalise_joint_name(name)
                for name in self._lula_solver.get_joint_names()
            ]
        except Exception:
            # Compatibility fallback for Isaac versions without get_joint_names.
            return list(self.JOINT_NAMES)

    def _controller_deg_to_lula_rad(self, controller_deg) -> np.ndarray:
        """Map controller JOINT_NAMES order (deg) to Lula order (rad)."""
        controller_deg = np.asarray(controller_deg, dtype=np.float64)

        if controller_deg.shape != (6,):
            raise ValueError(
                f"Expected six controller joints, got shape={controller_deg.shape}"
            )

        by_name = {
            self._normalise_joint_name(name): float(value)
            for name, value in zip(self.JOINT_NAMES, controller_deg)
        }
        lula_names = self._get_lula_joint_names()

        missing = [name for name in lula_names if name not in by_name]
        if missing:
            raise ValueError(
                "Cannot map controller joints to Lula order. "
                f"Missing={missing}, Lula order={lula_names}, "
                f"controller order={self.JOINT_NAMES}"
            )

        return np.deg2rad(np.array([by_name[name] for name in lula_names]))

    def _lula_rad_to_controller_deg(self, lula_rad) -> np.ndarray:
        """Map Lula-order radians back to controller JOINT_NAMES degrees."""
        lula_rad = np.asarray(lula_rad, dtype=np.float64)
        lula_names = self._get_lula_joint_names()

        if lula_rad.shape != (len(lula_names),):
            raise ValueError(
                f"Unexpected Lula result shape={lula_rad.shape}; "
                f"Lula order={lula_names}"
            )

        by_name = {
            self._normalise_joint_name(name): float(value)
            for name, value in zip(lula_names, np.rad2deg(lula_rad))
        }

        missing = [name for name in self.JOINT_NAMES if name not in by_name]
        if missing:
            raise ValueError(
                "Cannot map Lula result back to controller order. "
                f"Missing={missing}"
            )

        return np.array([by_name[name] for name in self.JOINT_NAMES])

    def _solve_ik_lula(
        self, pos_local, orient=None, seed_deg=None,
    ):
        """
        Perform one Lula IK attempt.

        Important:
          - this function returns exactly one branch;
          - it does NOT run an expensive branch search;
          - the warm-start seed decides which nearby IK branch Lula attempts.

        Inputs and outputs exposed to the rest of the controller use the USD
        controller JOINT_NAMES order.  Internally, values are mapped to Lula's
        own c-space ordering.
        """
        if self._lula_solver is None:
            return None

        pos = np.array(pos_local, dtype=np.float64)
        orient = (
            np.array([0.0, 1.0, 0.0, 0.0])
            if orient is None
            else np.array(orient, dtype=np.float64)
        )
        seed_controller = (
            np.asarray(seed_deg, dtype=np.float64)
            if seed_deg is not None
            else self.get_joint_targets_deg()
        )

        try:
            seed_lula = self._controller_deg_to_lula_rad(seed_controller)

            result_lula, ok = self._lula_solver.compute_inverse_kinematics(
                frame_name=self._ee_frame,
                target_position=pos,
                target_orientation=orient,
                warm_start=seed_lula,
            )

            if not ok:
                return None

            return self._lula_rad_to_controller_deg(result_lula)

        except Exception as e:
            self._log(f"  [IK] Lula solve failed: {e}")
            return None

    def _solve_ik_retries(
        self,
        pos_local,
        orient    = None,
        retries:  int = None,
        seed_deg  = None,
    ) -> tuple:
        """
        Legacy convenience solver used outside the lean pick planner.

        Default behaviour is deliberately small and deterministic:
          1. provided/current seed
          2. home seed
          3. optional random retries only when explicitly configured

        The lean pick planner below does NOT use this method for branch search.
        """
        if retries is None:
            retries = int(self.config.get("ik_legacy_random_retries", 0))
        else:
            retries = int(self.config.get("ik_legacy_random_retries", retries))

        meta = {
            "attempts":      0,
            "total_retries": retries + 2,
            "solver":        self._ik_mode,
            "success":       False,
            "seed_used":     None,
        }

        # Attempt 1: user-provided or current joint seed.
        meta["attempts"] += 1
        r = self._solve_ik_lula(pos_local, orient, seed_deg)
        if r is not None:
            meta["success"]   = True
            meta["seed_used"] = "provided"
            return r, meta

        # Attempt 2: known home pose seed.
        meta["attempts"] += 1
        home = self.go_home()
        r = self._solve_ik_lula(pos_local, orient, home)
        if r is not None:
            meta["success"]   = True
            meta["seed_used"] = "home"
            return r, meta

        # Optional legacy random retries. Disabled by default.
        for i in range(retries):
            meta["attempts"] += 1
            seed = [
                random.uniform(lo, hi)
                for lo, hi in self.JOINT_LIMITS_DEG
            ]
            r = self._solve_ik_lula(pos_local, orient, seed)
            if r is not None:
                meta["success"]   = True
                meta["seed_used"] = f"random_{i}"
                return r, meta

        return None, meta


    # ══════════════════════════════════════════════════════════
    # CALIBRATION FALLBACK (no Lula available)
    # ══════════════════════════════════════════════════════════

    def _find_config_for_flange_z(self, z: float):
        """
        Interpolate the calibration table to find
        (shoulder_lift, elbow, reach) for a desired flange Z.
        """
        table = self.CALIB_TABLE
        zv    = [r[3] for r in table]

        if z > max(zv):
            return table[0][0], table[0][1], table[0][2]
        if z < min(zv):
            return table[-1][0], table[-1][1], table[-1][2]

        for i in range(len(table) - 1):
            if table[i + 1][3] <= z <= table[i][3]:
                f = (
                    (z - table[i + 1][3])
                    / (table[i][3] - table[i + 1][3])
                )
                return (
                    table[i + 1][0] + f * (table[i][0] - table[i + 1][0]),
                    table[i + 1][1] + f * (table[i][1] - table[i + 1][1]),
                    table[i + 1][2] + f * (table[i][2] - table[i + 1][2]),
                )

        m = len(table) // 2
        return table[m][0], table[m][1], table[m][2]

    def _compute_pick_calibration(
        self, object_world_pos, pan_deg, flange_targets,
    ) -> dict:
        """
        Compute joint angles for all pick waypoints using the
        calibration table (no IK solver needed).
        """
        pan    = pan_deg - self.PAN_OFFSET_DEG
        w2, w3 = -90.0, 0.0
        results = {}

        waypoint_order = [
            "safe_above", "pre_grasp", "grasp",
            "lift", "safe_retreat", "retract",
        ]

        for name in waypoint_order:
            if name not in flange_targets:
                continue
            fp = flange_targets[name]
            if not isinstance(fp, np.ndarray) or len(fp) != 3:
                continue

            z                = fp[2]
            lift, elbow, _   = self._find_config_for_flange_z(z)
            w1               = -90.0 - lift - elbow
            p                = pan

            if name == "retract":
                home_pan = self.config.get(
                    "arm_home_deg",
                    [0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
                )[0]
                p = pan + 0.5 * (home_pan - pan)

            results[name] = [p, lift, elbow, w1, w2, w3]

        return results

    # ══════════════════════════════════════════════════════════
    # GRASP ORIENTATION CANDIDATES
    # ══════════════════════════════════════════════════════════

    def _get_prim_yaw_rad(self, prim_path: str) -> float:
        """Extract the world-space yaw of a prim from its transform."""
        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
        if not prim.IsValid():
            return 0.0
        xf  = UsdGeom.Xformable(prim)
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return math.atan2(float(mtx[1][0]), float(mtx[0][0]))

    def _compute_grasp_candidates_for_object(
        self, object_metadata, prim_path,
    ) -> list:
        """
        Generate candidate grasp orientations based on object shape.

        For Rectangle: MUST grasp across the short side (width).
        For Cube: either face works (both sides equal).
        For Cylinder/Disc/Sphere: any yaw works (rotationally symmetric).

        Returns list of (label, quaternion[w,x,y,z]).
        """
        shape   = object_metadata.get("shape", "Cube")
        obj_yaw = (
            self._get_prim_yaw_rad(prim_path) if prim_path else 0.0
        )

        if shape == "Cube":
            # Both faces are equal.  Keep a deterministic preferred order.
            # Alternative face_B is available only if the caller explicitly
            # enables alternate grasp orientations.
            return [
                ("face_A", compute_grasp_orientation(obj_yaw)),
                ("face_B", compute_grasp_orientation(
                    obj_yaw + math.pi / 2)),
            ]

        if shape == "Rectangle":
            # ── CRITICAL: grip across the WIDTH (short side) ──────
            width  = object_metadata.get("width",  0.050)
            length = object_metadata.get("length", 0.080)

            # Primary: approach along length axis (grip across width)
            grip_across_width_yaw = obj_yaw + math.pi / 2.0

            candidates = [
                (f"across_width({width*1000:.0f}mm)",
                compute_grasp_orientation(grip_across_width_yaw)),
                (f"across_width_rev({width*1000:.0f}mm)",
                compute_grasp_orientation(
                    grip_across_width_yaw + math.pi)),
            ]

            # Only add length grasp if length fits in gripper range
            grip_min = self._grip_range_min
            grip_max = self._grip_range_max
            if grip_min <= length <= grip_max:
                candidates.append(
                    (f"across_length({length*1000:.0f}mm)",
                    compute_grasp_orientation(obj_yaw)),
                )
                self._log(
                    f"  [Grasp] Rectangle: length {length*1000:.0f}mm "
                    f"also fits gripper — added as fallback")
            else:
                self._log(
                    f"  [Grasp] Rectangle: length {length*1000:.0f}mm "
                    f"exceeds grip range "
                    f"[{grip_min*1000:.0f}–{grip_max*1000:.0f}mm] — "
                    f"width-only grasp")

            return candidates

        if shape in ("Box",):
            # Legacy Box shape with size_xyz
            sz = object_metadata.get(
                "size_xyz", (0.06, 0.03, 0.02))
            sx, sy = sz[0], sz[1]
            yaw_a  = obj_yaw
            yaw_b  = obj_yaw + math.pi / 2.0

            if sx <= sy:
                short_dim = sx
                long_dim  = sy
                short_yaw = yaw_a
                long_yaw  = yaw_b
            else:
                short_dim = sy
                long_dim  = sx
                short_yaw = yaw_b
                long_yaw  = yaw_a

            candidates = [
                (f"short({short_dim*1000:.0f}mm)",
                compute_grasp_orientation(short_yaw)),
            ]

            grip_min = self._grip_range_min
            grip_max = self._grip_range_max
            if grip_min <= long_dim <= grip_max:
                candidates.append(
                    (f"long({long_dim*1000:.0f}mm)",
                    compute_grasp_orientation(long_yaw)),
                )

            return candidates

        # ── Rotationally symmetric shapes ─────────────────
        # Use deterministic yaws.  Environment randomization is useful;
        # safety-critical grasp orientation selection should be repeatable.
        if shape in ("Cylinder", "Disc", "Sphere"):
            yaws = [obj_yaw, obj_yaw + math.pi / 2.0]
            return [
                (f"yaw_{i}", compute_grasp_orientation(y))
                for i, y in enumerate(yaws)
            ]

        # Unknown shape — deterministic conservative default.
        return [
            ("default", compute_grasp_orientation(obj_yaw)),
            ("default_90", compute_grasp_orientation(
                obj_yaw + math.pi / 2.0)),
        ]

    # ══════════════════════════════════════════════════════════
    # LEAN PLANNER-LEVEL FK COLLISION PREDICTION
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _wrapped_joint_delta_deg(a, b) -> np.ndarray:
        """Smallest signed revolute-joint difference a-b in degrees."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return (a - b + 180.0) % 360.0 - 180.0

    def _resolve_lula_frame(self, requested_name: str):
        """
        Resolve a configured link name to a frame exposed by Lula.

        If the Isaac build does not expose get_all_frame_names(), return the
        requested name and let compute_forward_kinematics perform the check.
        """
        if self._lula_solver is None:
            return None

        try:
            frames = list(self._lula_solver.get_all_frame_names())
        except Exception:
            return requested_name

        if requested_name in frames:
            return requested_name

        short = str(requested_name).split("/")[-1]
        for frame in frames:
            if str(frame).split("/")[-1] == short:
                return frame

        return None

    def _predict_lula_frame_world_pos(self, frame_name: str, joints_deg):
        """
        Predict a link-frame world position using Lula FK without moving PhysX.

        This is the crucial difference between:
          - planner-level checking: predict before moving;
          - runtime checking: observe the real simulated robot while moving.
        """
        if self._lula_solver is None:
            return None

        resolved = self._resolve_lula_frame(frame_name)
        if resolved is None:
            self._log(f"  [FK] Missing Lula frame: {frame_name}")
            return None

        try:
            q_lula = self._controller_deg_to_lula_rad(joints_deg)
            fk_result = self._lula_solver.compute_forward_kinematics(
                resolved,
                q_lula,
            )

            # Isaac versions commonly return (position, rotation).
            local_pos = fk_result[0] if isinstance(fk_result, tuple) else fk_result
            world = self._robot_base_to_world(np.asarray(local_pos))
            return np.asarray(world, dtype=np.float64)

        except Exception as e:
            self._log(f"  [FK] Failed for frame '{resolved}': {e}")
            return None

    def _planner_collision_segments(self) -> list:
        """Robot-link segments used by the cheap offline table checker."""
        return self.config.get(
            "lula_collision_segments",
            [
                ["upper_arm_link", "forearm_link"],
                ["forearm_link", "wrist_1_link"],
                ["wrist_1_link", "wrist_2_link"],
                ["wrist_2_link", "wrist_3_link"],
                ["wrist_3_link", self._ee_frame],
            ],
        )

    def _evaluate_predicted_table_clearance(
        self,
        joints_deg,
        label: str = "",
    ) -> dict:
        """
        FK-check one candidate arm posture before execution.

        The arm is approximated by thick sampled line segments.  This is much
        cheaper than a full collision engine query, but it is enough to reject
        the known elbow-down/table-collision failure mode.
        """
        result = {
            "safe": True,
            "label": label,
            "min_clearance_m": float("inf"),
            "reason": None,
        }

        if self._table_info is None:
            return result

        if self._lula_solver is None:
            result.update(
                safe=False,
                reason="Lula solver unavailable for predictive FK checking",
            )
            return result

        table_z = float(self._get_table_surface_z())
        body_margin = max(
            float(self._collision_check_margin),
            float(self.ARM_BODY_CLEARANCE),
        )
        required_body_z = table_z + body_margin
        required_flange_z = table_z + float(self._collision_check_margin)

        n_samples = max(
            1,
            int(self.config.get("planner_link_sample_count", 5)),
        )
        fail_closed = bool(
            self.config.get(
                "planner_fail_closed_on_missing_fk_frame",
                True,
            )
        )

        for link_a, link_b in self._planner_collision_segments():
            pos_a = self._predict_lula_frame_world_pos(link_a, joints_deg)
            pos_b = self._predict_lula_frame_world_pos(link_b, joints_deg)

            if pos_a is None or pos_b is None:
                if fail_closed:
                    result.update(
                        safe=False,
                        reason=f"missing FK frame for {link_a}->{link_b}",
                    )
                    return result
                continue

            for sample_idx in range(n_samples + 1):
                alpha = sample_idx / n_samples
                pt = pos_a + alpha * (pos_b - pos_a)

                if not self._is_over_table(pt):
                    continue

                clearance = float(pt[2] - required_body_z)
                result["min_clearance_m"] = min(
                    result["min_clearance_m"],
                    clearance,
                )

                if clearance < 0.0:
                    result.update(
                        safe=False,
                        reason=(
                            f"segment {link_a}->{link_b} too close to table "
                            f"at sample {sample_idx}/{n_samples}: "
                            f"z={pt[2]:.3f}, required>{required_body_z:.3f}"
                        ),
                    )
                    return result

        flange = self._predict_lula_frame_world_pos(self._ee_frame, joints_deg)

        if flange is None:
            if fail_closed:
                result.update(
                    safe=False,
                    reason="missing FK end-effector frame",
                )
                return result

        elif self._is_over_table(flange):
            clearance = float(flange[2] - required_flange_z)
            result["min_clearance_m"] = min(
                result["min_clearance_m"],
                clearance,
            )

            if clearance < 0.0:
                result.update(
                    safe=False,
                    reason=(
                        f"end effector too close to table: "
                        f"z={flange[2]:.3f}, required>{required_flange_z:.3f}"
                    ),
                )
                return result

        if not np.isfinite(result["min_clearance_m"]):
            # No sampled body point was over the table footprint.
            result["min_clearance_m"] = 1.0

        return result

    def _evaluate_predicted_path_clearance(
        self,
        from_deg,
        to_deg,
        label: str = "",
        n_samples: int = None,
    ) -> dict:
        """
        FK-check a low-resolution joint-space transition before execution.

        This is deliberately lean: the runtime collision guard still checks
        every simulation update while the actual robot moves.
        """
        n_samples = max(
            1,
            int(
                n_samples
                if n_samples is not None
                else self.config.get("planner_path_sample_count", 6)
            ),
        )

        q0 = np.asarray(from_deg, dtype=np.float64)
        q1 = np.asarray(to_deg, dtype=np.float64)

        result = {
            "safe": True,
            "label": label,
            "min_clearance_m": float("inf"),
            "reason": None,
        }

        delta = self._wrapped_joint_delta_deg(q1, q0)

        for sample_idx in range(n_samples + 1):
            alpha = sample_idx / n_samples
            q = q0 + alpha * delta

            pose_eval = self._evaluate_predicted_table_clearance(
                q,
                label=f"{label}@{sample_idx}/{n_samples}",
            )

            result["min_clearance_m"] = min(
                result["min_clearance_m"],
                pose_eval["min_clearance_m"],
            )

            if not pose_eval["safe"]:
                result.update(
                    safe=False,
                    reason=(
                        f"{label} unsafe at sample {sample_idx}/{n_samples}: "
                        f"{pose_eval['reason']}"
                    ),
                )
                return result

        if not np.isfinite(result["min_clearance_m"]):
            result["min_clearance_m"] = 1.0

        return result

    def _optional_fallback_seed_bank(self, reference_deg) -> list:
        """
        Return a tiny deterministic fallback seed bank.

        Disabled by default.  Turn it on only if the nominal chain is rejected
        too often despite visually reachable targets.
        """
        if not self.config.get("ik_enable_fallback_branches", False):
            return []

        reference = np.asarray(reference_deg, dtype=np.float64)
        home = np.asarray(self.go_home(), dtype=np.float64)

        templates = [
            (
                "home",
                home,
            ),
            (
                "elbow_up_A",
                np.array([
                    home[0], -90.0, 90.0, -90.0, home[4], home[5],
                ]),
            ),
            (
                "elbow_up_B",
                np.array([
                    home[0], -60.0, 90.0, -120.0, home[4], home[5],
                ]),
            ),
        ]

        max_count = max(
            0,
            int(self.config.get("ik_max_fallback_seeds", 2)),
        )

        unique = []
        for label, q in templates:
            q = np.asarray(q, dtype=np.float64)

            if np.linalg.norm(
                self._wrapped_joint_delta_deg(q, reference)
            ) < 1.0:
                continue

            if any(
                np.linalg.norm(
                    self._wrapped_joint_delta_deg(q, previous_q)
                ) < 1.0
                for _previous_label, previous_q in unique
            ):
                continue

            unique.append((label, q))

        return unique[:max_count]

    def _solve_checked_waypoint_nominal_then_fallback(
        self,
        pos_local,
        orient,
        reference_deg,
        waypoint_name: str,
    ) -> tuple:
        """
        Solve one waypoint cheaply and safely.

        Fast path:
          1. solve once from the continuous previous waypoint;
          2. FK-check the resulting posture;
          3. FK-check the transition into it.

        Optional fallback:
          only if explicitly enabled and only after the nominal result fails,
          try at most a few deterministic alternative warm starts.
        """
        reference = np.asarray(reference_deg, dtype=np.float64)

        attempts = [("nominal_reference", reference)]
        attempts.extend(self._optional_fallback_seed_bank(reference))

        meta = {
            "waypoint": waypoint_name,
            "attempts": 0,
            "selected_seed": None,
            "used_fallback": False,
            "min_clearance_m": None,
            "rejections": [],
        }

        for attempt_idx, (seed_name, seed_deg) in enumerate(attempts):
            meta["attempts"] += 1

            joints = self._solve_ik_lula(
                pos_local,
                orient,
                seed_deg=seed_deg,
            )

            if joints is None:
                meta["rejections"].append(
                    {
                        "seed": seed_name,
                        "reason": "IK did not converge",
                    }
                )
                continue

            pose_eval = self._evaluate_predicted_table_clearance(
                joints,
                label=waypoint_name,
            )

            if not pose_eval["safe"]:
                meta["rejections"].append(
                    {
                        "seed": seed_name,
                        "reason": pose_eval["reason"],
                    }
                )
                continue

            path_eval = self._evaluate_predicted_path_clearance(
                reference,
                joints,
                label=f"to_{waypoint_name}",
            )

            if not path_eval["safe"]:
                meta["rejections"].append(
                    {
                        "seed": seed_name,
                        "reason": path_eval["reason"],
                    }
                )
                continue

            meta["selected_seed"] = seed_name
            meta["used_fallback"] = attempt_idx > 0
            meta["min_clearance_m"] = min(
                pose_eval["min_clearance_m"],
                path_eval["min_clearance_m"],
            )

            if meta["used_fallback"]:
                self._log(
                    f"  [LeanIK] {waypoint_name}: nominal rejected; "
                    f"using fallback seed '{seed_name}'"
                )

            return np.asarray(joints, dtype=np.float64), meta

        return None, meta

    def _pick_planning_waypoint_order(self) -> list:
        """
        Return only the waypoint horizon needed by the current experiment.

        Default baseline:
          safe_above -> pre_grasp -> grasp

        Lift and retreat are deliberately postponed until they are actually
        under test.  This avoids paying for unnecessary future planning while
        debugging generic grasp geometry.
        """
        order = [
            "safe_above",
            "pre_grasp",
            "grasp",
        ]

        if (
            self.config.get("enable_micro_lift_test", False)
            or self.config.get("enable_lift_test", False)
        ):
            order.append("lift")

        if self.config.get("plan_retreat_during_pick", False):
            order.extend([
                "safe_retreat",
                "retract",
            ])

        return order

    # ══════════════════════════════════════════════════════════
    # IK VALIDATION (collision-free joint configs)
    # ══════════════════════════════════════════════════════════

    def _validate_joints_no_table_collision(
        self, joint_angles_deg, waypoint_name: str = "",
    ) -> bool:
        """
        Backward-compatible FK-based posture validation wrapper.

        Unlike the old implementation, this does not write PhysX drive targets
        and immediately inspect stale USD transforms.
        """
        evaluation = self._evaluate_predicted_table_clearance(
            joint_angles_deg,
            label=waypoint_name,
        )

        if not evaluation["safe"]:
            self._log(
                f"  [IK] Predictive collision at '{waypoint_name}': "
                f"{evaluation['reason']}"
            )

        return bool(evaluation["safe"])

    def _validate_path_between_joints(
        self,
        from_deg,
        to_deg,
        n_samples: int = None,
        label:     str = "",
    ) -> bool:
        """Backward-compatible FK-based transition validation wrapper."""
        evaluation = self._evaluate_predicted_path_clearance(
            from_deg,
            to_deg,
            label=label,
            n_samples=n_samples,
        )

        if not evaluation["safe"]:
            self._log(
                f"  [IK] Predictive path collision at '{label}': "
                f"{evaluation['reason']}"
            )

        return bool(evaluation["safe"])


    # ══════════════════════════════════════════════════════════
    # CASCADED PICK PLANNER HELPERS
    # ══════════════════════════════════════════════════════════

    def _chain_fallback_initial_seeds(self) -> list:
        """
        Return at most two deterministic whole-chain warm-start hypotheses.

        These values are NOT commanded to the robot.  They are only Lula
        warm-start hints for the first waypoint (safe_above).  Once a safe
        safe_above branch has been found, each following waypoint is solved
        continuously from the previous waypoint solution.

        Why whole-chain retries?
          An elbow-down and an elbow-up branch may both be safe at safe_above.
          The bad branch can reveal itself only later during descent.  Retrying
          only the final grasp waypoint is therefore insufficient: we must
          restart the chain from safe_above using a different IK hypothesis.
        """
        if not self.config.get("ik_enable_chain_fallbacks", True):
            return []

        home = np.asarray(self.go_home(), dtype=np.float64)
        templates = [
            (
                "elbow_up_A",
                np.array([
                    home[0], -90.0, 90.0, -90.0, home[4], home[5],
                ], dtype=np.float64),
            ),
            (
                "elbow_up_B",
                np.array([
                    home[0], -60.0, 90.0, -120.0, home[4], home[5],
                ], dtype=np.float64),
            ),
        ]

        max_count = max(
            0,
            min(2, int(self.config.get("ik_max_chain_fallbacks", 2))),
        )
        return templates[:max_count]

    def _solve_chain_from_initial_seed(
        self,
        flange_targets,
        tool_orient,
        waypoint_order,
        actual_start_deg,
        initial_seed_deg,
        chain_label: str,
    ) -> tuple:
        """
        Solve and FK-check one complete waypoint chain sequentially.

        The first waypoint is solved using initial_seed_deg as a Lula warm
        start, but its path is checked from the robot's ACTUAL current state.
        Subsequent waypoints are warm-started from the previous solved pose.

        Returns:
            (results_dict | None, chain_meta)
        """
        actual_start = np.asarray(actual_start_deg, dtype=np.float64)
        seed = np.asarray(initial_seed_deg, dtype=np.float64)
        previous_solution = None
        results = {}
        waypoint_meta = {}

        for waypoint_name in waypoint_order:
            world_target = flange_targets.get(waypoint_name)
            if not isinstance(world_target, np.ndarray) or len(world_target) != 3:
                continue

            local_target = self._world_to_robot_base(world_target)
            warm_start = seed if previous_solution is None else previous_solution
            path_start = actual_start if previous_solution is None else previous_solution

            joints = self._solve_ik_lula(
                local_target,
                tool_orient,
                seed_deg=warm_start,
            )

            meta = {
                "waypoint": waypoint_name,
                "warm_start": chain_label if previous_solution is None else "previous_waypoint",
                "ik_converged": joints is not None,
                "pose_safe": False,
                "path_safe": False,
                "min_clearance_m": None,
                "reason": None,
            }

            if joints is None:
                meta["reason"] = "IK did not converge"
                waypoint_meta[waypoint_name] = meta
                return None, {
                    "chain_label": chain_label,
                    "success": False,
                    "failed_waypoint": waypoint_name,
                    "waypoint_meta": waypoint_meta,
                }

            pose_eval = self._evaluate_predicted_table_clearance(
                joints,
                label=f"{chain_label}:{waypoint_name}",
            )
            meta["pose_safe"] = bool(pose_eval["safe"])

            if not pose_eval["safe"]:
                meta["reason"] = pose_eval["reason"]
                waypoint_meta[waypoint_name] = meta
                return None, {
                    "chain_label": chain_label,
                    "success": False,
                    "failed_waypoint": waypoint_name,
                    "waypoint_meta": waypoint_meta,
                }

            path_eval = self._evaluate_predicted_path_clearance(
                path_start,
                joints,
                label=f"{chain_label}:to_{waypoint_name}",
            )
            meta["path_safe"] = bool(path_eval["safe"])

            if not path_eval["safe"]:
                meta["reason"] = path_eval["reason"]
                waypoint_meta[waypoint_name] = meta
                return None, {
                    "chain_label": chain_label,
                    "success": False,
                    "failed_waypoint": waypoint_name,
                    "waypoint_meta": waypoint_meta,
                }

            meta["min_clearance_m"] = float(min(
                pose_eval["min_clearance_m"],
                path_eval["min_clearance_m"],
            ))
            waypoint_meta[waypoint_name] = meta
            results[waypoint_name] = list(np.asarray(joints, dtype=np.float64))
            previous_solution = np.asarray(joints, dtype=np.float64)

        if not results:
            return None, {
                "chain_label": chain_label,
                "success": False,
                "failed_waypoint": None,
                "waypoint_meta": waypoint_meta,
                "reason": "no valid waypoints were present",
            }

        return results, {
            "chain_label": chain_label,
            "success": True,
            "failed_waypoint": None,
            "waypoint_meta": waypoint_meta,
        }

    def _score_safe_beam_branch(
        self,
        joints_deg,
        reference_deg,
        pose_eval: dict,
        path_eval: dict,
    ) -> float:
        """Lower score is better: short motion with generous clearance."""
        delta = self._wrapped_joint_delta_deg(joints_deg, reference_deg)
        motion_cost = float(np.dot(delta, delta)) / (180.0 ** 2)
        min_clearance = max(
            1e-3,
            min(
                float(pose_eval["min_clearance_m"]),
                float(path_eval["min_clearance_m"]),
            ),
        )
        clearance_cost = 1.0 / min_clearance
        return (
            float(self.config.get("ik_motion_score_weight", 1.0)) * motion_cost
            + float(self.config.get("ik_clearance_score_weight", 0.05))
            * clearance_cost
        )

    def _beam_seed_bank(self, reference_deg) -> list:
        """
        Return a small deterministic seed bank for last-resort beam search.

        Beam search is intentionally lazy: this function is used only after
        the nominal whole-chain attempt AND both lightweight whole-chain
        alternatives have failed.
        """
        reference = np.asarray(reference_deg, dtype=np.float64)
        home = np.asarray(self.go_home(), dtype=np.float64)

        templates = [
            ("reference", reference),
            ("home", home),
            ("elbow_up_A", [home[0], -90.0, 90.0, -90.0, home[4], home[5]]),
            ("elbow_up_B", [home[0], -60.0, 90.0, -120.0, home[4], home[5]]),
            ("elbow_up_C", [home[0], -120.0, 90.0, -60.0, home[4], home[5]]),
            ("elbow_alt_A", [home[0], -90.0, -90.0, 90.0, home[4], home[5]]),
            ("shoulder_plus", [home[0] + 180.0, -90.0, 90.0, -90.0, home[4], home[5]]),
            ("shoulder_minus", [home[0] - 180.0, -90.0, 90.0, -90.0, home[4], home[5]]),
        ]

        max_count = max(1, int(self.config.get("ik_beam_max_seed_hypotheses", 6)))
        unique = []
        for label, values in templates:
            q = np.asarray(values, dtype=np.float64)
            if q.shape != (6,):
                continue
            if any(
                np.linalg.norm(self._wrapped_joint_delta_deg(q, old_q)) < 1.0
                for _old_label, old_q in unique
            ):
                continue
            unique.append((label, q))
        return unique[:max_count]

    def _collect_safe_beam_branches(
        self,
        pos_local,
        orient,
        reference_deg,
        waypoint_name: str,
    ) -> tuple:
        """Generate, deduplicate, FK-check and score beam-search branches."""
        reference = np.asarray(reference_deg, dtype=np.float64)
        solutions = []
        rejections = []
        dedup_deg = float(self.config.get("ik_solution_dedup_deg", 4.0))

        for seed_name, seed_deg in self._beam_seed_bank(reference):
            joints = self._solve_ik_lula(pos_local, orient, seed_deg=seed_deg)
            if joints is None:
                rejections.append({"seed": seed_name, "reason": "IK did not converge"})
                continue

            joints = np.asarray(joints, dtype=np.float64)
            if any(
                np.linalg.norm(self._wrapped_joint_delta_deg(joints, old["joints"]))
                < dedup_deg
                for old in solutions
            ):
                continue

            pose_eval = self._evaluate_predicted_table_clearance(
                joints,
                label=f"beam:{waypoint_name}:{seed_name}",
            )
            if not pose_eval["safe"]:
                rejections.append({"seed": seed_name, "reason": pose_eval["reason"]})
                continue

            path_eval = self._evaluate_predicted_path_clearance(
                reference,
                joints,
                label=f"beam:to_{waypoint_name}:{seed_name}",
            )
            if not path_eval["safe"]:
                rejections.append({"seed": seed_name, "reason": path_eval["reason"]})
                continue

            solutions.append({
                "seed": seed_name,
                "joints": joints,
                "score": self._score_safe_beam_branch(
                    joints,
                    reference,
                    pose_eval,
                    path_eval,
                ),
                "min_clearance_m": float(min(
                    pose_eval["min_clearance_m"],
                    path_eval["min_clearance_m"],
                )),
            })

        solutions.sort(key=lambda item: item["score"])
        return solutions, {
            "waypoint": waypoint_name,
            "safe_branch_count": len(solutions),
            "rejections": rejections,
        }

    def _solve_chain_with_beam_fallback(
        self,
        flange_targets,
        tool_orient,
        waypoint_order,
        actual_start_deg,
    ) -> tuple:
        """
        Last-resort downstream-aware beam search through the current horizon.

        This is NOT run during normal operation.  It activates only after:
          1. nominal whole-chain planning failed;
          2. elbow_up_A whole-chain retry failed;
          3. elbow_up_B whole-chain retry failed.
        """
        start = np.asarray(actual_start_deg, dtype=np.float64)
        beam_width = max(1, int(self.config.get("ik_beam_width", 4)))
        max_children = max(
            1,
            int(self.config.get("ik_beam_max_children_per_state", 3)),
        )

        frontier = [{
            "prev_joints": start.copy(),
            "results": {},
            "total_score": 0.0,
            "trace": [],
            "branch_meta": {},
        }]

        for waypoint_name in waypoint_order:
            world_target = flange_targets.get(waypoint_name)
            if not isinstance(world_target, np.ndarray) or len(world_target) != 3:
                continue

            local_target = self._world_to_robot_base(world_target)
            expanded = []

            for state_idx, state in enumerate(frontier):
                branches, wp_meta = self._collect_safe_beam_branches(
                    local_target,
                    tool_orient,
                    reference_deg=state["prev_joints"],
                    waypoint_name=waypoint_name,
                )

                for branch in branches[:max_children]:
                    branch_meta = dict(state["branch_meta"])
                    branch_meta[waypoint_name] = {
                        **wp_meta,
                        "selected_seed": branch["seed"],
                        "selected_score": float(branch["score"]),
                        "selected_min_clearance_m": float(branch["min_clearance_m"]),
                    }
                    expanded.append({
                        "prev_joints": np.asarray(branch["joints"], dtype=np.float64),
                        "results": {
                            **state["results"],
                            waypoint_name: list(np.asarray(branch["joints"], dtype=np.float64)),
                        },
                        "total_score": float(state["total_score"]) + float(branch["score"]),
                        "trace": state["trace"] + [{
                            "waypoint": waypoint_name,
                            "seed": branch["seed"],
                            "score": float(branch["score"]),
                            "min_clearance_m": float(branch["min_clearance_m"]),
                            "parent_state": state_idx,
                        }],
                        "branch_meta": branch_meta,
                    })

            if not expanded:
                return None, {
                    "planner": "beam_fallback",
                    "success": False,
                    "failed_waypoint": waypoint_name,
                }

            expanded.sort(key=lambda item: item["total_score"])
            frontier = expanded[:beam_width]
            self._log(
                f"  [BeamFallback] {waypoint_name}: kept={len(frontier)}/"
                f"{len(expanded)} partial chains; "
                f"best_score={frontier[0]['total_score']:.4f}"
            )

        if not frontier:
            return None, {
                "planner": "beam_fallback",
                "success": False,
                "failed_waypoint": None,
            }

        best = min(frontier, key=lambda item: item["total_score"])
        return best["results"], {
            "planner": "beam_fallback",
            "success": True,
            "failed_waypoint": None,
            "chain_score": float(best["total_score"]),
            "trace": best["trace"],
            "branch_meta": best["branch_meta"],
        }

    # ══════════════════════════════════════════════════════════
    # LULA IK WAYPOINT SOLVER (with orientation candidates)
    # ══════════════════════════════════════════════════════════

    def _compute_pick_lula(
        self,
        flange_targets,
        object_world_pos = None,
        object_metadata  = None,
        prim_path        = None,
    ) -> tuple:
        """
        Solve a cascaded collision-aware pick chain.

        Fast normal case:
          Tier 0 — one intuitive continuous chain:
            actual current pose -> safe_above -> pre_grasp -> grasp

        Recovery only when Tier 0 fails:
          Tier 1 — restart the COMPLETE chain from at most two deterministic
                   elbow-up warm-start hypotheses;
          Tier 2 — if enabled, run a small downstream-aware beam search.

        Default horizon ends at grasp.  Lift is appended only when lift testing
        is enabled; retreat is appended only when explicitly requested.
        """
        planning_started = time.perf_counter()

        if object_metadata and prim_path:
            orientation_candidates = self._compute_grasp_candidates_for_object(
                object_metadata,
                prim_path,
            )
        else:
            orientation_candidates = [
                ("X", compute_grasp_orientation(0.0)),
                ("Y", compute_grasp_orientation(math.pi / 2.0)),
            ]

        # Keep contact-quality intent in control.  A different gripper yaw is
        # tried only when explicitly requested by the grasp-strategy layer.
        if not self.config.get("grasp_try_alternate_orientations", False):
            orientation_candidates = orientation_candidates[:1]

        waypoint_order = self._pick_planning_waypoint_order()
        actual_start = np.asarray(self.get_joint_targets_deg(), dtype=np.float64)
        rejected_orientations = []

        for orient_name, tool_orient in orientation_candidates:
            self._log(
                f"  [CascadeIK] Orientation '{orient_name}' through "
                f"{waypoint_order}"
            )
            tier_meta = []

            # ── Tier 0: intuitive continuous chain ────────────
            nominal_results, nominal_meta = self._solve_chain_from_initial_seed(
                flange_targets,
                tool_orient,
                waypoint_order,
                actual_start_deg=actual_start,
                initial_seed_deg=actual_start,
                chain_label="nominal_current",
            )
            tier_meta.append({"tier": 0, **nominal_meta})

            if nominal_results is not None:
                planning_time_s = time.perf_counter() - planning_started
                print(
                    f"    [CascadeIK] ✅ Tier 0 nominal chain selected: "
                    f"orientation={orient_name}, "
                    f"waypoints={list(nominal_results.keys())}, "
                    f"planning_time={planning_time_s:.3f}s"
                )
                return nominal_results, {
                    "solver": self._ik_mode,
                    "planner": "cascade_nominal_then_chain_fallback_then_beam",
                    "selected_tier": 0,
                    "selected_chain": "nominal_current",
                    "orient_name": orient_name,
                    "orient_quat": list(tool_orient),
                    "waypoints_solved": list(nominal_results.keys()),
                    "waypoints_failed": [],
                    "tier_meta": tier_meta,
                    "planning_time_s": planning_time_s,
                }

            self._log(
                "  [CascadeIK] Tier 0 nominal chain rejected; "
                "trying lightweight whole-chain alternatives"
            )

            # ── Tier 1: two lightweight complete-chain retries ─
            for fallback_label, fallback_seed in self._chain_fallback_initial_seeds():
                fallback_results, fallback_meta = self._solve_chain_from_initial_seed(
                    flange_targets,
                    tool_orient,
                    waypoint_order,
                    actual_start_deg=actual_start,
                    initial_seed_deg=fallback_seed,
                    chain_label=fallback_label,
                )
                tier_meta.append({"tier": 1, **fallback_meta})

                if fallback_results is not None:
                    planning_time_s = time.perf_counter() - planning_started
                    print(
                        f"    [CascadeIK] ✅ Tier 1 whole-chain fallback "
                        f"selected: {fallback_label}, "
                        f"orientation={orient_name}, "
                        f"waypoints={list(fallback_results.keys())}, "
                        f"planning_time={planning_time_s:.3f}s"
                    )
                    return fallback_results, {
                        "solver": self._ik_mode,
                        "planner": "cascade_nominal_then_chain_fallback_then_beam",
                        "selected_tier": 1,
                        "selected_chain": fallback_label,
                        "orient_name": orient_name,
                        "orient_quat": list(tool_orient),
                        "waypoints_solved": list(fallback_results.keys()),
                        "waypoints_failed": [],
                        "tier_meta": tier_meta,
                        "planning_time_s": planning_time_s,
                    }

            # ── Tier 2: lazy beam search as last resort ───────
            beam_results = None
            beam_meta = None
            if self.config.get("ik_enable_beam_fallback", True):
                self._log(
                    "  [CascadeIK] Lightweight alternatives failed; "
                    "activating lazy beam-search fallback"
                )
                beam_results, beam_meta = self._solve_chain_with_beam_fallback(
                    flange_targets,
                    tool_orient,
                    waypoint_order,
                    actual_start_deg=actual_start,
                )
                tier_meta.append({"tier": 2, **beam_meta})

            if beam_results is not None:
                planning_time_s = time.perf_counter() - planning_started
                print(
                    f"    [CascadeIK] ✅ Tier 2 beam fallback selected: "
                    f"orientation={orient_name}, "
                    f"waypoints={list(beam_results.keys())}, "
                    f"planning_time={planning_time_s:.3f}s"
                )
                return beam_results, {
                    "solver": self._ik_mode,
                    "planner": "cascade_nominal_then_chain_fallback_then_beam",
                    "selected_tier": 2,
                    "selected_chain": "beam_fallback",
                    "orient_name": orient_name,
                    "orient_quat": list(tool_orient),
                    "waypoints_solved": list(beam_results.keys()),
                    "waypoints_failed": [],
                    "tier_meta": tier_meta,
                    "planning_time_s": planning_time_s,
                }

            rejected_orientations.append({
                "orient_name": orient_name,
                "tier_meta": tier_meta,
            })

        planning_time_s = time.perf_counter() - planning_started
        self._log("  [CascadeIK] ❌ No safe grasp chain found")
        return None, {
            "solver": self._ik_mode,
            "planner": "cascade_nominal_then_chain_fallback_then_beam",
            "selected_tier": None,
            "selected_chain": None,
            "orient_name": None,
            "orient_quat": None,
            "waypoints_solved": [],
            "waypoints_failed": waypoint_order,
            "rejected_orientations": rejected_orientations,
            "planning_time_s": planning_time_s,
        }

    # ══════════════════════════════════════════════════════════
    # PUBLIC API — compute_pick_joints
    # ══════════════════════════════════════════════════════════

    def compute_pick_joints(
        self,
        object_world_pos,
        pan_to_object_deg: float,
        table_height:      float = 0.75,
        object_metadata          = None,
        prim_path                = None,
    ) -> dict:
        """
        Compute joint angles for all pick waypoints.

        PATCH: result dict now always contains 'ik_meta' key.

        Returns:
            {
                "joints":        {waypoint: [6 floats deg]},
                "flange_targets":{waypoint: np.ndarray[3]},
                "target_force_n": float,
                "grasp_strategy": str,
                "object_height":  float,
                "height_info":    dict,
                "ik_meta":        dict,   ← NEW
            }
        """
        base  = self.get_base_world_pos()
        dx    = object_world_pos[0] - base[0]
        dy    = object_world_pos[1] - base[1]
        horiz = math.sqrt(dx ** 2 + dy ** 2)

        if horiz > 0.85:
            print(f"  [IK] ⚠️  Near max reach ({horiz:.2f}m)")

        # ── Compute flange targets for all waypoints ──────────
        targets = self._compute_flange_targets(
            object_world_pos, table_height, object_metadata)

        # ── Base result dict ──────────────────────────────────
        result = {
            "flange_targets": targets,
            "target_force_n": targets.get("target_force_n", 80.0),
            "grasp_strategy": targets.get("grasp_strategy", "full_wrap"),
            "object_height":  targets.get("object_height", 0.035),
            "height_info":    targets.get("height_info", {}),
            "planning_failed": False,
            "ik_meta": {
                "solver":           self._ik_mode,
                "orient_name":      None,
                "orient_quat":      None,
                "orient_yaw_deg":   None,
                "waypoints_solved": [],
                "waypoints_failed": [],
                "attempts_per_wp":  {},
                "used_calibration": False,
            },
        }

        # ── Try Lula IK first ─────────────────────────────────
        if self._ik_mode in ("lula", "rmpflow"):
            # PATCHED: unpack (joints, ik_meta) tuple
            lula_joints, lula_meta = self._compute_pick_lula(
                targets,
                object_world_pos=object_world_pos,
                object_metadata=object_metadata,
                prim_path=prim_path,
            )
            if lula_joints is not None:
                result["joints"]  = lula_joints
                result["ik_meta"] = lula_meta
                return result

            self._log("  [IK] Lula failed → calibration fallback")
            result["ik_meta"].update(lula_meta)

        # ── Calibration fallback ──────────────────────────────
        # Safety-first default: helps to not to allow silent bypass of predictive FK checks
        # when Lula could not produce a verified/valid grasp chain.
        if not self.config.get(
            "allow_calibration_fallback_for_pick",
            True,
        ):
            result["joints"] = {}
            result["planning_failed"] = True
            result["ik_meta"]["used_calibration"] = False
            result["ik_meta"]["solver"] = self._ik_mode
            return result

        result["joints"] = self._compute_pick_calibration(
            object_world_pos,
            pan_to_object_deg,
            targets,
        )
        result["planning_failed"] = False
        result["ik_meta"]["used_calibration"] = True
        result["ik_meta"]["solver"] = "calibration"
        return result

    # ══════════════════════════════════════════════════════════
    # PUBLIC API — compute_place_joints
    # ══════════════════════════════════════════════════════════

    def compute_place_joints(
        self,
        place_world_pos,
        pan_to_place_deg: float,
        table_height:     float = 0.75,
        object_metadata         = None,
    ) -> dict:
        """
        Compute joint angles for all place waypoints.

        Returns:
            {waypoint: [6 joint angles deg]}
        """
        base = self.get_base_world_pos()
        obj  = np.array(place_world_pos, dtype=np.float64)

        dir_x, dir_y, approach_angle = (
            self._compute_approach_direction(place_world_pos))

        safe_z = self._safe_z_above_table()

        # ── Flange targets for place ──────────────────────────
        flange_safe_above    = obj.copy()
        flange_safe_above[2] = safe_z
        flange_safe_above    = self._clamp_flange_z_above_table(
            flange_safe_above, "place_safe_above")

        flange_pre    = obj.copy()
        flange_pre[2] = (
            table_height
            + self._pre_grasp_above_table
            + self._flange_to_finger_base
            + self.ARM_BODY_CLEARANCE
        )

        flange_place    = obj.copy()
        flange_place[2] = (
            obj[2] + self._flange_to_fingertips + 0.002)
        min_tool0_z = (
            table_height + self._flange_to_fingertips + 0.002)
        if flange_place[2] < min_tool0_z:
            flange_place[2] = min_tool0_z

        flange_lift    = flange_place.copy()
        flange_lift[2] = (
            table_height
            + self._lift_above_table
            + self._flange_to_finger_base
            + self.ARM_BODY_CLEARANCE
        )

        flange_safe_retreat    = flange_lift.copy()
        flange_safe_retreat[2] = safe_z

        to_base = np.array(base[:2]) - flange_lift[:2]
        norm    = np.linalg.norm(to_base)
        if norm > 1e-6:
            to_base /= norm
        flange_retract    = flange_safe_retreat.copy()
        flange_retract[0] += to_base[0] * 0.20
        flange_retract[1] += to_base[1] * 0.20
        flange_retract[2]  = safe_z

        if self._is_near_cabinet(flange_retract):
            flange_retract[2] = max(
                flange_retract[2], base[2] + 0.30)

        place_targets = {
            "safe_above":         flange_safe_above,
            "pre_place":          flange_pre,
            "place":              flange_place,
            "lift":               flange_lift,
            "safe_retreat":       flange_safe_retreat,
            "retract":            flange_retract,
            "approach_angle_rad": approach_angle,
        }

        # ── Lula IK ──────────────────────────────────────────
        if self._ik_mode in ("lula", "rmpflow"):
            tool_orient    = compute_grasp_orientation(approach_angle)
            waypoint_order = [
                "safe_above", "pre_place", "place",
                "lift", "safe_retreat", "retract",
            ]
            results  = {}
            prev     = None
            all_ok   = True

            for name in waypoint_order:
                if name not in place_targets:
                    continue
                w = place_targets[name]
                if not isinstance(w, np.ndarray) or len(w) != 3:
                    continue

                local = self._world_to_robot_base(w)
                j, _wp_meta = self._solve_ik_retries(local, tool_orient, seed_deg=prev)

                if j is None:
                    self._log(
                        f"  [IK] Place: no solution for '{name}'")
                    all_ok = False
                    break

                if not self._validate_joints_no_table_collision(
                        j, waypoint_name=name):
                    all_ok = False
                    break

                if prev is not None:
                    prev_name = waypoint_order[
                        waypoint_order.index(name) - 1]
                    if not self._validate_path_between_joints(
                            prev, j, n_samples=10,
                            label=f"{prev_name}→{name}"):
                        all_ok = False
                        break

                results[name] = list(j)
                prev = j

            if all_ok and results:
                return results

            self._log("  [IK] Place Lula failed → calibration")

        # ── Calibration fallback ──────────────────────────────
        pan    = pan_to_place_deg - self.PAN_OFFSET_DEG
        w2, w3 = -90.0, 0.0
        results = {}
        waypoint_order = [
            "safe_above", "pre_place", "place",
            "lift", "safe_retreat", "retract",
        ]
        for name in waypoint_order:
            if name not in place_targets:
                continue
            fp = place_targets[name]
            if not isinstance(fp, np.ndarray) or len(fp) != 3:
                continue
            z              = fp[2]
            lift, elbow, _ = self._find_config_for_flange_z(z)
            w1             = -90.0 - lift - elbow
            p              = pan
            if name == "retract":
                home_pan = self.config.get(
                    "arm_home_deg",
                    [0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
                )[0]
                p = pan + 0.5 * (home_pan - pan)
            results[name] = [p, lift, elbow, w1, w2, w3]

        return results

    # ══════════════════════════════════════════════════════════
    # PUBLIC API — HOME + STATUS
    # ══════════════════════════════════════════════════════════

    def go_home(self) -> list:
        """Return home joint configuration (degrees)."""
        home = self.config.get(
            "arm_home_deg",
            [0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
        )

        # Handle accidental nested YAML list: [[...]]
        if isinstance(home, list) and len(home) == 1 and isinstance(home[0], list):
            home = home[0]

        if len(home) != 6:
            raise ValueError(f"arm_home_deg must contain exactly 6 values, got: {home}")

        return [float(v) for v in home]
    async def move_home(
        self,
        duration: float = 4.0,
        steps:    int   = 200,
    ) -> bool:
        """Move arm to home position, via safe height if over table."""
        home = self.go_home()
        print(f"  [UR5E] Home target deg: {home}")

        ok = await self.move_via_safe_height(
            home, duration=duration, steps=steps, step_callback=None
        )
        if not ok:
            # Last resort — ignore collision checks
            await self.move_to(
                home,
                duration=duration * 1.5,
                steps=steps * 2,
                check_table_collision=False,
            )
        return ok

    def get_status(self) -> dict:
        """Return a snapshot of current arm state for diagnostics."""
        joints = self.get_joint_targets_deg()
        flange = self.get_flange_world_pos()
        base   = self.get_base_world_pos()

        status = {
            "ik_mode":                 self._ik_mode,
            "joint_targets_deg":       list(joints),
            "tool0_world_pos":         flange,
            "base_world_pos":          base,
            "has_table_info":          self._table_info is not None,
            "finger_grasp_height_mm":  self._finger_grasp_height * 1000,
            "flange_to_finger_top":    self._flange_to_finger_base,
            "flange_to_finger_bottom": self._flange_to_fingertips,
            "grip_mode":               self._grip_mode,
        }

        if self._table_info is not None:
            status["over_table"]      = self._is_over_table(flange)
            status["table_surface_z"] = self._get_table_surface_z()
            status["safe_travel_z"]   = self._safe_z_above_table()

        return status