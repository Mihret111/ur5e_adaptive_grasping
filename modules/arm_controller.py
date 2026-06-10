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
        Check all configured arm links for table penetration.

        Uses long_link_pairs for dense sampling and point checks
        for remaining collision links.
        """
        if self._table_info is None:
            return False

        table_z   = self._get_table_surface_z()
        margin    = self._collision_check_margin
        n_samples = self.config.get("forearm_sample_count", 5)
        long_link_pairs = self.config.get("long_link_pairs", [
            ["forearm_link",   "wrist_1_link"],
            ["upper_arm_link", "forearm_link"],
        ])

        covered_links = set()

        # ── Dense sample along long link pairs ────────────────
        for pair in long_link_pairs:
            if len(pair) != 2:
                continue
            link_a, link_b = pair[0], pair[1]
            covered_links.add(link_a)

            pos_a, pos_b = self._get_link_endpoints(link_a, link_b)
            if pos_a is None or pos_b is None:
                pos = pos_a or pos_b
                if pos and self._check_point_table_collision(
                        pos, table_z, margin):
                    return True
                continue

            for pt in self._sample_points_along_link(
                    pos_a, pos_b, n_samples):
                if self._check_point_table_collision(pt, table_z, margin):
                    return True

        # ── Point check for remaining links ───────────────────
        for link_name in self.ARM_COLLISION_LINKS:
            if link_name in covered_links:
                continue
            link_pos = self._get_link_world_pos(link_name)
            if link_pos is None:
                continue
            if self._check_point_table_collision(
                    link_pos, table_z, margin):
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

    def _solve_ik_lula(
        self, pos_local, orient=None, seed_deg=None,
    ):
        """
        Single Lula IK attempt.

        Args:
            pos_local: [x, y, z] in robot base_link frame
            orient:    [w, x, y, z] quaternion (default: straight down)
            seed_deg:  6-element seed in degrees

        Returns:
            np.ndarray of 6 joint angles in degrees, or None
        """
        if self._lula_solver is None:
            return None
        pos = np.array(pos_local, dtype=np.float64)
        orient = (
            np.array([0.0, 1.0, 0.0, 0.0])
            if orient is None
            else np.array(orient, dtype=np.float64)
        )
        seed = (
            np.deg2rad(np.array(seed_deg, dtype=np.float64))
            if seed_deg is not None
            else np.deg2rad(self.get_joint_targets_deg())
        )
        try:
            result, ok = self._lula_solver.compute_inverse_kinematics(
                frame_name=self._ee_frame,
                target_position=pos,
                target_orientation=orient,
                warm_start=seed,
            )
            return np.rad2deg(result) if ok else None
        except Exception:
            return None

    def _solve_ik_retries(
        self,
        pos_local,
        orient    = None,
        retries:  int = 12,
        seed_deg  = None,
    ) -> tuple:
        """
        Multi-attempt IK: current joints → home → random seeds.

        PATCH: now returns (joint_angles_deg | None, meta_dict)
               so callers can record IK diagnostics.

        Returns:
            (np.ndarray of 6 joint deg, meta dict)
            OR
            (None, meta dict)   ← all attempts failed
        """
        meta = {
            "attempts":      0,
            "total_retries": retries + 2,   # seed + home + randoms
            "solver":        self._ik_mode,
            "success":       False,
            "seed_used":     None,
        }

        # ── Attempt 1: user-provided or current joint seed ────
        meta["attempts"] += 1
        r = self._solve_ik_lula(pos_local, orient, seed_deg)
        if r is not None:
            meta["success"]   = True
            meta["seed_used"] = "provided"
            return r, meta

        # ── Attempt 2: home position seed ────────────────────
        meta["attempts"] += 1
        home = self.config.get(
            "arm_home_deg", [0.0, 90.0, 90.0, 0.0, 0.0, 0.0])
        r = self._solve_ik_lula(pos_local, orient, home)
        if r is not None:
            meta["success"]   = True
            meta["seed_used"] = "home"
            return r, meta

        # ── Attempts 3..N: random seeds ───────────────────────
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
            # Both faces are equal — try both, randomize order
            candidates = [
                ("face_A", compute_grasp_orientation(obj_yaw)),
                ("face_B", compute_grasp_orientation(
                    obj_yaw + math.pi / 2)),
            ]
            random.shuffle(candidates)
            return candidates

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

        # ── Cylinder/Disc — random orientations ──────────
        if shape in ("Cylinder", "Disc"):
            yaws = [random.uniform(0, 2 * math.pi) for _ in range(3)]
            return [
                (f"random_{i}", compute_grasp_orientation(y))
                for i, y in enumerate(yaws)
            ]

        # ── Sphere — random orientations ─────────────────
        if shape == "Sphere":
            yaws = [random.uniform(0, 2 * math.pi) for _ in range(3)]
            return [
                (f"random_{i}", compute_grasp_orientation(y))
                for i, y in enumerate(yaws)
            ]

        # Unknown shape — random
        yaws = [random.uniform(0, 2 * math.pi) for _ in range(3)]
        return [
            ("random", compute_grasp_orientation(y)) for y in yaws
        ]

    # ══════════════════════════════════════════════════════════
    # IK VALIDATION (collision-free joint configs)
    # ══════════════════════════════════════════════════════════

    def _validate_joints_no_table_collision(
        self, joint_angles_deg, waypoint_name: str = "",
    ) -> bool:
        """
        Temporarily set joints and check for table collision.
        Restores original joints afterward.
        """
        if self._table_info is None:
            return True

        saved = self.get_joint_targets_deg()
        self.set_joint_targets_deg(list(joint_angles_deg))

        collision = self._check_arm_body_table_collision()
        if not collision:
            flange  = self.get_flange_world_pos()
            table_z = self._get_table_surface_z()
            if (self._is_over_table(flange)
                    and flange[2] < table_z + self._collision_check_margin):
                collision = True

        self.set_joint_targets_deg(list(saved))

        if collision:
            self._log(
                f"  [IK] Collision at waypoint '{waypoint_name}'")

        return not collision

    def _validate_path_between_joints(
        self,
        from_deg,
        to_deg,
        n_samples: int = 8,
        label:     str = "",
    ) -> bool:
        """
        Check n_samples interpolated joint configs between from→to
        for table collision. Restores original joints afterward.
        """
        if self._table_info is None:
            return True

        saved    = self.get_joint_targets_deg()
        from_arr = np.array(from_deg, dtype=float)
        to_arr   = np.array(to_deg,   dtype=float)
        collision = False

        for k in range(1, n_samples):
            t      = k / n_samples
            interp = from_arr + t * (to_arr - from_arr)
            self.set_joint_targets_deg(list(interp))

            if self._check_arm_body_table_collision():
                self._log(
                    f"  [IK] Path collision at {label} "
                    f"sample {k}/{n_samples}")
                collision = True
                break

            flange  = self.get_flange_world_pos()
            table_z = self._get_table_surface_z()
            if (self._is_over_table(flange)
                    and flange[2] < table_z + self._collision_check_margin):
                self._log(
                    f"  [IK] Flange path collision at {label} "
                    f"sample {k}/{n_samples}")
                collision = True
                break

        self.set_joint_targets_deg(list(saved))
        return not collision

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
        Solve IK for all pick waypoints.

        PATCH: now returns (results | None, ik_meta dict)

        ik_meta keys:
            orient_name     : str   — chosen orientation label
            orient_quat     : list  — [w, x, y, z]
            orient_yaw_deg  : float
            waypoints_solved: list  — names solved successfully
            waypoints_failed: list  — names that failed
            attempts_per_wp : dict  — {wp_name: attempt_count}
            solver          : str
        """
        ik_meta = {
            "orient_name":      None,
            "orient_quat":      None,
            "orient_yaw_deg":   None,
            "waypoints_solved": [],
            "waypoints_failed": [],
            "attempts_per_wp":  {},
            "solver":           self._ik_mode,
        }

        # ── Orientation candidates ────────────────────────────
        if object_metadata and prim_path:
            candidates = self._compute_grasp_candidates_for_object(
                object_metadata, prim_path)
        else:
            candidates = [
                ("X", compute_grasp_orientation(0.0)),
                ("Y", compute_grasp_orientation(math.pi / 2.0)),
            ]
            random.shuffle(candidates)

        waypoint_order = [
            "safe_above",
            "pre_grasp",
            "grasp",
            "lift",
            "safe_retreat",
            "retract",
        ]

        for orient_name, tool_orient in candidates:
            results     = {}
            prev_joints = None
            all_ok      = True

            # Reset per-candidate tracking
            wp_solved = []
            wp_failed = []
            attempts  = {}

            for name in waypoint_order:
                if name not in flange_targets:
                    continue
                w = flange_targets[name]
                if not isinstance(w, np.ndarray) or len(w) != 3:
                    continue

                local = self._world_to_robot_base(w)

                # ── PATCHED: unpack (joints, meta) tuple ──────
                j, wp_meta = self._solve_ik_retries(
                    local, tool_orient, seed_deg=prev_joints)
                attempts[name] = wp_meta["attempts"]

                if j is None:
                    self._log(
                        f"  [IK] No solution for '{name}' "
                        f"orient={orient_name}")
                    wp_failed.append(name)
                    all_ok = False
                    break

                if not self._validate_joints_no_table_collision(
                        j, waypoint_name=name):
                    wp_failed.append(name)
                    all_ok = False
                    break

                if prev_joints is not None:
                    prev_name = waypoint_order[
                        waypoint_order.index(name) - 1]
                    if not self._validate_path_between_joints(
                            prev_joints, j, n_samples=10,
                            label=f"{prev_name}→{name}"):
                        wp_failed.append(name)
                        all_ok = False
                        break

                results[name] = list(j)
                wp_solved.append(name)
                prev_joints = j

            if all_ok and results:
                # ── Compute yaw from quaternion ───────────────
                yaw_rad = math.atan2(
                    2.0 * (tool_orient[0] * tool_orient[3]
                         + tool_orient[1] * tool_orient[2]),
                    1.0 - 2.0 * (tool_orient[2] ** 2
                                + tool_orient[3] ** 2),
                )
                yaw_deg = math.degrees(yaw_rad)

                # ── Populate ik_meta ──────────────────────────
                ik_meta["orient_name"]      = orient_name
                ik_meta["orient_quat"]      = list(tool_orient)
                ik_meta["orient_yaw_deg"]   = yaw_deg
                ik_meta["waypoints_solved"] = wp_solved
                ik_meta["waypoints_failed"] = wp_failed
                ik_meta["attempts_per_wp"]  = attempts

                print(
                    f"    [IK] ✅ Grasp orientation: {orient_name}  "
                    f"yaw={yaw_deg:.1f}°  "
                    f"quat=[{tool_orient[0]:.3f}, "
                    f"{tool_orient[1]:.3f}, "
                    f"{tool_orient[2]:.3f}, "
                    f"{tool_orient[3]:.3f}]"
                )
                return results, ik_meta

        # ── All candidates failed ─────────────────────────────
        self._log("  [IK] All orientation candidates failed")
        ik_meta["waypoints_failed"] = waypoint_order
        return None, ik_meta
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
        result["joints"] = self._compute_pick_calibration(
            object_world_pos, pan_to_object_deg, targets)
        result["ik_meta"]["used_calibration"] = True
        result["ik_meta"]["solver"]           = "calibration"
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