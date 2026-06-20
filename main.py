# ═══════════════════════════════════════════════════════════════
# 0. MAKE PROJECT IMPORTABLE
# ═══════════════════════════════════════════════════════════════

from asyncio import runners
import sys
import os
import traceback
import shutil

import random
import numpy as np
from pxr import UsdPhysics, Sdf
import omni.kit.app

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════

import asyncio
import omni.usd
import omni.kit.commands  # needed for loading omni commands

PROJECT_ROOT = os.path.expanduser(
    "~/Desktop/SecondSem/COGAR/ur5e_adaptive_grasping" # Follow this path or change with yours
)
# insert project root into sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"  [path] Injected {PROJECT_ROOT} into sys.path")

_modules_dir = os.path.join(PROJECT_ROOT, "modules")
if not os.path.isdir(_modules_dir):
    raise FileNotFoundError(
        f"Expected modules/ folder at:\n  {_modules_dir}\n"
        f"Check PROJECT_ROOT in main.py"
    )

# ── Clear __pycache__ so stale .pyc files never shadow edits ──
_pycache = os.path.join(_modules_dir, "__pycache__")
if os.path.isdir(_pycache):
    shutil.rmtree(_pycache)
    print(f"  [cache] Cleared {_pycache}")
else:
    print(f"  [cache] No __pycache__ to clear")

# ── Force-reload all modules (Script Editor caches old versions) ──
MODULE_NAMES = [
    "modules",
    "modules.config_loader",
    "modules.sim_utils",
    "modules.event_bus",
    "modules.scene_builder",
    "modules.trial_runner",
    "modules.config_validator",
    "modules.gripper_controller",
    "modules.arm_controller",
    "modules.pick_and_place_executor",
    "modules.preflight",
    "modules.soft_object_observer",
    "modules.micro_lift_validator",
    "modules.retry_policy",
    "modules.force_observer",
]
for mod_name in MODULE_NAMES:
    if mod_name in sys.modules:
        del sys.modules[mod_name]
        print(f"  [evict] {mod_name}")



from modules.config_loader import load_all_configs
from modules.sim_utils     import (
    step_simulation,
    step_simulation_seconds,
    start_simulation,
    stop_simulation,
)
from modules.preflight import preflight_check
from modules.event_bus    import bus         
from modules.trial_runner import TrialRunner
from modules.config_validator import validate_config
# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

CONFIG, TABLE_MATERIALS, TABLE_SEAT_SLOTS = load_all_configs(config_dir=CONFIG_DIR)

print("[config] enable_pre_grasp_test =", CONFIG.get("enable_pre_grasp_test"))
print("[config] move_home_before_pick =", CONFIG.get("move_home_before_pick"))

validate_config(CONFIG, TABLE_MATERIALS, TABLE_SEAT_SLOTS)

##
# ═══════════════════════════════════════════════════════════════
# make same random choices repeat in the simulaition
# ═══════════════════════════════════════════════════════════════
DEBUG_SEED = 7
seed = CONFIG.get("debug_seed", None)
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    print(f"[main] Debug random seed = {seed}")

# -------------------------------------------------------------------------------
# Safety Helper functions for setting drive targets to home before sim starts ----------------- 
# -------------------------------------------------------------------------------
def _safe_set_drive_target(drive_api, value):
    """
    Set drive target position and zero velocity safely.
    """
    target_attr = drive_api.GetTargetPositionAttr()
    velocity_attr = drive_api.GetTargetVelocityAttr()

    if target_attr and target_attr.IsValid():
        target_attr.Set(float(value))

    if velocity_attr and velocity_attr.IsValid():
        velocity_attr.Set(0.0)


def preplay_sync_ur5e_drives(stage, config):
    """
    Synchronize UR5e drive targets before starting physics.

    Real-robot principle:
      controller command should not wake up with stale targets.
    In simulation, we set a known safe target before Play so the old
    grasp/lift target from the previous run cannot wake up.
    """
    robot_paths = config.get("paths", {}).get("robot", {})
    joints_base = robot_paths.get(
        "ur5e_joints_base",
        "/mir/base_link_cabinet/cabinet/ur_mount/ur5e_physics/joints",
    )

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    home = config.get(
        "arm_home_deg",
        [0.0, -90.0, 30.0, -90.0, -90.0, 0.0],
    )

    if isinstance(home, list) and len(home) == 1 and isinstance(home[0], list):
        home = home[0]

    if len(home) != 6:
        raise ValueError(f"arm_home_deg must contain 6 values, got: {home}")

    print("[preplay] Synchronizing UR5e drive targets before PLAY...")

    for joint_name, target_deg in zip(joint_names, home):
        path = f"{joints_base}/{joint_name}"
        prim = stage.GetPrimAtPath(Sdf.Path(path))

        if not prim.IsValid():
            print(f"[preplay] ⚠️ Missing UR5e joint: {path}")
            continue

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            print(f"[preplay] ⚠️ Missing angular DriveAPI: {path}")
            continue

        _safe_set_drive_target(drive, float(target_deg))
        print(f"[preplay]   {joint_name:22s} target={float(target_deg):8.3f} deg")

    print("[preplay] UR5e drive targets synchronized.")


def preplay_sync_gripper_drives(stage, config):
    """
    Synchronize gripper drive targets before starting physics.
    """
    gripper_paths = config.get("paths", {}).get("gripper", {})

    joint_paths = [
        gripper_paths.get(
            "left_finger_joint",
            "/onrobot_2fg7/joints/left_finger_joint",
        ),
        gripper_paths.get(
            "right_finger_joint",
            "/onrobot_2fg7/joints/right_finger_joint",
        ),
    ]

    print("[preplay] Synchronizing 2FG7 drive targets before PLAY...")

    for path in joint_paths:
        prim = stage.GetPrimAtPath(Sdf.Path(path))

        if not prim.IsValid():
            print(f"[preplay] ⚠️ Missing gripper joint: {path}")
            continue

        drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        if not drive:
            print(f"[preplay] ⚠️ Missing linear DriveAPI: {path}")
            continue

        # Open position for this gripper model.
        _safe_set_drive_target(drive, 0.0)
        print(f"[preplay]   {path} target=0.000 m")

    print("[preplay] 2FG7 drive targets synchronized.")
#
# ═══════════════════════════════════════════════════════════════
# 3. MAIN
# ═══════════════════════════════════════════════════════════════

async def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("  ❌ No USD stage open. Load a scene first.")
        return

    if not preflight_check(stage, CONFIG):
        return

    try:
        print("[TRACE] preplay sync before start_simulation")
        preplay_sync_ur5e_drives(stage, CONFIG)
        preplay_sync_gripper_drives(stage, CONFIG)

        print("[TRACE] before start_simulation")
        start_simulation()
        print("[TRACE] after start_simulation")

        runner = TrialRunner(                                                                     # You have to create your TrialRunner
            config          = CONFIG,
            table_materials = TABLE_MATERIALS,
            table_slots     = TABLE_SEAT_SLOTS,
            step_fn         = step_simulation,
            step_seconds_fn = step_simulation_seconds,
        )
        
        await runner.run_all()
        # After SceneBuilder has created /World/Trial, focus the viewport there
        # focus_view_on_trial_root()

        # A forced wait: Keep the scene visible for a few seconds before stopping
        post_run_hold_seconds = float(CONFIG.get("post_run_hold_seconds", 0.0))
        if post_run_hold_seconds > 0:
            print(f"  [main] Holding scene for inspection for {post_run_hold_seconds:.2f}s...")
            await step_simulation_seconds(post_run_hold_seconds)

    except Exception as e:
        print(f"\n  ❌ FATAL: {e}")
        traceback.print_exc()

    finally:
        debug_leave_running = CONFIG.get("leave_sim_running_after_trial", False)

        if debug_leave_running:
            print("[main] Debug mode: leaving simulation running.")
        else:
            stop_simulation()
            print("[main] Simulation stopped.")

        if runner is not None:
            print(
                f"  [main] Done.  "
                f"Attempts: {runner._total_attempts}  "
                f"Successes: {runner._total_successes}"
            )
        else:
            print("  [main] Done before TrialRunner was created.")


# ═══════════════════════════════════════════════════════════════
# 4. LAUNCH
# ═══════════════════════════════════════════════════════════════

asyncio.ensure_future(main())
