# ═══════════════════════════════════════════════════════════════
# 0. MAKE PROJECT IMPORTABLE
# ═══════════════════════════════════════════════════════════════

import sys
import os
import traceback
import shutil

import random
import numpy as np


PROJECT_ROOT = os.path.expanduser(
    "~/Desktop/SecondSem/COGAR/ur5e_adaptive_grasping" # Follow this path or change with yours
)
import omni.kit.app

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
    "modules.preflight",
    "modules.event_bus",
    "modules.scene_builder",
    "modules.trial_runner",
    "modules.object_context",
    "modules.strategy_selector",
    "modules.execution_manager",
    "modules.primitive_library",
    "modules.motion_interface",
    "modules.config_validator",
    "modules.gripper_controller",
    "modules.arm_controller",
    "modules.pick_and_place_executor"
]
for mod_name in MODULE_NAMES:
    if mod_name in sys.modules:
        del sys.modules[mod_name]
        print(f"  [evict] {mod_name}")

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════

import asyncio
import omni.usd
import omni.kit.commands  # needed for loading omni commands


from modules.config_loader import load_all_configs
from modules.sim_utils     import (
    step_simulation,
    step_simulation_seconds,
    start_simulation,
    stop_simulation,
)
from modules.preflight    import preflight_check
from modules.event_bus    import bus         
from modules.trial_runner import TrialRunner
from modules.primitive_library import init_motion
from modules.config_validator import validate_config
# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

CONFIG, TABLE_MATERIALS, TABLE_SEAT_SLOTS = load_all_configs(config_dir=CONFIG_DIR)

print("[config] enable_pre_grasp_test =", CONFIG.get("enable_pre_grasp_test"))
print("[config] move_home_before_pick =", CONFIG.get("move_home_before_pick"))

validate_config(CONFIG, TABLE_MATERIALS, TABLE_SEAT_SLOTS)

flange_path = CONFIG["paths"]["robot"]["flange_prim"]
init_motion(flange_path)

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

def focus_view_on_trial_root():
    """
    Focus the Isaac Sim viewport on the generated trial scene.
    (equivalent to selecting /World/Trial and pressing F in the UI)
    """
    try:
        omni.kit.commands.execute(
            "FramePrimsCommand",
            prim_paths=["/World/Trial"]
        )
        print("  [camera] Focused viewport on /World/Trial")
    except Exception as e:
        print(f"  [camera] Warning: could not focus viewport: {e}")

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
        start_simulation()
        print("  [main] Initial settle: stepping 60 frames...")
        await step_simulation(60)

        runner = TrialRunner(                                                                     # You have to create your TrialRunner
            config          = CONFIG,
            table_materials = TABLE_MATERIALS,
            table_slots     = TABLE_SEAT_SLOTS,
            step_fn         = step_simulation,
            step_seconds_fn = step_simulation_seconds,
        )
        await runner.run_all()

        # After SceneBuilder has created /World/Trial, focus the viewport there
        focus_view_on_trial_root()

        # Keep the scene visible for a few seconds before stopping
        print("  [main] Holding scene for inspection...")
        await step_simulation_seconds(5.0)

    except Exception as e:
        print(f"\n  ❌ FATAL: {e}")
        traceback.print_exc()

    finally:
        # stop_simulation()
        DEBUG_LEAVE_RUNNING = True

        if DEBUG_LEAVE_RUNNING:
            print("[main] Debug mode: leaving simulation running.")
        else:
            stop_simulation()
        
        print(
            f"  [main] Done.  "
            f"Attempts: {runner._total_attempts}  "
            f"Successes: {runner._total_successes}"
        )


# ═══════════════════════════════════════════════════════════════
# 4. LAUNCH
# ═══════════════════════════════════════════════════════════════

asyncio.ensure_future(main())
