# ═══════════════════════════════════════════════════════════════
# 0. MAKE PROJECT IMPORTABLE
# ═══════════════════════════════════════════════════════════════

import sys
import os
import traceback
import shutil

PROJECT_ROOT = os.path.expanduser(
    "~/Desktop/SecondSem/COGAR/ur5e_adaptive_grasping" # Follow this path or change with yours
)

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

# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
# (
#     CONFIG,
# ) = load_all_configs(config_dir=CONFIG_DIR)
CONFIG, TABLE_MATERIALS, TABLE_SEAT_SLOTS = load_all_configs(config_dir=CONFIG_DIR)
flange_path = CONFIG["paths"]["robot"]["flange_prim"]
init_motion(flange_path)


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
