"""
Validates the USD stage and CONFIG before any motion begins.
"""

from pxr import Sdf, UsdPhysics


# ═══════════════════════════════════════════════════════════════
# INTERNAL CHECKS
# ═══════════════════════════════════════════════════════════════

def _resolve_paths(config: dict) -> dict:
    """
    Build a flat {label: prim_path} dict from config["paths"].

    Actual layout (from config dump):
        paths.robot.ur5e_joints_base
        paths.robot.flange_prim
        paths.gripper.joints_base
        paths.gripper.left_finger_link
        paths.gripper.right_finger_link
    """
    robot   = config.get("paths", {}).get("robot",   {})
    gripper = config.get("paths", {}).get("gripper", {})

    return {
        "UR5E joints":    robot.get("ur5e_joints_base"),
        "Flange":         robot.get("flange_prim"),
        "Gripper joints": gripper.get("joints_base"),
        "Left finger":    gripper.get("left_finger_link"),
        "Right finger":   gripper.get("right_finger_link"),
    }


def _check_prims_exist(stage, config: dict) -> bool:
    """
    Verify that all critical USD prims are present on the stage.

    Returns True if every prim is found, False otherwise.
    """
    checks = _resolve_paths(config)

    # Report any paths that could not be resolved from config
    missing_cfg = [name for name, path in checks.items() if not path]
    if missing_cfg:
        print(f"\n  ❌ CONFIG missing path keys for: {', '.join(missing_cfg)}")
        print(f"     robot   block: {config.get('paths', {}).get('robot',   '(none)')}")
        print(f"     gripper block: {config.get('paths', {}).get('gripper', '(none)')}")
        return False

    all_ok = True
    for name, path in checks.items():
        try:
            prim = stage.GetPrimAtPath(Sdf.Path(path))
            if prim.IsValid():
                print(f"  ✅ {name:20s}  {path}")
            else:
                print(f"  ❌ {name:20s}  {path}  — NOT FOUND")
                all_ok = False
        except Exception as e:
            print(f"  ❌ {name:20s}  {path}  — ERROR: {e}")
            all_ok = False

    return all_ok


def _check_joints(stage, config: dict) -> bool:
    """
    Verify that key joints exist and have a DriveAPI.

    Reads stiffness / damping / max-force for logging only.
    Does NOT write or override any drive attributes.

    Returns True if all joints and drives are found, False otherwise.
    """
    robot        = config.get("paths", {}).get("robot",   {})
    gripper      = config.get("paths", {}).get("gripper", {})
    ur5e_base    = robot.get("ur5e_joints_base")
    gripper_base = gripper.get("joints_base")

    if not ur5e_base or not gripper_base:
        print(f"  ❌ Cannot resolve joint base paths:")
        print(f"     ur5e_base    = {ur5e_base}")
        print(f"     gripper_base = {gripper_base}")
        return False

    test_joints = [
        ("shoulder_pan_joint", f"{ur5e_base}/shoulder_pan_joint",    "angular"),
        ("left_finger_joint",  f"{gripper_base}/left_finger_joint",  "linear"),
    ]

    all_ok = True
    for jname, jpath, drive_type in test_joints:
        try:
            prim = stage.GetPrimAtPath(Sdf.Path(jpath))
            if not prim.IsValid():
                print(f"  ❌ {jname:20s}  NOT FOUND at {jpath}")
                all_ok = False
                continue

            drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if not drive:
                print(f"  ⚠️  {jname:20s}  prim OK but NO DriveAPI ({drive_type})")
                all_ok = False
                continue

            k = drive.GetStiffnessAttr().Get()
            d = drive.GetDampingAttr().Get()
            f = drive.GetMaxForceAttr().Get()
            print(f"  ✅ {jname:20s}  K={k}  D={d}  F={f}")

        except Exception as e:
            print(f"  ❌ {jname:20s}  ERROR: {e}")
            all_ok = False

    return all_ok


def _check_config_consistency(config: dict) -> bool:
    """
    Verify that critical CONFIG values are internally consistent.

    Returns True always — issues are warnings only.
    """
    issues = []

    # ── Arm reach ──────────────────────────────────────────────
    arm_min = config.get("arm_min_reach",      0.30)
    arm_max = config.get("arm_max_reach",       0.85)
    safety  = config.get("reach_safety_margin", 0.03)

    if arm_min >= arm_max:
        issues.append(
            f"arm_min_reach ({arm_min}) >= arm_max_reach ({arm_max})"
        )

    eff_min = arm_min + safety
    eff_max = arm_max - safety
    if eff_min >= eff_max:
        issues.append(
            f"Effective reach zone empty: {eff_min:.3f} >= {eff_max:.3f}"
        )

    # ── Gripper forces ─────────────────────────────────────────
    # Force is computed per-object from physics (mass, friction, CoM).
    # These are hardware clamp limits, not a fixed value.
    min_force    = config.get("min_grip_force",         20.0)
    max_force    = config.get("max_grip_force",        140.0)
    hold_ratio   = config.get("gripper_hold_force_ratio", 0.6)
    default_force = config.get("gripper_default_force",  60.0)

    if min_force >= max_force:
        issues.append(
            f"min_grip_force ({min_force}) >= max_grip_force ({max_force})"
        )
    if not (0.0 < hold_ratio <= 1.0):
        issues.append(
            f"gripper_hold_force_ratio ({hold_ratio}) must be in (0, 1]"
        )

    hold_min = min_force  * hold_ratio
    hold_max = max_force  * hold_ratio

    # ── IK seeds ───────────────────────────────────────────────
    ik_pref   = config.get("ik_preference", {})
    num_seeds = ik_pref.get("num_random_seeds", 24)
    if num_seeds < 10:
        issues.append(
            f"num_random_seeds ({num_seeds}) too low "
            f"— may miss elbow-up solutions"
        )

    # ── Report ─────────────────────────────────────────────────
    if issues:
        print(f"\n  ⚠️  CONFIG WARNINGS:")
        for iss in issues:
            print(f"     • {iss}")
    else:
        print(
            f"  ✅ Config consistency OK\n"
            f"     Reach:        {eff_min:.2f}–{eff_max:.2f} m\n"
            f"     Force range:  {min_force:.0f}–{max_force:.0f} N  "
            f"(object-dependent, computed per grasp)\n"
            f"     Hold range:   {hold_min:.0f}–{hold_max:.0f} N  "
            f"(ratio={hold_ratio}  default={default_force:.0f} N)\n"
            f"     IK seeds:     {num_seeds}"
        )

    return True


# ═══════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def preflight_check(stage, config: dict) -> bool:
    """
    Run all pre-flight checks before simulation starts.

    Args:
        stage:  Active USD stage.
        config: Merged CONFIG dict from config_loader.

    Returns:
        True  → safe to start.
        False → abort.
    """
    print("\n" + "═" * 60)
    print("  PRE-FLIGHT CHECK")
    print("═" * 60)

    # ── Show resolved paths for debugging ─────────────────────
    print(f"\n  [paths] Resolving from config['paths']:")
    resolved = _resolve_paths(config)
    for name, path in resolved.items():
        status = path if path else "⚠ NOT FOUND IN CONFIG"
        print(f"     {name:20s} → {status}")
    print()

    prims_ok  = _check_prims_exist(stage, config)
    joints_ok = _check_joints(stage, config)
    config_ok = _check_config_consistency(config)

    all_ok = prims_ok and joints_ok and config_ok

    print("\n" + "═" * 60)
    if not all_ok:
        print("  ❌ Pre-flight FAILED — fix paths in CONFIG before retrying.")
    else:
        print(f"  ✅ All checks passed!")
        print(f"     Starting {config.get('num_trials', '?')} trials...")
    print("═" * 60 + "\n")

    return all_ok