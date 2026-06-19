"""Offline sanity checks for soft-object phase.

Run from the project root:
    python3 tools/check_soft_object_setup.py

checks that the soft object catalogue is
visible, physically plausible for the 2FG7 range, and that referenced USD assets
exist.
"""
from __future__ import annotations

from pathlib import Path
import sys

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required for this check: {exc}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOFT_CFG = PROJECT_ROOT / "config" / "soft_objects.yaml"

# OnRobot 2FG7 datasheet external grip range used by this project setup.
GRIP_MIN_MM = 35.0
GRIP_MAX_MM = 73.0
FORCE_MIN_N = 20.0
FORCE_MAX_N = 140.0


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _usd_token_report(path: Path) -> dict:
    data = path.read_bytes()[:2_000_000]
    tokens = {
        "usd_crate_magic": b"PXR-USDC" in data,
        "has_deformable_token": b"Deformable" in data or b"deformable" in data,
        "has_physx_deformable_token": b"PhysxDe" in data or b"physxDeformable" in data,
        "has_physics_material_binding": b"material:binding:physics" in data or b":physics" in data,
        "has_youngs_modulus_token": b"youngsMod" in data or b"youngsModulus" in data,
        "has_poisson_token": b"poissons" in data or b"poissonsRatio" in data,
    }
    return tokens


def main() -> int:
    cfg = _load_yaml(SOFT_CFG)
    enabled = bool(cfg.get("soft_object_catalog_enabled", False))
    objects = cfg.get("soft_objects", []) or cfg.get("soft_object_catalog", []) or []

    print("=== B2B soft-object setup sanity check ===")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"soft catalogue enabled: {enabled}")
    print(f"soft commissioning mode: {bool(cfg.get('soft_object_commissioning_mode', False))}")
    print(f"override num objects: {cfg.get('soft_object_override_num_objects')}")
    print(f"spawn local xy: {cfg.get('soft_object_spawn_local_xy_m')}")
    print(f"align bottom to table: {bool(cfg.get('soft_asset_align_bottom_to_table', True))}")
    print(f"soft object entries: {len(objects)}")

    if not objects:
        print("ERROR: no soft objects found. Expected key: soft_objects:")
        return 2

    failures = 0
    warnings = 0

    for obj in objects:
        label = obj.get("label", "<unnamed>")
        print(f"\n--- {label} ---")
        asset_type = obj.get("asset_type", "primitive_proxy")
        asset_path = obj.get("asset_path")
        grip_dim = float(obj.get("grip_dim_mm", -1))
        preferred_force = float(obj.get("preferred_force_n", FORCE_MIN_N))
        max_force = float(obj.get("max_force_n", FORCE_MAX_N))
        mass = float(obj.get("mass_kg", obj.get("mass", -1)))
        hold_extra = float(obj.get("gripper_hold_extra_close_m", obj.get("hold_extra_m", 0.0)) or 0.0)
        asset_scale = obj.get("asset_scale", None)
        ref_height = float(obj.get("asset_reference_height_m", 1.0) or 1.0)
        height_mm = float(obj.get("height_mm", obj.get("grip_dim_mm", 50.0)) or 50.0)
        derived_scale = (height_mm / 1000.0) / max(ref_height, 1e-6)

        print(f"asset_type: {asset_type}")
        print(f"grip_dim_mm: {grip_dim}")
        print(f"mass_kg: {mass}")
        print(f"force: preferred={preferred_force} N, max={max_force} N")
        print(f"hold_extra_m: {hold_extra}")
        if asset_type in ("deformable_usd", "usd_reference"):
            print(f"asset_scale: {asset_scale}  derived_if_missing={derived_scale:.5f}")
            print(f"asset_reference_height_m: {ref_height}")

        if not (GRIP_MIN_MM <= grip_dim <= GRIP_MAX_MM):
            print(f"ERROR: grip_dim_mm outside 2FG7 external range {GRIP_MIN_MM}-{GRIP_MAX_MM} mm")
            failures += 1
        if preferred_force < FORCE_MIN_N:
            print(f"ERROR: preferred_force_n below 2FG7 minimum {FORCE_MIN_N} N")
            failures += 1
        if max_force > FORCE_MAX_N or max_force < FORCE_MIN_N or preferred_force > max_force:
            print("ERROR: inconsistent soft force limits")
            failures += 1
        if mass <= 0:
            print("ERROR: mass must be positive")
            failures += 1
        if hold_extra > 0.002:
            print("WARN: hold_extra_m is large for a delicate soft object")
            warnings += 1
        if asset_type in ("deformable_usd", "usd_reference") and asset_scale is not None:
            try:
                asset_scale_f = float(asset_scale)
                if asset_scale_f <= 0:
                    print("ERROR: asset_scale must be positive")
                    failures += 1
                elif asset_scale_f > 0.2:
                    print("WARN: asset_scale is large; a 1-unit USD cube would become >20 cm")
                    warnings += 1
            except Exception:
                print("ERROR: asset_scale must be numeric if provided")
                failures += 1

        if asset_type in ("deformable_usd", "usd_reference"):
            if not asset_path:
                print("ERROR: USD asset type but no asset_path")
                failures += 1
            else:
                resolved = Path(asset_path)
                if not resolved.is_absolute():
                    resolved = PROJECT_ROOT / resolved
                print(f"resolved_asset_path: {resolved}")
                if not resolved.exists():
                    print("ERROR: referenced USD asset does not exist")
                    failures += 1
                else:
                    tokens = _usd_token_report(resolved)
                    for k, v in tokens.items():
                        print(f"{k}: {v}")
                    if not tokens["usd_crate_magic"]:
                        print("WARN: file does not look like USD crate; Isaac may still load ASCII USDA if extension differs")
                        warnings += 1
                    if not tokens["has_deformable_token"]:
                        print("WARN: no obvious deformable token found in first 2 MB; verify in Isaac GUI")
                        warnings += 1
                    if not tokens["has_physics_material_binding"]:
                        print("WARN: no obvious physics material binding token found; verify material binding in Isaac GUI")
                        warnings += 1

    print("\n=== Summary ===")
    print(f"failures: {failures}")
    print(f"warnings: {warnings}")
    if failures:
        print("RESULT: FAIL — fix the errors before running trials.")
        return 1
    print("RESULT: PASS WITH WARNINGS" if warnings else "RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())