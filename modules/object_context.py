"""Object context / perceptual-schema representation.

This module converts the raw scene/perception target dictionary into the
semantic object profile used by action selection and execution monitoring.
For the COGAR report, this is a perceptual-schema output: it labels the
object with affordance-relevant properties such as compliance and fragility.
"""


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def build_object_profile(target):
    material = target.get("material_name", target.get("material", "unknown"))
    compliance = target.get("compliance", "rigid")
    deformable = _as_bool(target.get("deformable", False))
    fragile = _as_bool(target.get("fragile", False))
    fragility = str(target.get("fragility", "high" if fragile else "low")).lower()
    thin_object = _as_bool(target.get("thin_object", False))

    # Backward-compatible fragility inference for old rigid random objects.
    if not fragile:
        mass = float(target.get("mass", 1.0) or 1.0)
        fragile = mass < 0.2 or material in ("glass", "ceramic")

    if deformable or str(compliance).lower() in ("soft", "deformable", "compliant"):
        affordance = "soft_deformable_pickable"
    elif fragile:
        affordance = "fragile_pickable"
    else:
        affordance = "rigid_pickable"

    return {
        "name": target.get("label", target.get("name", "object")),
        "shape": target.get("shape", "Unknown"),
        "mass": target.get("mass", 0.1),
        "material": material,
        "compliance": compliance,
        "deformable": deformable,
        "fragile": fragile,
        "fragility": fragility,
        "thin_object": thin_object,
        "affordance": affordance,
        "preferred_force_n": target.get("preferred_force_n"),
        "max_force_n": target.get("max_force_n"),
        "gripper_hold_extra_close_m": target.get("gripper_hold_extra_close_m"),
        "micro_lift_speed_scale": target.get("micro_lift_speed_scale"),
        "close_speed_scale": target.get("close_speed_scale"),
        "validation_profile": target.get("validation_profile", "standard"),
        "density_kg_m3": target.get("density_kg_m3"),
        "youngs_modulus_pa": target.get("youngs_modulus_pa"),
        "poissons_ratio": target.get("poissons_ratio"),
        "strategy_hint": target.get("strategy_hint"),
    }