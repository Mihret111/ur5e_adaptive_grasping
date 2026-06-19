"""Strategy selection / action-selection schema.

This module is intentionally simple: it chooses a named motor strategy from
object affordances.  The arm/gripper implementation still executes the
primitive, but this gives the architecture a clean cognitive layer.
"""


def select_strategy(object_profile):
    compliance = str(object_profile.get("compliance", "rigid")).lower()
    deformable = bool(object_profile.get("deformable", False))
    fragile = bool(object_profile.get("fragile", False))
    thin_object = bool(object_profile.get("thin_object", False))
    hint = object_profile.get("strategy_hint")

    if hint == "soft_slide":
        return {
            "name": "soft_slide",
            "force_limit": object_profile.get("max_force_n", 35.0),
            "sequence": ["approach", "low_force_contact", "slide", "verify_pose"],
        }

    if deformable or compliance in ("soft", "deformable", "compliant"):
        sequence = [
            "approach",
            "close_gentle",
            "micro_lift_verify",
            "full_lift_if_safe",
            "place_gently",
        ]
        if thin_object:
            sequence.insert(1, "consider_slide_if_pick_unsafe")
        return {
            "name": "delicate_pick",
            "force_limit": object_profile.get("max_force_n", 35.0),
            "sequence": sequence,
        }

    if fragile:
        return {
            "name": "fragile_pick",
            "force_limit": object_profile.get("max_force_n", 60.0),
            "sequence": [
                "approach",
                "close_limited_force",
                "micro_lift_verify",
                "place_gently",
            ],
        }

    return {
        "name": "normal_pick",
        "force_limit": object_profile.get("max_force_n", 140.0),
        "sequence": ["approach", "close", "micro_lift_verify", "lift", "place"],
    }