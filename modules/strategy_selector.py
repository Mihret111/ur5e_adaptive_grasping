# Chooses strategy based on object

def select_strategy(object_profile):
    fragility = object_profile.get("fragility", "low")

    if fragility == "high":
        return {
            "name": "delicate_grasp",
            "force_limit": 5,
            "sequence": ["approach", "close_gentle", "lift", "place"]
        }

    return {
        "name": "normal_grasp",
        "force_limit": 20,
        "sequence": ["approach", "close", "lift", "place"]
    }