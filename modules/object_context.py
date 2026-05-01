# store object info

def build_object_profile(target):
    return {
        "name": target["label"],
        # "fragility": infer_fragility(target),
        # "stiffness": infer_stiffness(target),
        "mass": target["mass"],
        "material": target["material_name"],
        "fragility": "high" if target["mass"] < 0.2 else "low"
    }