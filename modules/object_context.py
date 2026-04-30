# store object info
def get_object_profile():
    """
    now return a fixed object
    maybe later to be changed to load from config or perception
    """
    return {
        "name": "banana",
        "fragility": "high",
        "stiffness": "low",
        "deformable": True
        # "shape": "cylindrical",
        # "size": "medium",
        # "weight": "light"
    }