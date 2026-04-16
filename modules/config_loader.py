"""
config_loader.py
────────────────
Reads all YAML files and merges into a flat CONFIG dict.

The flat dict is backwards-compatible with all existing module code.
Extracts TABLE_MATERIALS and TABLE_SEAT_SLOTS as separate lists
for direct use by SceneBuilder.

Objects are now generated randomly per trial from shapes/colors/materials
defined in objects.yaml — no fixed catalogue.

Owns:
  - YAML file discovery + loading
  - Config merging strategy
  - List → tuple conversion for vectors/colors
  - Table material + seat slot extraction
"""

import os
import yaml
from typing import Optional

# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _find_config_dir() -> str:
    """
    Find the config/ directory relative to this file.
    Searches project root first, then current working directory.
    """
    module_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(module_dir)
    config_dir  = os.path.join(project_dir, "config")

    if os.path.isdir(config_dir):
        return config_dir

    cwd_config = os.path.join(os.getcwd(), "config")
    if os.path.isdir(cwd_config):
        return cwd_config

    raise FileNotFoundError(
        f"Cannot find config/ directory. Searched:\n"
        f"  {config_dir}\n"
        f"  {cwd_config}"
    )


def _load_yaml(config_dir: str, filename: str) -> dict:
    """
    Load a single YAML file. Returns empty dict if file not found.
    """
    path = os.path.join(config_dir, filename)
    if not os.path.isfile(path):
        print(f"  ⚠️  Config not found: {path}")
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data else {}


def _convert_lists_to_tuples(d: dict):
    """
    Recursively convert numeric lists to tuples in a dict.

    Converts:
      [r, g, b]          → (r, g, b)    color / vec3
      [x, y, z]          → (x, y, z)    position / size
      [min, max]         → (min, max)   range pair

    Keeps as list:
      ["upper_arm_link", ...]   string lists (collision links)
      [{"name": ...}, ...]      list of dicts (shapes, colors, materials)
    """
    for k, v in d.items():
        if isinstance(v, list):
            if v and all(isinstance(x, (int, float)) for x in v):
                d[k] = tuple(v)
            # else: keep as list (string lists, list of dicts)
        elif isinstance(v, dict):
            _convert_lists_to_tuples(v)


def _convert_color_entries(entries: list):
    """
    Convert rgb/color lists to tuples inside a list of dicts.
    Works for colors, materials, shapes, etc.
    """
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # Convert known color/vector keys
        for key in ("rgb", "color"):
            if key in entry and isinstance(entry[key], list):
                if all(isinstance(x, (int, float)) for x in entry[key]):
                    entry[key] = tuple(entry[key])
        # Convert any nested numeric lists
        for key, val in entry.items():
            if isinstance(val, list):
                if val and all(isinstance(x, (int, float)) for x in val):
                    entry[key] = tuple(val)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def load_all_configs(config_dir: Optional[str] = None):
    """
    Load all YAML configs and merge into a flat CONFIG dict.

    Merge strategy:
      environment.yaml  →  flat merge  (room, sim, logging, debug)
      robot.yaml        →  flat merge  (arm params, reach, collision)
      gripper.yaml      →  flat merge  (grip geometry, control, force)
      table.yaml        →  flat merge  excluding "materials"/"seat_slots"
      objects.yaml      →  flat merge  (shapes, colors, grip_range,
                                        mass, materials, spawn constraints)
      paths.yaml        →  nested under CONFIG["paths"]

    Returns:
        CONFIG           dict  — all merged params
        TABLE_MATERIALS  list  — material definitions from table.yaml
        TABLE_SEAT_SLOTS list  — seat slot definitions from table.yaml
    """
    if config_dir is None:
        config_dir = _find_config_dir()

    print(f"\n{'═' * 60}")
    print(f"  LOADING CONFIG from: {config_dir}")
    print(f"{'═' * 60}")

    # ── Load all YAML files ─────────────────────────────────────────
    env     = _load_yaml(config_dir, "environment.yaml")
    robot   = _load_yaml(config_dir, "robot.yaml")
    gripper = _load_yaml(config_dir, "gripper.yaml")
    table   = _load_yaml(config_dir, "table.yaml")
    objects = _load_yaml(config_dir, "objects.yaml")
    paths   = _load_yaml(config_dir, "paths.yaml")

    # ── Build CONFIG ────────────────────────────────────────────────
    CONFIG: dict = {}

    # Flat merge — environment, robot, gripper
    for source in [env, robot, gripper]:
        CONFIG.update(source)

    # Flat merge — table (exclude catalogue keys)
    for k, v in table.items():
        if k not in ("materials", "seat_slots"):
            CONFIG[k] = v

    # Flat merge — objects 
    for k, v in objects.items():
        CONFIG[k] = v

    # Nested — paths stays as CONFIG["paths"]
    CONFIG["paths"] = paths

    # ── Convert numeric lists → tuples ──────────────────────────────
    _convert_lists_to_tuples(CONFIG)

    # ── Extract table catalogues ────────────────────────────────────
    TABLE_MATERIALS  = table.get("materials",  [])
    TABLE_SEAT_SLOTS = table.get("seat_slots", [])

    # Convert colors + vectors inside table materials
    _convert_color_entries(TABLE_MATERIALS)

    # ── Convert colors inside object config lists ───────────────────
    # shapes:   may have numeric range tuples (already handled above)
    # colors:   have rgb lists → need tuple conversion
    # materials: have numeric friction values (already handled)
    if "colors" in CONFIG and isinstance(CONFIG["colors"], list):
        _convert_color_entries(CONFIG["colors"])

    if "shapes" in CONFIG and isinstance(CONFIG["shapes"], list):
        _convert_color_entries(CONFIG["shapes"])

    if ("object_physics_materials" in CONFIG
            and isinstance(CONFIG["object_physics_materials"], list)):
        _convert_color_entries(CONFIG["object_physics_materials"])

    # ── Validate required object generation keys ────────────────────
    required_keys = [
        "shapes", "colors", "grip_range_mm",
        "object_mass", "object_physics_materials",
        "num_objects_range",
    ]
    missing = [k for k in required_keys if k not in CONFIG]
    if missing:
        print(f"  ⚠️  [config_loader] Missing keys from objects.yaml: "
              f"{missing}")

    # ── Log summary ─────────────────────────────────────────────────
    n_shapes = len(CONFIG.get("shapes", []))
    n_colors = len(CONFIG.get("colors", []))
    n_mats   = len(CONFIG.get("object_physics_materials", []))
    grip_cfg = CONFIG.get("grip_range_mm", {})

    print(f"  [config_loader] Loaded {len(CONFIG)} top-level keys")
    print(f"  [config_loader] Objects: {n_shapes} shapes, "
          f"{n_colors} colors, {n_mats} materials")
    print(f"  [config_loader] Grip range: "
          f"{grip_cfg.get('min', '?')}–{grip_cfg.get('max', '?')}mm")
    print(f"  [config_loader] Table: "
          f"{len(TABLE_MATERIALS)} materials, "
          f"{len(TABLE_SEAT_SLOTS)} seat slots")
    print("═" * 60)

    return CONFIG, TABLE_MATERIALS, TABLE_SEAT_SLOTS