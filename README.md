# 🤖 UR5e Adaptive Grasping for Soft & Fragile Objects

A robotic manipulation framework designed to execute delicate, compliant, and force-regulated grasping behaviors on deformable, soft, or fragile objects. Built for **Universal Robots UR5e** equipped with an **OnRobot 2FG7** parallel gripper within **NVIDIA Isaac Sim**.

---

## ✨ Key Features

- **🛡️ Adaptive Force & Compliance Regulation**: Monitors contact forces and material deformation in real-time to prevent crushing delicate objects.
- **🤹 anipulation Primitives**: Supports a versatile library of primitives beyond simple picks, including *soft-push*, *pull*, *slide*, *delicate pick-and-place*, and *micro-lift validation*.
- **🔄 Retry & Recovery Policies**: Automatically detects grasp failures, slip events, or excessive resistance, dynamically adjusting strategy or gripper compliance on subsequent attempts.

---

## 📂 Repository Structure

```text
ur5e_adaptive_grasping/
├── assets/                  # 3D assets, meshes, and USD props
├── config/                  # Modular YAML system configurations
│   ├── environment.yaml     # Physics step, viewport, and execution settings
│   ├── robot.yaml           # UR5e joint definitions, home poses, and kinematics
│   ├── gripper.yaml         # OnRobot 2FG7 linear drive and force limits
│   ├── soft_objects.yaml    # Compliance, stiffness, and fragility profiles
│   └── objects.yaml         # Rigid & deformable target object definitions
├── modules/                 # Core Python engine and architectural modules
│   ├── arm_controller.py    # UR5e articulation control & trajectory execution
│   ├── gripper_controller.py# 2FG7 linear actuation & compliance management
│   ├── force_observer.py    # Contact force sensing & feedback observer
│   ├── soft_object_observer.py # Deformation & material slip monitoring
│   ├── adaptive_safety_monitor.py # Real-time safety interlocks & interrupters
│   ├── trial_runner.py      # Benchmark orchestration & lifecycle manager
│   ├── scene_builder.py     # Programmatic USD stage & object spawner
│   └── ...                  # Strategy selectors, primitive executors, etc
├── usd/                     # Base USD environment scenes (e.g., cabinet + robot)
├── main.py                  # Primary Isaac Sim execution & entry point script
└── architecture.md          # Detailed technical architecture & data flow docs
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **NVIDIA Isaac Sim**
- **Python 3.10+** (bundled within Isaac Sim)

### 2. Running Simulation Trials
This project is designed to be executed directly inside Isaac Sim's embedded Python runtime.

1. Launch **NVIDIA Isaac Sim**.
2. Open the base robot stage:
   * Navigate to `File` -> `Open` -> `usd/robots/mir250_cabinet_ur5e_2fg7_test.usd` *(or your configured USD scene)*.
3. Open the Script Editor:
   * Navigate to `Window` -> `Script Editor`.
4. Load & Run the Engine:
   * Open `main.py` in your favorite editor, verify that `PROJECT_ROOT` matches your local absolute path.
   * Copy the contents of `main.py` into the Isaac Sim Script Editor window.
   * Press **`Ctrl + Enter`** to execute the simulation pipeline.
   5. The last lines of "environment.yaml" should be changed to switch from one mode of opperation to another: 

# run only pick-place:
# phase7_multi_object_batch_enabled: true
# phase8_manipulation_primitives_enabled: false
# phase8_run_timing: standalone

# run primitives only:
# phase7_multi_object_batch_enabled: false
# phase8_manipulation_primitives_enabled: true
# phase8_run_timing: standalone


# run primitives after pick-place:
# phase7_multi_object_batch_enabled: true
# phase8_manipulation_primitives_enabled: true
# phase8_run_timing: after_phase7
   

The runner will automatically evict stale module caches, synchronize drive targets to prevent startup jumps, build randomized object trials on the table surface, execute adaptive grasping primitives, and report aggregate success metrics to the console.

---

## ⚙️ Configuration Overview

All operational parameters are cleanly separated into YAML files inside the `config/` directory:

- **`environment.yaml`**: Configures simulation step rates, debug random seeds (`debug_seed: 7`), post-run holding times, and pre-flight validation toggles.
- **`soft_objects.yaml`**: Defines material thresholds (e.g., max yield force, deformation tolerances, friction coefficients) used by the `StrategySelector` and `AdaptiveSafetyMonitor`.

