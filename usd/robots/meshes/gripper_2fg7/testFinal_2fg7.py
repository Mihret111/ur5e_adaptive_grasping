"""
gripper_2fg7_controller.py
OnRobot 2FG7 gripper controller for Isaac Sim 5.0
"""

import omni.usd
from pxr import UsdPhysics


class OnRobot2FG7:
    """
    OnRobot 2FG7 parallel gripper controller.

    Joint mapping (outwards config):
        left_finger_joint:  prismatic, axis=X, limits [0, 0.019]
        right_finger_joint: prismatic, axis=X, limits [0, 0.019]

        target = 0.0   -> OPEN  (fingers spread apart)
        target = 0.019 -> CLOSED (fingers together)

    Datasheet specs (v1.9):
        Force range:    20 - 140 N
        Speed range:    16 - 450 mm/s
        Stroke:         38 mm total
        Repeatability:  ±0.1 mm
        Grip time:      ~200 ms (at 4mm, 80N)
    """

    # ══════════════════════════════════════
    # PHYSICAL CONSTANTS (from datasheet)
    # ══════════════════════════════════════

    OPEN_POS = 0.0
    CLOSED_POS = 0.019

    # ══════════════════════════════════════
    # STATES
    # ══════════════════════════════════════

    STATE_IDLE = "IDLE"
    STATE_OPENING = "OPENING"
    STATE_CLOSING = "CLOSING"       # Phase 1: speed-controlled approach
    STATE_HOLDING = "HOLDING"       # Phase 2: force-controlled hold

    # ══════════════════════════════════════
    # TIMING CONSTANTS
    # ══════════════════════════════════════

    GRASP_SETTLE_TICKS = 100
    GRASP_TICK_PERIOD = 0.02        # seconds
    FINE_MOTION_STEPS = 200
    COARSE_MOTION_STEPS = 100
    STALL_THRESHOLD = 0.0005        # m — movement below this = stalled
    STALL_TICKS_REQUIRED = 5        # consecutive stall ticks to confirm contact
    FULLY_CLOSED_THRESHOLD = 0.018  # m — consider fully closed above this

    # ══════════════════════════════════════
    # FORCE PROFILES (for HOLD phase)
    # Determines how hard to squeeze
    # ══════════════════════════════════════

    FORCE_SOFT = {
        "name": "SOFT",
        "max_force": 20.0,         # N — datasheet minimum
        "stiffness": 4e4,          # N/m — 20N / 0.5mm error
        "damping": 300.0,          # N·s/m — prevent oscillation at contact
        "pos_tolerance": 0.0005,   # m — acceptable position error
        "use_case": "eggs, fruit, foam, electronics, glass"
    }

    FORCE_MEDIUM = {
        "name": "MEDIUM",
        "max_force": 80.0,         # N — datasheet midrange
        "stiffness": 4e5,          # N/m — 80N / 0.2mm error
        "damping": 700.0,          # N·s/m — moderately overdamped
        "pos_tolerance": 0.0002,   # m — acceptable position error
        "use_case": "plastic parts, boxes, bottles, general pick-and-place"
    }

    FORCE_HARD = {
        "name": "HARD",
        "max_force": 140.0,        # N — datasheet maximum
        "stiffness": 1.4e6,        # N/m — 140N / 0.1mm error
        "damping": 1000.0,         # N·s/m — slightly overdamped
        "pos_tolerance": 0.0001,   # m — matches datasheet repeatability
        "use_case": "metal parts, CNC pieces, heavy rigid objects"
    }

    # ══════════════════════════════════════
    # SPEED PROFILES (for APPROACH phase)
    # Determines how fast fingers close
    # Independent from force!
    # ══════════════════════════════════════

    SPEED_SLOW = {
        "name": "SLOW",
        "target_speed": 0.016,      # m/s — 16 mm/s (datasheet min)
        "approach_damping": 1250.0, # D = F/v — controls approach speed
        "approach_force": 20.0,     # N — enough to maintain speed
    }

    SPEED_NORMAL = {
        "name": "NORMAL",
        "target_speed": 0.100,      # m/s — 100 mm/s
        "approach_damping": 800.0,
        "approach_force": 80.0,
    }

    SPEED_FAST = {
        "name": "FAST",
        "target_speed": 0.450,      # m/s — 450 mm/s (datasheet max)
        "approach_damping": 311.0,
        "approach_force": 140.0,
    }

    # ══════════════════════════════════════
    # INITIALIZATION
    # ══════════════════════════════════════

    def __init__(self):
        self.left_joint = None
        self.right_joint = None
        self.state = self.STATE_IDLE

        # Active profiles (defaults)
        self.force_profile = self.FORCE_MEDIUM
        self.speed_profile = self.SPEED_NORMAL

        # Stall detection
        self._last_positions = None
        self._stall_count = 0
        self._contact_position = None

        # Find and configure joints
        self._find_joints()
        self._fix_right_joint()

    def _find_joints(self):
        """Auto-discover finger joints in the USD stage."""
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.PrismaticJoint):
                pp = str(prim.GetPath()).lower()
                if "left_finger" in pp:
                    self.left_joint = prim
                elif "right_finger" in pp:
                    self.right_joint = prim

        if self.left_joint:
            print("[2FG7] Left:  " + str(self.left_joint.GetPath()))
        else:
            print("[2FG7] WARNING: Left finger joint not found!")

        if self.right_joint:
            print("[2FG7] Right: " + str(self.right_joint.GetPath()))
        else:
            print("[2FG7] WARNING: Right finger joint not found!")

    def _fix_right_joint(self):
        """Ensure right joint has correct limits."""
        if self.right_joint:
            jp = UsdPhysics.PrismaticJoint(self.right_joint)
            jp.GetLowerLimitAttr().Set(0.0)
            jp.GetUpperLimitAttr().Set(0.019)

    # ══════════════════════════════════════
    # PRIVATE: Drive Parameter Control
    # ══════════════════════════════════════

    def _set_drive_params(self, stiffness, damping, max_force,
                          target_pos, target_vel=0.0):
        """
        Set ALL drive parameters on both finger joints.

        Args:
            stiffness:  Spring constant K (N/m)
            damping:    Damper coefficient D (N·s/m)
            max_force:  Force clamp (N) — from datasheet: 20-140N
            target_pos: Target position (m) — 0.0=open, 0.019=closed
            target_vel: Target velocity (m/s) — 0.0 for position hold
        """
        for joint in [self.left_joint, self.right_joint]:
            if not joint:
                continue
            drive = UsdPhysics.DriveAPI.Get(joint, "linear")
            if not drive:
                continue
            drive.GetTypeAttr().Set("force")
            drive.GetStiffnessAttr().Set(stiffness)
            drive.GetDampingAttr().Set(damping)
            drive.GetMaxForceAttr().Set(max_force)
            drive.GetTargetPositionAttr().Set(target_pos)
            drive.GetTargetVelocityAttr().Set(target_vel)

    def _get_positions(self):
        """
        Get current target positions of both finger joints.

        Returns:
            List of positions [left, right] in meters.

        NOTE: These are TARGET positions from DriveAPI, not actual
              physics positions. For stall detection, actual positions
              from the articulation would be more accurate.
        """
        positions = []
        for joint in [self.left_joint, self.right_joint]:
            if joint:
                drive = UsdPhysics.DriveAPI.Get(joint, "linear")
                if drive:
                    pos = drive.GetTargetPositionAttr().Get()
                    if pos is not None:
                        positions.append(pos)
        return positions

    # ══════════════════════════════════════
    # PUBLIC: Gripper Actions
    # ══════════════════════════════════════

    def open(self, speed_profile=None):
        """
        Open gripper fully.

        Args:
            speed_profile: How fast to open (default: SPEED_NORMAL)
        """
        if speed_profile:
            self.speed_profile = speed_profile

        self.state = self.STATE_OPENING
        self._reset_stall_tracking()

        # Open uses force profile stiffness for firm retraction
        self._set_drive_params(
            stiffness=self.force_profile["stiffness"],
            damping=self.speed_profile["approach_damping"],
            max_force=self.speed_profile["approach_force"],
            target_pos=self.OPEN_POS,
            target_vel=0.0 
        )

        print(f"[2FG7] Opening at {self.speed_profile['name']} speed")

    def close(self, force_profile=None, speed_profile=None):
        """
        Close gripper — Phase 1: speed-controlled approach.

        Fingers move toward each other at controlled speed.
        Call update() every sim step to detect contact and
        auto-transition to Phase 2 (force-controlled hold).

        Args:
            force_profile: How hard to hold (default: FORCE_MEDIUM)
            speed_profile: How fast to approach (default: SPEED_NORMAL)
        """
        if force_profile:
            self.force_profile = force_profile
        if speed_profile:
            self.speed_profile = speed_profile

        self.state = self.STATE_CLOSING
        self._reset_stall_tracking()

        # PHASE 1: Speed-controlled approach
        # Low stiffness — we want velocity control, not position snapping
        # High-ish damping — controls the approach speed
        self._set_drive_params(
            stiffness=100.0,                                # Low K: no position snapping
            damping=self.speed_profile["approach_damping"],  # D controls speed
            max_force=self.speed_profile["approach_force"],  # Enough force for speed
            target_pos=self.CLOSED_POS,                      # Move toward closed
            target_vel=self.speed_profile["target_speed"]    # Desired approach speed
        )

        print(f"[2FG7] Closing: {self.speed_profile['name']} speed → "
              f"{self.force_profile['name']} force ({self.force_profile['max_force']}N)")

    def hold(self):
        """
        Phase 2: Force-controlled hold at contact position.

        Switches from speed control to position+force control.
        Uses active force_profile for stiffness, damping, and max force.
        """
        self.state = self.STATE_HOLDING

        # Get current contact position
        positions = self._get_positions()
        if positions:
            self._contact_position = positions[0]
        else:
            self._contact_position = self.CLOSED_POS

        # Target slightly behind contact to avoid crushing
        safe_target = self._contact_position - self.force_profile["pos_tolerance"]
        safe_target = max(safe_target, self.OPEN_POS)  # Don't go negative

        # PHASE 2: Force-controlled hold
        self._set_drive_params(
            stiffness=self.force_profile["stiffness"],
            damping=self.force_profile["damping"],
            max_force=self.force_profile["max_force"],
            target_pos=safe_target,
            target_vel=0.0              # Stop — hold position
        )

        print(f"[2FG7] Holding at {self._contact_position:.4f}m "
              f"with {self.force_profile['max_force']}N "
              f"({self.force_profile['name']})")

    def release(self, speed_profile=None):
        """Release object and open gripper."""
        self._contact_position = None
        self.open(speed_profile=speed_profile)
        print("[2FG7] Released")

    # ══════════════════════════════════════
    # PUBLIC: Update Loop
    # ══════════════════════════════════════

    def update(self):
        """
        Call every simulation step.

        Phase 1 (CLOSING): Detects object contact via stall detection.
                           Auto-transitions to HOLDING on contact.

        Phase 2 (HOLDING): Monitors grip — could add slip/compression
                           detection here in the future.

        Returns:
            Current state string.
        """
        if self.state == self.STATE_CLOSING:
            self._update_closing()

        elif self.state == self.STATE_OPENING:
            self._update_opening()

        return self.state

    def _update_closing(self):
        """Stall detection during close — detects object contact."""
        current = self._get_positions()
        if not current:
            return

        # Compare with previous tick
        if self._last_positions is not None:
            moved = any(
                abs(current[i] - self._last_positions[i]) > self.STALL_THRESHOLD
                for i in range(len(current))
            )

            if not moved:
                self._stall_count += 1
            else:
                self._stall_count = 0

            # Stalled long enough → object detected
            if self._stall_count > self.STALL_TICKS_REQUIRED:
                self.hold()
                print(f"[2FG7] Object detected! Stalled for "
                      f"{self._stall_count} ticks")
                return

        self._last_positions = current

        # Check if fully closed (no object)
        if all(p >= self.FULLY_CLOSED_THRESHOLD for p in current):
            self.hold()
            print("[2FG7] Fully closed (no object)")

    def _update_opening(self):
        """Check if gripper has finished opening."""
        current = self._get_positions()
        if not current:
            return

        # Check if fully open
        if all(p <= self.OPEN_POS + 0.001 for p in current):
            self.state = self.STATE_IDLE
            self._reset_stall_tracking()
            print("[2FG7] Fully open — IDLE")

    # ══════════════════════════════════════
    # PRIVATE: Helpers
    # ══════════════════════════════════════

    def _reset_stall_tracking(self):
        """Reset all stall detection state."""
        self._last_positions = None
        self._stall_count = 0

    # ══════════════════════════════════════
    # PUBLIC: State Queries
    # ══════════════════════════════════════

    def is_holding(self):
        """True if gripper is holding an object (or fully closed)."""
        return self.state == self.STATE_HOLDING

    def is_idle(self):
        """True if gripper is open and idle."""
        return self.state == self.STATE_IDLE

    def is_closing(self):
        """True if gripper is actively closing."""
        return self.state == self.STATE_CLOSING

    def is_opening(self):
        """True if gripper is actively opening."""
        return self.state == self.STATE_OPENING

    def get_state(self):
        """Return current state string."""
        return self.state

    def get_contact_position(self):
        """Return position where contact was detected (or None)."""
        return self._contact_position

    def get_active_profiles(self):
        """Return current force and speed profiles."""
        return {
            "force": self.force_profile,
            "speed": self.speed_profile
        }


# ══════════════════════════════════════════════
# HOW TO USE
# ══════════════════════════════════════════════
#
# gripper = OnRobot2FG7()
#
# # ─── Simple usage (medium force, normal speed) ───
# gripper.close()
#
# # ─── Fragile object: gentle force, slow approach ───
# gripper.close(
#     force_profile=OnRobot2FG7.FORCE_SOFT,
#     speed_profile=OnRobot2FG7.SPEED_SLOW
# )
#
# # ─── Strong grip, fast approach ───
# gripper.close(
#     force_profile=OnRobot2FG7.FORCE_HARD,
#     speed_profile=OnRobot2FG7.SPEED_FAST
# )
#
# # ─── Gentle force BUT fast approach (independent!) ───
# gripper.close(
#     force_profile=OnRobot2FG7.FORCE_SOFT,
#     speed_profile=OnRobot2FG7.SPEED_FAST
# )
#
# # ─── In your simulation loop ───
# state = gripper.update()  # Call EVERY tick
# if gripper.is_holding():
#     print("Object gripped!")
#     # Move robot to place position...
#
# # ─── Release ───
# gripper.release()
#
# # ─── In sim loop, wait for idle ───
# state = gripper.update()
# if gripper.is_idle():
#     print("Gripper open, ready for next pick")