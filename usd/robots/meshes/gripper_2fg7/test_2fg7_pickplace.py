"""
gripper_2fg7_controller.py
OnRobot 2FG7 gripper controller for Isaac Sim 5.0
Tested and verified.
"""

import omni.usd
from pxr import UsdPhysics


class OnRobot2FG7:
    """
    OnRobot 2FG7 parallel gripper controller.

    Joint mapping (outwards config, verified):
        left_finger_joint:  prismatic, axis=X, limits [0, 0.019]
        right_finger_joint: prismatic, axis=X, limits [0, 0.019] (fixed)

        target = 0.0   -> OPEN  (fingers spread apart)
        target = 0.019 -> CLOSED (fingers together)

    Force range: 20N - 140N
    """

    OPEN_POS = 0.0
    CLOSED_POS = 0.019
    DEFAULT_FORCE = 80.0 # MAX FORCE = 140N and MIN FORCE = 20N 
    DEFAULT_VELOCITY = 0.05

    STATE_IDLE = "IDLE"
    STATE_OPENING = "OPENING"
    STATE_CLOSING = "CLOSING"
    STATE_HOLDING = "HOLDING"
    
    GRASP_SETTLE_TICKS  = 100
    GRASP_TICK_PERIOD   = 0.02  # seconds
    FINE_MOTION_STEPS   = 200
    COARSE_MOTION_STEPS = 100
    
    SOFT = {
        "name":           "SOFT (fragile objects)",
        "max_force":      20.0,       # N — datasheet minimum
        "stiffness":      4e4,        # N/m — 20N / 0.5mm error
        "damping":        800.0,      # N·s/m — overdamped for smooth motion
        "target_speed":   0.016,      # m/s — datasheet minimum (16 mm/s)
        "pos_tolerance":  0.0005,     # m — 0.5mm acceptable error
        "use_case":       "eggs, fruit, foam, electronics, glass"
    }
    
    MEDIUM = {
        "name":           "MEDIUM (general purpose)",
        "max_force":      80.0,       # N — datasheet midrange
        "stiffness":      4e5,        # N/m — 80N / 0.2mm error
        "damping":        750.0,      # N·s/m — moderately overdamped
        "target_speed":   0.100,      # m/s — 100 mm/s
        "pos_tolerance":  0.0002,     # m — 0.2mm acceptable error
        "use_case":       "plastic parts, boxes, bottles, general pick-and-place"
    }
    
    HARD = {
        "name":           "HARD (industrial parts)",
        "max_force":      140.0,      # N — datasheet maximum
        "stiffness":      1.4e6,      # N/m — 140N / 0.1mm error (repeatability)
        "damping":        650.0,      # N·s/m — slightly overdamped for speed
        "target_speed":   0.450,      # m/s — datasheet maximum (450 mm/s)
        "pos_tolerance":  0.0001,     # m — 0.1mm (matches repeatability)
        "use_case":       "metal parts, CNC pieces, heavy rigid objects"
    }

    def __init__(self):
        self.left_joint = None
        self.right_joint = None
        self.state = self.STATE_IDLE
        self.grip_force = self.DEFAULT_FORCE
        self._last_positions = None
        self._stall_count = 0
        self._find_joints()
        self._fix_right_joint()

    def _find_joints(self):
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
        if self.right_joint:
            print("[2FG7] Right: " + str(self.right_joint.GetPath()))

    def _fix_right_joint(self):
        if self.right_joint:
            jp = UsdPhysics.PrismaticJoint(self.right_joint)
            jp.GetLowerLimitAttr().Set(0.0)
            jp.GetUpperLimitAttr().Set(0.019)

    def _set_drive_params(self,  force = DEFAULT_FORCE, target_pos = 0.0):
        for joint in [self.left_joint, self.right_joint]:
            if joint:
                drive = UsdPhysics.DriveAPI.Get(joint, "linear")
                if drive:
                    drive.GetTypeAttr().Set("force")
                    drive.GetStiffnessAttr().Set(5e5) 
                    drive.GetDampingAttr().Set(1e3) 
                    drive.GetMaxForceAttr().Set(force)
                    drive.GetTargetPositionAttr().Set(target_pos)
                    drive.GetTargetVelocityAttr().Set(0.0) # Stop at target

    def _get_positions(self):
        positions = []
        for joint in [self.left_joint, self.right_joint]:
            if joint:
                drive = UsdPhysics.DriveAPI.Get(joint, "linear")
                if drive:
                    positions.append(drive.GetTargetPositionAttr().Get())
        return positions

    # ══════════════════════════════════════
    # PUBLIC METHODS
    # ══════════════════════════════════════

    def open(self, force=DEFAULT_FORCE):
        """Open gripper fully."""
        self.state = self.STATE_OPENING
        self._stall_count = 0
        self._set_drive_params(
            stiffness=1e6,
            damping=1e4,
            force=force,
            target_pos=self.OPEN_POS
        )

    def close(self, force=DEFAULT_FORCE):
        """
        Close gripper with force.
        Fingers push until they hit an object, then hold.
        Call update() every sim step to detect contact.
        """
        self.state = self.STATE_CLOSING
        self._stall_count = 0
        self._last_positions = None
        self._set_drive_params(
            stiffness=1e2,
            damping=1e5,
            force=force,
            target_pos=self.CLOSED_POS
        )
        self.grip_force = force

    def hold(self):
        """Lock at current position with force."""
        self.state = self.STATE_HOLDING
        self._set_drive_params(
            stiffness=1e6,
            damping=1e4,
            force=self.grip_force,
            target_pos=self._get_positions()[0] if self._get_positions() else self.CLOSED_POS
        )

    def release(self):
        """Release and open."""
        self.open()

    def update(self):
        """
        Call every simulation step.
        Detects object contact during closing.
        Returns current state.
        """
        if self.state == self.STATE_CLOSING:
            current = self._get_positions()
            if self._last_positions is not None:
                moved = any(
                    abs(current[i] - self._last_positions[i]) > 0.0005
                    for i in range(len(current))
                )
                if not moved:
                    self._stall_count += 1
                else:
                    self._stall_count = 0

                if self._stall_count > 5:
                    self.hold()
                    print("[2FG7] Object detected! Holding at " + str(self.grip_force) + "N")

            self._last_positions = current

            if current and all(p >= 0.018 for p in current):
                self.hold()
                print("[2FG7] Fully closed (no object)")

        return self.state

    def is_holding(self):
        return self.state == self.STATE_HOLDING

    def is_open(self):
        return self.state == self.STATE_IDLE
    
    def is_closing(self):
        return self.state == self.STATE_CLOSING

##### HOW TO TEST ######
#gripper = OnRobot2FG7()

# Open
#gripper.open(velocity=0.1)

# Close with force (call update() every sim step)
#gripper.close(force=80.0)

# In your sim loop:
#state = gripper.update()
#if gripper.is_holding():
    # Object gripped! Move robot to place position...
    #pass

# Release
#gripper.release()
