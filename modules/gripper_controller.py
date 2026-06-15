"""
gripper_controller.py  v3
─────────────────────────
OnRobot 2FG7 Gripper controller — velocity-controlled approach,
force-controlled hold.

v3 changes:
  - Hold targets PAST contact (+2mm) to maintain pressure
  - Minimum simulation force = 40N (physics can't hold at 20N)
  - Startup banner confirms this version is loaded
  - _tick_holding reinforces drive every physics tick
"""

from typing import Optional

import omni.usd
from pxr import UsdPhysics, Sdf, PhysxSchema


class Gripper2FG7:

    # ── Joint travel limits ────────────────────────────────────────
    OPEN_POS   = 0.0     #TODO  do this actually apply to the actual hardware
    CLOSED_POS = 0.019

    # ── State constants ────────────────────────────────────────────
    IDLE    = "IDLE"     #TODO are only these states enough for my case? 
    OPENING = "OPENING"
    CLOSING = "CLOSING"
    HOLDING = "HOLDING"  #TODO  can holding be represented as just one state? safe holding, slipped kind of... don't know check

    # ── Version marker ─────────────────────────────────────────────
    VERSION = "v3-hold-fix" #??

    def __init__(
        self,
        left_joint_path:  str,
        right_joint_path: str,
        config:           Optional[dict] = None,
    ):
        self.config = config or {}
        self.stage  = omni.usd.get_context().get_stage()

        self._verbose = self.config.get("gripper_verbose", False)

        # ── Stall detection ────────────────────────────────────────
        self._stall_threshold      = self.config.get("gripper_stall_threshold",   0.0003)
        self._stall_ticks_required = self.config.get("gripper_stall_ticks",       5)
        self._min_closing_ticks    = self.config.get("gripper_min_closing_ticks", 10)

        # ── Approach: velocity-controlled ──────────────────────────
        self._approach_stiffness = self.config.get("gripper_approach_stiffness", 100.0)
        self._approach_damping   = self.config.get("gripper_approach_damping",   800.0)
        self._approach_speed     = self.config.get("gripper_approach_speed",     0.080)

        # ── Squeeze: builds contact pressure ───────────────────────
        self._squeeze_stiffness = self.config.get("gripper_squeeze_stiffness", 2e5)
        self._squeeze_damping   = self.config.get("gripper_squeeze_damping",   1e4)

        # ── Hold: PAST contact to maintain grip ────────────────────
        # These are critical for grip during lift
        self._hold_stiffness   = self.config.get("gripper_hold_stiffness",   1e5)
        self._hold_damping     = self.config.get("gripper_hold_damping",     2000.0)
        # How far PAST contact to target (positive = more closed)
        self._hold_extra_close = self.config.get("gripper_hold_extra_close", 0.002)
        # Hold at full close force — no reduction
        self._hold_force_ratio = self.config.get("gripper_hold_force_ratio", 1.0)

        # ── Open: fast retraction ──────────────────────────────────
        self._open_stiffness = self.config.get("gripper_open_stiffness", 4e5)
        self._open_damping   = self.config.get("gripper_open_damping",   800.0)

        # ── Force bounds ───────────────────────────────────────────
        # IMPORTANT: 20N is too low for simulation — use 40N minimum
        self._min_force     = max(
            40.0,
            self.config.get("min_grip_force", 40.0)
        )
        self._max_force     = self.config.get("max_grip_force",       140.0)
        self._default_force = max(
            self._min_force,
            self.config.get("gripper_default_force", 80.0)
        )

        # ── Grip geometry ──────────────────────────────────────────
        grip_mode      = self.config.get("grip_mode", "outwards")
        grip_cfg       = self.config.get(f"{grip_mode}_grip", {})
        self._grip_min = grip_cfg.get("grip_range_min", 0.035)
        self._grip_max = grip_cfg.get("grip_range_max", 0.073)

        # ── Expected-contact plausibility check ────────────────────
        # The gripper should not declare contact while still too open
        # for the expected object size
        self._contact_pos_tolerance = self.config.get(
            "gripper_contact_position_tolerance_m", 0.004
        )
        self._min_unexpected_contact_pos = self.config.get(
            "gripper_min_unexpected_contact_joint_pos", 0.0015
        )
        self._max_closing_ticks = self.config.get(
            "gripper_max_closing_ticks", 180
        )

        self._expected_grip_dim_m = None
        self._expected_contact_position = None
        self._close_failure_reason = None

        # ── Per-grasp active forces ────────────────────────────────
        self._close_force = self._default_force
        self._hold_force  = self._default_force * self._hold_force_ratio

        # ── Internal state ─────────────────────────────────────────
        self._state            = self.IDLE
        self._stall_count      = 0
        self._closing_ticks    = 0
        self._last_positions   = None
        self._contact_position = None
        self._squeeze_phase    = False
        self._squeeze_ticks    = 0
        self._had_contact      = False 

        # ── Find joints ────────────────────────────────────────────
        self.left_joint  = self.stage.GetPrimAtPath(Sdf.Path(left_joint_path))
        self.right_joint = self.stage.GetPrimAtPath(Sdf.Path(right_joint_path))

        print(f"  [2FG7] Left  joint: {left_joint_path}")
        print(f"         → valid={self.left_joint.IsValid()}"
              f"  type={self.left_joint.GetTypeName() if self.left_joint.IsValid() else 'N/A'}")
        print(f"  [2FG7] Right joint: {right_joint_path}")
        print(f"         → valid={self.right_joint.IsValid()}"
              f"  type={self.right_joint.GetTypeName() if self.right_joint.IsValid() else 'N/A'}")

        self._ensure_joint_state_api()
        self._setup_joints()

    # ══════════════════════════════════════════════════════════════
    # INTERNAL — Logging
    # ══════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        if self._verbose:
            print(f"  [2FG7] {msg}")

    # ══════════════════════════════════════════════════════════════
    # INTERNAL — Joint access
    # ══════════════════════════════════════════════════════════════

    def _valid_joints(self):
        for joint in [self.left_joint, self.right_joint]:
            if joint and joint.IsValid():
                yield joint

    def _ensure_joint_state_api(self):
        for joint in self._valid_joints():
            try:
                existing = PhysxSchema.JointStateAPI.Get(joint, "linear")
                if not existing:
                    PhysxSchema.JointStateAPI.Apply(joint, "linear")
                    print(f"  [2FG7] Applied JointStateAPI to {joint.GetPath()}")
                else:
                    print(f"  [2FG7] JointStateAPI already on {joint.GetPath()}")
            except Exception as e:
                print(f"  [2FG7] ⚠ JointStateAPI error: {e}")
                try:
                    PhysxSchema.JointStateAPI.Apply(joint, "linear")
                except Exception:
                    pass

    def _setup_joints(self):
        configured = 0
        for joint in self._valid_joints():
            pj = UsdPhysics.PrismaticJoint(joint)
            pj.GetLowerLimitAttr().Set(self.OPEN_POS)
            pj.GetUpperLimitAttr().Set(self.CLOSED_POS)

            drive = UsdPhysics.DriveAPI.Get(joint, "linear")
            if not drive:
                print(f"  [2FG7] DriveAPI not found on {joint.GetPath()} — applying")
                drive = UsdPhysics.DriveAPI.Apply(joint, "linear")

            if drive:
                drive.GetTypeAttr().Set("force")
                drive.GetStiffnessAttr().Set(self._open_stiffness)
                drive.GetDampingAttr().Set(self._open_damping)
                drive.GetMaxForceAttr().Set(self._default_force)
                drive.GetTargetPositionAttr().Set(self.OPEN_POS)
                drive.GetTargetVelocityAttr().Set(0.0)
                #k = drive.GetStiffnessAttr().Get()
                #d = drive.GetDampingAttr().Get()
                #f = drive.GetMaxForceAttr().Get()
                configured += 1
            else:
                print(f"  [2FG7] ❌ Could not apply DriveAPI to {joint.GetPath()}")

        if configured == 0:
            print(f"  [2FG7] ❌ NO joints configured!")
        elif configured < 2:
            print(f"  [2FG7] ⚠️  Only {configured}/2 joints configured")
        else:
            print(f"  [2FG7] Both joints configured successfully")
            print(f"         Approach  K={self._approach_stiffness:.0f}"
                  f"  D={self._approach_damping:.0f}"
                  f"  v={self._approach_speed:.3f}m/s")
            print(f"         Squeeze  K={self._squeeze_stiffness:.0f}"
                  f"  D={self._squeeze_damping:.0f}")
            print(f"         Hold     K={self._hold_stiffness:.0f}"
                  f"  D={self._hold_damping:.0f}"
                  f"  extra={self._hold_extra_close*1000:.1f}mm"
                  f"  ratio={self._hold_force_ratio:.1f}")
            print(f"         Open     K={self._open_stiffness:.0f}"
                  f"  D={self._open_damping:.0f}")
            print(f"         Force    min={self._min_force:.0f}N"
                  f"  max={self._max_force:.0f}N"
                  f"  default={self._default_force:.0f}N")

    def _set_drive(
        self,
        target:     float,
        force:      float,
        stiffness:  float,
        damping:    float,
        velocity:   float = 0.0,
    ):
        for joint in self._valid_joints():
            drive = UsdPhysics.DriveAPI.Get(joint, "linear")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint, "linear")
            if drive:
                drive.GetTypeAttr().Set("force")
                drive.GetTargetPositionAttr().Set(float(target))
                drive.GetMaxForceAttr().Set(float(force))
                drive.GetStiffnessAttr().Set(float(stiffness))
                drive.GetDampingAttr().Set(float(damping))
                drive.GetTargetVelocityAttr().Set(float(velocity))
            else:
                print(f"  [2FG7] ❌ _set_drive: no DriveAPI on {joint.GetPath()}")

    def _get_physx_positions(self) -> list:
        positions = []
        for joint in self._valid_joints():
            try:
                state_api = PhysxSchema.JointStateAPI.Get(joint, "linear")
                if state_api:
                    val = state_api.GetPositionAttr().Get()
                    if val is not None:
                        positions.append(float(val))
                        self._log(f"  physx({joint.GetPath().name})={float(val):.6f}m")
            except Exception:
                pass
        return positions

    def _get_positions(self) -> list:
        physx = self._get_physx_positions()
        if physx:
            return physx

        positions = []
        for joint in self._valid_joints():
            drive = UsdPhysics.DriveAPI.Get(joint, "linear")
            if drive:
                val = drive.GetTargetPositionAttr().Get()
                if val is not None:
                    positions.append(float(val))
                    self._log(f"  drive({joint.GetPath().name})={float(val):.6f}m")
        return positions

    def _clamp_force(self, force: float) -> float:
        return max(self._min_force, min(self._max_force, float(force)))

    def _reset_tracking(self):
        self._stall_count    = 0
        self._closing_ticks  = 0
        self._last_positions = None

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API — Commands
    # ══════════════════════════════════════════════════════════════

    def open(self):
        self._state = self.OPENING
        self._squeeze_phase = False
        self._squeeze_ticks = 0
        self._reset_tracking()
        self._contact_position = None
        self._had_contact = False 

        self._set_drive(
            target=self.OPEN_POS,
            force=self._default_force,
            stiffness=self._open_stiffness,
            damping=self._open_damping,
            velocity=0.0,
        )
        self._log("→ OPENING")
    
    # helper functions for contact position estimation
    def _estimate_contact_position_from_grip_dim(
        self,
        grip_dim_m: Optional[float],
    ) -> Optional[float]:
        """Estimate joint position where the fingers should contact the object.

        The controller's opening estimate is:

            opening = grip_max - 2 * joint_position

        Therefore, for an object of width d:

            joint_contact = (grip_max - d) / 2

        The result is clamped to the physical joint range.
        """
        if grip_dim_m is None:
            return None

        try:
            d = float(grip_dim_m)
        except Exception:
            return None

        expected = (self._grip_max - d) / 2.0

        return max(
            self.OPEN_POS,
            min(self.CLOSED_POS, expected),
        )

    # helper function for plausibility check of the stall
    def _contact_position_is_plausible(self, joint_pos: float) -> bool:
        """Return True if a detected stall is plausible object contact.

        If an expected object width is known, the gripper must have closed
        near the expected contact position before stall can count as contact.

        If no object width is known, use a small generic progress threshold.
        """
        joint_pos = float(joint_pos)

        if self._expected_contact_position is None:
            return joint_pos >= self._min_unexpected_contact_pos

        required = max(
            self.OPEN_POS,
            self._expected_contact_position - self._contact_pos_tolerance,
        )

        return joint_pos >= required

    def close(
        self,
        force_n: Optional[float] = None,
        expected_grip_dim_m: Optional[float] = None,
    ):
        """
        Close gripper — velocity-controlled approach.
        """
        self._close_force = (
            self._clamp_force(force_n) if force_n is not None
            else self._default_force
        )
        self._hold_force = self._clamp_force(
            self._close_force * self._hold_force_ratio
        )

        self._state         = self.CLOSING
        self._squeeze_phase = False
        self._squeeze_ticks = 0
        self._reset_tracking()
        self._contact_position = None
        self._had_contact = False 

# Added behavior to estimate contact position from grip dimension
        self._close_failure_reason = None
        self._expected_grip_dim_m = expected_grip_dim_m
        self._expected_contact_position = (
            self._estimate_contact_position_from_grip_dim(
                expected_grip_dim_m
            )
        )
#
        self._set_drive(
            target=self.CLOSED_POS,
            force=self._close_force,
            stiffness=self._approach_stiffness,
            damping=self._approach_damping,
            velocity=self._approach_speed,
        )
        self._log(
            f"→ CLOSING  F={self._close_force:.0f}N  "
            f"v={self._approach_speed:.3f}m/s  "
            f"expected_dim={self._expected_grip_dim_m}  "
            f"expected_contact={self._expected_contact_position}"
        )

    def hold(self):
        physx_pos = self._get_physx_positions()
        positions = physx_pos if physx_pos else self._get_positions()

        if positions:
            self._contact_position = min(positions)
        else:
            self._contact_position = self.CLOSED_POS

        # ── Safe hold target ───────────────────────────────────────
        hold_target = self._contact_position + self._hold_extra_close
        hold_target = min(hold_target, self.CLOSED_POS)

        # ── Verify hold target is physically reachable ─────────────
        # If contact_position is already at CLOSED_POS, object slipped
        if self._contact_position >= self.CLOSED_POS - 0.0005:
            self._had_contact = False
            self._log("Hold: fingers fully closed — no object")

        self._state = self.HOLDING
        self._set_drive(
            target=hold_target,
            force=self._hold_force,
            stiffness=self._hold_stiffness,
            damping=self._hold_damping,
            velocity=0.0,
        )

        print(
            f"          [2FG7] HOLD: contact={self._contact_position:.5f}m  "
            f"target={hold_target:.5f}m  (+{self._hold_extra_close*1000:.1f}mm)  "
            f"F={self._hold_force:.0f}N  K={self._hold_stiffness:.0f}"
        )

    def release(self):
        self._contact_position = None
        self.open()
        self._log("→ RELEASE")

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API — Tick
    # ══════════════════════════════════════════════════════════════

    def update(self) -> str:
        if self._state == self.CLOSING:
            self._tick_closing()
        elif self._state == self.OPENING:
            self._tick_opening()
        elif self._state == self.HOLDING:
            self._tick_holding()
        return self._state

    # ══════════════════════════════════════════════════════════════
    # INTERNAL — State machine ticks
    # ══════════════════════════════════════════════════════════════

    def _tick_closing(self):
        physx = self._get_physx_positions()
        cur   = physx if physx else self._get_positions()
        if not cur:
            self._closing_ticks += 1
            return

        self._closing_ticks += 1

        # Added timeout logic: If the simulated drive never
        # moves enough, we do not want infinite closing.
        if (
            not self._squeeze_phase
            and self._closing_ticks > self._max_closing_ticks
        ):
            self._had_contact = False
            self._close_failure_reason = "closing_timeout_no_plausible_contact"
            self._log(
                "Closing timeout — no plausible contact detected"
            )
            self.hold()
            return

        # ── Phase 2: SQUEEZE settling ──────────────────────────────
        if self._squeeze_phase:
            self._squeeze_ticks += 1
            settle = self.config.get("gripper_squeeze_ticks", 40)

            if self._squeeze_ticks % 10 == 0:
                self._log(
                    f"Squeeze {self._squeeze_ticks}/{settle}  "
                    f"pos={[f'{p:.5f}' for p in cur]}"
                )

            if self._squeeze_ticks >= settle:
                self._log(f"Squeeze done → HOLDING")
                self.hold()
            return

        # ── Phase 1: wait min ticks ────────────────────────────────
        if self._closing_ticks <= self._min_closing_ticks:
            self._last_positions = cur
            return

        # ── Phase 1: stall detection ───────────────────────────────
        if self._last_positions is not None:
            n     = min(len(cur), len(self._last_positions))
            moved = any(
                abs(cur[i] - self._last_positions[i]) > self._stall_threshold
                for i in range(n)
            )

            if moved:
                self._stall_count = 0
            else:
                self._stall_count += 1

            if self._closing_ticks % 10 == 0:
                self._log(
                    f"Approach tick={self._closing_ticks}  "
                    f"pos={[f'{p:.5f}' for p in cur]}  "
                    f"stall={self._stall_count}/{self._stall_ticks_required}  "
                    f"src={'physx' if physx else 'drive'}"
                )

            if self._stall_count >= self._stall_ticks_required:
                min_pos = min(cur)
                no_obj_limit = self.CLOSED_POS - 0.0005

                # If the fingers are essentially fully closed, no object was captured.
                if min_pos >= no_obj_limit:
                    self._had_contact = False
                    self._close_failure_reason = "fully_closed_no_object"
                    self._log(
                        f"Fully closed at {min_pos:.5f}m — no object"
                    )
                    self.hold()
                    return

                # Important fix:
                # A stall near the open position is not necessarily contact.
                # It may simply mean the simulated drive has not moved enough yet.
                if not self._contact_position_is_plausible(min_pos):
                    self._close_failure_reason = "ignored_implausible_early_stall"

                    self._log(
                        "Ignoring implausible early stall: "
                        f"joint={min_pos:.5f}m, "
                        f"expected_contact={self._expected_contact_position}, "
                        f"expected_dim={self._expected_grip_dim_m}"
                    )

                    # Reset stall counter and keep closing.
                    self._stall_count = 0
                    self._last_positions = cur
                    return

                # Plausible contact.
                self._had_contact = True
                self._close_failure_reason = None
                self._squeeze_phase = True
                self._squeeze_ticks = 0
                self._stall_count = 0
                self._closing_ticks = 0
                self._last_positions = None

                squeeze_target = min(
                    min_pos + 0.001,
                    self.CLOSED_POS,
                )

                self._set_drive(
                    target=squeeze_target,
                    force=self._close_force,
                    stiffness=self._squeeze_stiffness,
                    damping=self._squeeze_damping,
                    velocity=0.0,
                )
                return

            if all(p >= (self.CLOSED_POS - 0.001) for p in cur):
                self._had_contact = False
                self._close_failure_reason = "fully_closed_position_check_no_object"
                self._log("Fully closed (position check) — no object")
                self.hold()
                return

        self._last_positions = cur

    def _tick_holding(self):
        """
        Reinforce hold drive every physics tick.

        Without this, PhysX can relax the joint drive during arm motion
        and the fingers spring open, dropping the object.
        """
        if self._contact_position is None:
            return

        hold_target = self._contact_position + self._hold_extra_close
        hold_target = min(hold_target, self.CLOSED_POS)

        # Re-apply EVERY tick to fight PhysX solver drift
        self._set_drive(
            target=hold_target,
            force=self._hold_force,
            stiffness=self._hold_stiffness,
            damping=self._hold_damping,
            velocity=0.0,
        )

    def _tick_opening(self):
        cur = self._get_positions()
        if not cur:
            return
        if all(p <= 0.001 for p in cur):
            self._state = self.IDLE
            self._squeeze_phase = False
            self._squeeze_ticks = 0
            self._reset_tracking()
            self._log("→ IDLE (fully open)")

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API — Queries
    # ══════════════════════════════════════════════════════════════

    def get_state(self) -> str:
        return self._state

    def is_holding(self) -> bool:
        return self._state == self.HOLDING

    def has_object(self) -> bool:
        if self._state != self.HOLDING:
            return False

        # Primary: did squeeze phase detect contact?
        if not self._had_contact:
            self._log("has_object: no contact during close → False")
            return False

        # Secondary: check if object has since slipped out
        physx_pos = self._get_physx_positions()
        if physx_pos:
            current = min(physx_pos)
            # Use tighter limit — only "no object" if truly slammed shut
            no_obj_limit = self.CLOSED_POS - 0.0005
            if current >= no_obj_limit:
                self._log(
                    f"has_object: fingers at {current:.5f}m "
                    f"(≥ {no_obj_limit:.5f}) → slipped"
                )
                return False

        return True

    def get_contact_position(self) -> Optional[float]:
        return self._contact_position

    def get_active_forces(self) -> dict:
        return {
            "close_force_n": self._close_force,
            "hold_force_n":  self._hold_force,
        }

    def get_opening(self) -> float:
        pos = self._get_positions()
        if not pos:
            return self._grip_max
        avg     = sum(pos) / len(pos)
        opening = self._grip_max - (avg * 2.0)
        return max(self._grip_min, min(self._grip_max, opening))

    def get_diagnostics(self) -> dict:
        physx = self._get_physx_positions()
        return {
            "state":            self._state,
            "version":          self.VERSION,
            "physx_positions":  physx,
            "drive_positions":  self._get_positions(),
            "contact_position": self._contact_position,
            "has_object":       self.has_object(),
            "opening_m":        self.get_opening(),
            "close_force_n":    self._close_force,
            "hold_force_n":     self._hold_force,
            "hold_extra_mm":    self._hold_extra_close * 1000,
            "squeeze_phase":    self._squeeze_phase,
            "squeeze_ticks":    self._squeeze_ticks,
            "stall_count":      self._stall_count,
            "closing_ticks":    self._closing_ticks,
           
            # added for pick failure analysis
            "had_contact": self._had_contact,
            "close_failure_reason": self._close_failure_reason,
            "expected_grip_dim_m": self._expected_grip_dim_m,
            "expected_contact_position_m": self._expected_contact_position,
            "contact_position_tolerance_m": self._contact_pos_tolerance,
        }