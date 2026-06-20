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
    FAILED_CLOSE = "FAILED_CLOSE"  # close resolved, but no physically plausible two-finger grasp

    # ── Version marker ─────────────────────────────────────────────
    VERSION = "v5-balanced-contact-close" # pair-aware contact + balanced hold

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
        self._active_hold_extra_close = self._hold_extra_close
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

        # Contact quality is evaluated from the two-finger pair, not from
        # min(left, right) alone.  This prevents a bad case where one finger
        # stays almost open while the other finger closes deeply, yet the
        # scalar min-position makes the contact look plausible.
        self._max_clean_asymmetry = self.config.get(
            "gripper_max_clean_contact_asymmetry_m", 0.006
        )
        self._max_allowed_asymmetry = self.config.get(
            "gripper_max_allowed_contact_asymmetry_m", 0.012
        )
        # Small object-size / contact-model uncertainty allowance.  A contact
        # that is only slightly above the nominal upper window can still be a
        # real compressed grasp, especially for spheres and cylinders.
        self._compressed_contact_grace = self.config.get(
            "gripper_compressed_contact_grace_m", 0.0015
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
        # Per-finger contact/target memory.
        # The scalar _contact_position is kept for backward-compatible logs,
        # but the actual squeeze/hold motor command uses these per-finger
        # targets so an asymmetric real contact is not collapsed into one
        # common target.
        self._contact_positions = None
        self._squeeze_targets = None
        self._hold_targets = None
        self._hold_target_mode = "scalar_legacy"
        self._contact_quality = None
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

    def _set_drive_targets(
        self,
        targets: list,
        force: float,
        stiffness: float,
        damping: float,
        velocity: float = 0.0,
    ):
        """Set a separate target position for each finger joint.

        The order follows _valid_joints(), i.e. left then right in this
        controller. If the number of targets does not match the number of
        joints, the method falls back to the average target so the gripper
        remains controllable rather than crashing during a trial.
        """
        joints = list(self._valid_joints())

        if not joints:
            return

        if not targets or len(targets) != len(joints):
            if targets:
                target = sum(float(t) for t in targets) / len(targets)
            else:
                target = self.CLOSED_POS

            self._set_drive(
                target=target,
                force=force,
                stiffness=stiffness,
                damping=damping,
                velocity=velocity,
            )
            return

        for joint, target in zip(joints, targets):
            target = max(self.OPEN_POS, min(self.CLOSED_POS, float(target)))

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
                print(f"  [2FG7] ❌ _set_drive_targets: no DriveAPI on {joint.GetPath()}")

    def _clamp_joint_positions(self, positions: list) -> list:
        return [
            max(self.OPEN_POS, min(self.CLOSED_POS, float(p)))
            for p in positions
        ]

    def _make_more_closed_targets(self, positions: list, extra_close: float) -> list:
        positions = self._clamp_joint_positions(positions)
        return [
            min(p + float(extra_close), self.CLOSED_POS)
            for p in positions
        ]

    def _finger_asymmetry(self, positions: Optional[list] = None) -> Optional[float]:
        if positions is None:
            positions = self._get_positions()
        if not positions or len(positions) < 2:
            return None
        return abs(float(positions[0]) - float(positions[1]))

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
        self._contact_positions = None
        self._squeeze_targets = None
        self._hold_targets = None
        self._hold_target_mode = "scalar_legacy"
        self._contact_quality = None
        self._had_contact = False
        self._close_failure_reason = None
        self._expected_grip_dim_m = None
        self._expected_contact_position = None

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
    def _contact_position_window(self):
        """Return acceptable joint-position window for object contact."""
        if self._expected_contact_position is None:
            low = self._min_unexpected_contact_pos
            high = self.CLOSED_POS - 0.0005
            return low, high

        low = max(
            self.OPEN_POS,
            self._expected_contact_position - self._contact_pos_tolerance,
        )

        upper_tol = float(
            self.config.get(
                "gripper_contact_position_upper_tolerance_m",
                self._contact_pos_tolerance,
            )
        )

        high = min(
            self.CLOSED_POS,
            self._expected_contact_position + upper_tol,
        )

        return low, high

    def _contact_position_is_plausible(self, joint_pos: float) -> bool:
        """Legacy scalar plausibility check used only for simple queries.

        The state machine itself uses _contact_quality_from_positions(),
        because object width is related to the average two-finger opening,
        not the minimum joint position alone.
        """
        joint_pos = float(joint_pos)
        low, high = self._contact_position_window()
        return low <= joint_pos <= high

    def _contact_quality_from_positions(self, positions: list) -> dict:
        """Classify two-finger contact quality from the finger pair.

        For external gripping, object width is estimated from the average
        finger joint position:

            opening = grip_max - 2 * mean(q_left, q_right)

        Therefore contact plausibility should use the average position.
        Asymmetry is tracked separately because a very asymmetric closure can
        mean one finger missed, scraped the table, or pushed the object aside.

        A small overshoot beyond the upper edge is allowed as
        ``plausible_compressed_high_edge``. This avoids rejecting contacts that
        are only slightly more closed than the nominal object-size estimate.
        Micro-lift remains the final proof of capture.
        """
        positions = self._clamp_joint_positions(positions or [])

        if not positions:
            return {
                "quality": "missing_positions",
                "plausible": False,
                "clean": False,
            }

        avg = sum(positions) / len(positions)
        min_pos = min(positions)
        max_pos = max(positions)
        asym = max_pos - min_pos
        low, high = self._contact_position_window()
        compressed_high = min(
            self.CLOSED_POS,
            high + float(self._compressed_contact_grace),
        )

        avg_in_window = low <= avg <= high
        avg_in_compressed_grace = high < avg <= compressed_high
        width = max(1e-9, high - low)
        alpha = (avg - low) / width

        if avg < low:
            quality = "too_open_or_early"
            plausible = False
            clean = False
        elif avg_in_window or avg_in_compressed_grace:
            if asym > self._max_allowed_asymmetry:
                quality = "too_asymmetric"
                plausible = False
                clean = False
            elif avg_in_compressed_grace:
                quality = (
                    "plausible_compressed_but_asymmetric"
                    if asym > self._max_clean_asymmetry
                    else "plausible_compressed_high_edge"
                )
                plausible = True
                clean = asym <= self._max_clean_asymmetry
            elif asym > self._max_clean_asymmetry:
                quality = "plausible_but_asymmetric"
                plausible = True
                clean = False
            else:
                quality = "plausible_clean"
                plausible = True
                clean = True
        else:
            quality = "too_closed_or_missed"
            plausible = False
            clean = False

        return {
            "quality": quality,
            "plausible": plausible,
            "clean": clean,
            "positions": positions,
            "avg_position_m": avg,
            "min_position_m": min_pos,
            "max_position_m": max_pos,
            "asymmetry_m": asym,
            "window_m": [low, high],
            "compressed_high_m": compressed_high,
            "compressed_contact_grace_m": float(self._compressed_contact_grace),
            "avg_in_window": avg_in_window,
            "avg_in_compressed_grace": avg_in_compressed_grace,
            "normalized_avg_in_window": alpha,
            "max_clean_asymmetry_m": self._max_clean_asymmetry,
            "max_allowed_asymmetry_m": self._max_allowed_asymmetry,
        }

    def _make_balanced_more_closed_targets(self, positions: list, extra_close: float) -> list:
        """Build stable two-finger squeeze/hold targets without opening a finger.

        Pure per-finger targets preserve asymmetry forever.  A single scalar
        target may open the finger that had already closed farther.  This
        balanced rule closes the lagging finger toward the pair average while
        never commanding any finger to open.
        """
        positions = self._clamp_joint_positions(positions)
        if not positions:
            return [self.CLOSED_POS, self.CLOSED_POS]

        base = min(
            self.CLOSED_POS,
            (sum(positions) / len(positions)) + float(extra_close),
        )

        return [
            min(self.CLOSED_POS, max(float(p), base))
            for p in positions
        ]

    def _fail_close(self, reason: str, positions: Optional[list] = None):
        """Resolve a close attempt as failed without entering HOLDING.

        Earlier versions called hold() even after a failed close.  That made
        logs say HOLDING while has_object=False and could keep squeezing after
        we already knew the grasp was invalid.  FAILED_CLOSE is clearer and
        safer: the executor will open/recover.
        """
        positions = self._clamp_joint_positions(
            positions or self._get_physx_positions() or self._get_positions()
        )

        self._state = self.FAILED_CLOSE
        self._had_contact = False
        self._close_failure_reason = reason
        self._contact_positions = positions if positions else None
        self._contact_position = min(positions) if positions else None
        self._contact_quality = self._contact_quality_from_positions(positions)
        self._squeeze_phase = False
        self._squeeze_ticks = 0
        self._reset_tracking()

        # Stop the fingers where they are; do not build hold pressure.
        if positions:
            self._set_drive_targets(
                targets=positions,
                force=self._default_force,
                stiffness=self._approach_stiffness,
                damping=self._approach_damping,
                velocity=0.0,
            )

        self._log(f"Close failed: {reason}; positions={positions}")

    def close(
        self,
        force_n: Optional[float] = None,
        expected_grip_dim_m: Optional[float] = None,
        hold_extra_close_m: Optional[float] = None,
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
        self._active_hold_extra_close = (
            float(hold_extra_close_m)
            if hold_extra_close_m is not None
            else self._hold_extra_close
        )

        self._state         = self.CLOSING
        self._squeeze_phase = False
        self._squeeze_ticks = 0
        self._reset_tracking()
        self._contact_position = None
        self._contact_positions = None
        self._squeeze_targets = None
        self._hold_targets = None
        self._hold_target_mode = "scalar_legacy"
        self._contact_quality = None
        self._had_contact = False 

        # Object-size-aware expected contact position.
        self._close_failure_reason = None
        self._expected_grip_dim_m = expected_grip_dim_m
        self._expected_contact_position = (
            self._estimate_contact_position_from_grip_dim(
                expected_grip_dim_m
            )
        )
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
            # Preserve the actual two-finger geometry at the moment we enter
            # HOLDING. This is important because real/simulated contact is
            # often asymmetric by a few millimetres.
            self._contact_positions = self._clamp_joint_positions(positions)
            self._contact_position = min(self._contact_positions)
        elif self._contact_positions:
            self._contact_positions = self._clamp_joint_positions(
                self._contact_positions
            )
            self._contact_position = min(self._contact_positions)
        else:
            self._contact_positions = [self.CLOSED_POS, self.CLOSED_POS]
            self._contact_position = self.CLOSED_POS

        # ── Per-finger hold targets ────────────────────────────────
        # Old behavior used one scalar target for both fingers:
        #     min(left, right) + extra
        # That can unintentionally reduce pressure on the finger that had
        # already closed farther. Now each finger holds relative to its own
        # measured contact/settled position.
        self._contact_quality = self._contact_quality_from_positions(
            self._contact_positions
        )
        self._hold_targets = self._make_balanced_more_closed_targets(
            self._contact_positions,
            self._active_hold_extra_close,
        )
        self._hold_target_mode = "balanced_pair_no_opening"

        # ── Verify hold target is physically reachable ─────────────
        # Near-full closure can still be valid for objects close to the
        # lower grip-width limit, so use the object-aware contact window.
        if (
            min(self._contact_positions) >= self.CLOSED_POS - 0.0005
            and not self._contact_quality.get("plausible", False)
        ):
            self._had_contact = False
            self._close_failure_reason = "hold_fully_closed_not_plausible"
            self._log("Hold: fingers fully closed outside pair-aware plausible window — no object")

        self._state = self.HOLDING
        self._set_drive_targets(
            targets=self._hold_targets,
            force=self._hold_force,
            stiffness=self._hold_stiffness,
            damping=self._hold_damping,
            velocity=0.0,
        )

        print(
            f"          [2FG7] HOLD: contact={self._contact_position:.5f}m  "
            f"contact_positions={[round(p, 5) for p in self._contact_positions]}  "
            f"targets={[round(t, 5) for t in self._hold_targets]}  "
            f"(+{self._active_hold_extra_close*1000:.1f}mm per finger)  "
            f"F={self._hold_force:.0f}N  K={self._hold_stiffness:.0f}"
        )

    def release(self):
        self._contact_position = None
        self.open()
        self._log("→ RELEASE")

    def adjust_hold_targets(
        self,
        delta_close_m: float,
        reason: str = "adaptive_effort_regulation",
        force_n: Optional[float] = None,
    ) -> dict:
        """Small closed-loop correction of HOLDING targets.

        Positive ``delta_close_m`` closes the fingers more.
        Negative ``delta_close_m`` opens the fingers slightly.

        This method is intentionally tiny and conservative: it does not replace
        the normal contact/stall logic. It only adjusts the already-established
        HOLDING targets after the gripper has a plausible grasp. This gives the
        higher-level effort controller a safe actuator interface:

            measured effort too low  -> positive delta -> squeeze slightly
            measured effort too high -> negative delta -> relax slightly

        The returned dictionary is JSON-friendly so the trial log can prove
        what the controller actually commanded.
        """
        if self._state != self.HOLDING:
            return {
                "applied": False,
                "reason": "gripper_not_holding",
                "state": self._state,
                "requested_delta_close_m": float(delta_close_m),
            }

        delta = float(delta_close_m)

        if self._hold_targets is None:
            positions = self._get_physx_positions() or self._get_positions()
            if not positions:
                return {
                    "applied": False,
                    "reason": "no_hold_targets_or_positions",
                    "state": self._state,
                    "requested_delta_close_m": delta,
                }
            self._hold_targets = self._clamp_joint_positions(positions)
            self._hold_target_mode = "adaptive_recovered_from_positions"

        before = self._clamp_joint_positions(self._hold_targets)
        after = self._clamp_joint_positions([p + delta for p in before])

        # If everything is already clamped, still report honestly.
        applied_delta = [a - b for a, b in zip(after, before)]
        applied = any(abs(d) > 1.0e-9 for d in applied_delta)

        self._hold_targets = after
        self._hold_target_mode = "adaptive_effort_regulated"

        active_force = self._hold_force if force_n is None else self._clamp_force(force_n)
        self._set_drive_targets(
            targets=self._hold_targets,
            force=active_force,
            stiffness=self._hold_stiffness,
            damping=self._hold_damping,
            velocity=0.0,
        )

        return {
            "applied": bool(applied),
            "reason": str(reason),
            "state": self._state,
            "requested_delta_close_m": delta,
            "applied_delta_close_m": applied_delta,
            "hold_targets_before_m": before,
            "hold_targets_after_m": after,
            "hold_target_mode": self._hold_target_mode,
            "hold_force_n": active_force,
            "stiffness": self._hold_stiffness,
            "damping": self._hold_damping,
        }

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
        """Advance the closing state machine by one physics tick.

        The gripper uses object-size-aware contact plausibility:
        - too early stall  -> ignore and keep closing;
        - plausible stall  -> squeeze then hold;
        - closed outside expected window -> no object.
        """
        physx = self._get_physx_positions()
        cur = physx if physx else self._get_positions()

        self._closing_ticks += 1

        if not cur:
            return

        min_pos = min(cur)
        no_obj_limit = self.CLOSED_POS - 0.0005

        # If the simulated drive never reaches plausible contact, stop honestly.
        if (
            not self._squeeze_phase
            and self._closing_ticks > self._max_closing_ticks
        ):
            self._fail_close("closing_timeout_no_plausible_contact", cur)
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
                self._log("Squeeze done → HOLDING")
                self.hold()
            return

        # ── Phase 1: wait minimum ticks before stall reasoning ─────
        if self._closing_ticks <= self._min_closing_ticks:
            self._last_positions = cur
            return

        if self._last_positions is None:
            self._last_positions = cur
            return

        n = min(len(cur), len(self._last_positions))
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
                f"src={'physx' if physx else 'drive'}  "
                f"window={self._contact_position_window()}"
            )

        quality = self._contact_quality_from_positions(cur)

        # Check full closure every tick. For very small objects, near-full
        # closure can still be plausible, but only if the two-finger pair
        # average is inside the expected window and asymmetry is acceptable.
        if all(p >= (self.CLOSED_POS - 0.001) for p in cur):
            if quality.get("plausible", False):
                self._start_squeeze_from_positions(cur)
            else:
                self._fail_close("fully_closed_position_check_no_object", cur)
            return

        # Stall reasoning happens as soon as enough consecutive still ticks
        # accumulate, not only on logging ticks.
        if self._stall_count >= self._stall_ticks_required:
            # Case 1: plausible object contact from the pair geometry.
            if quality.get("plausible", False):
                self._start_squeeze_from_positions(cur)
                return

            # Case 2: the pair average closed beyond the object window.
            if quality.get("quality") == "too_closed_or_missed" or min_pos >= no_obj_limit:
                self._fail_close("fully_closed_no_object_or_missed_object", cur)
                return

            # Case 3: implausibly early or too asymmetric stall. Keep closing.
            self._close_failure_reason = "ignored_implausible_pair_stall"
            self._log(
                "Ignoring implausible pair stall: "
                f"quality={quality.get('quality')}, "
                f"positions={[f'{p:.5f}' for p in cur]}, "
                f"avg={quality.get('avg_position_m'):.5f}, "
                f"asym={quality.get('asymmetry_m'):.5f}, "
                f"window={quality.get('window_m')}"
            )
            self._stall_count = 0
            self._last_positions = cur
            return

        self._last_positions = cur

    def _start_squeeze_from_positions(self, positions: list):
        """Enter squeeze phase from a plausible contact configuration.

        We keep a per-finger record of the contact configuration. A single
        scalar contact position is still logged for backward compatibility,
        but motor commands use separate targets for left and right fingers.
        """
        positions = self._clamp_joint_positions(positions)
        if not positions:
            positions = [self.CLOSED_POS, self.CLOSED_POS]

        self._contact_positions = positions
        self._contact_position = min(positions)
        self._contact_quality = self._contact_quality_from_positions(positions)
        self._squeeze_targets = self._make_balanced_more_closed_targets(
            positions,
            0.001,
        )
        self._hold_targets = None
        self._hold_target_mode = "balanced_pair_pending_hold"

        self._had_contact = True
        self._close_failure_reason = None
        self._squeeze_phase = True
        self._squeeze_ticks = 0
        self._stall_count = 0
        self._closing_ticks = 0
        self._last_positions = None

        self._set_drive_targets(
            targets=self._squeeze_targets,
            force=self._close_force,
            stiffness=self._squeeze_stiffness,
            damping=self._squeeze_damping,
            velocity=0.0,
        )
        self._log(
            f"Plausible contact at positions "
            f"{[f'{p:.5f}' for p in positions]} "
            f"→ squeeze targets {[f'{t:.5f}' for t in self._squeeze_targets]}"
        )

    def _tick_holding(self):
        """
        Reinforce hold drive every physics tick.

        Without this, PhysX can relax the joint drive during arm motion
        and the fingers spring open, dropping the object.
        """
        if self._hold_targets is None:
            # Backward-compatible fallback: if HOLDING was entered without
            # explicit targets, build per-finger targets from the current
            # measured positions.
            positions = self._get_physx_positions() or self._get_positions()
            if not positions:
                return
            self._contact_positions = self._clamp_joint_positions(positions)
            self._contact_position = min(self._contact_positions)
            self._contact_quality = self._contact_quality_from_positions(
                self._contact_positions
            )
            self._hold_targets = self._make_balanced_more_closed_targets(
                self._contact_positions,
                self._active_hold_extra_close,
            )
            self._hold_target_mode = "balanced_pair_recovered"

        # Re-apply EVERY tick to fight PhysX solver drift, preserving the
        # per-finger targets rather than collapsing them into one scalar.
        self._set_drive_targets(
            targets=self._hold_targets,
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

        if self._contact_quality and not self._contact_quality.get("plausible", False):
            self._log(
                "has_object: stored contact quality is not plausible "
                f"({self._contact_quality.get('quality')}) → False"
            )
            return False

        # Secondary: check if the current pair has become an impossible
        # no-object configuration. Use the same pair-aware logic as closing;
        # do not fall back to the old scalar min-position test.
        physx_pos = self._get_physx_positions()
        if physx_pos:
            pair_quality = self._contact_quality_from_positions(physx_pos)
            if (
                min(physx_pos) >= self.CLOSED_POS - 0.0005
                and not pair_quality.get("plausible", False)
            ):
                self._log(
                    "has_object: current pair is fully closed outside "
                    f"plausible contact quality ({pair_quality.get('quality')}) → no object"
                )
                return False

        return True

    def get_contact_position(self) -> Optional[float]:
        return self._contact_position

    def get_contact_positions(self) -> Optional[list]:
        return self._contact_positions

    def get_hold_targets(self) -> Optional[list]:
        return self._hold_targets

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
            "contact_positions": self._contact_positions,
            "squeeze_targets": self._squeeze_targets,
            "hold_targets": self._hold_targets,
            "hold_target_mode": self._hold_target_mode,
            "finger_position_asymmetry_m": self._finger_asymmetry(physx),
            "hold_target_asymmetry_m": self._finger_asymmetry(self._hold_targets),
            "contact_quality":  self._contact_quality,
            "has_object":       self.has_object(),
            "opening_m":        self.get_opening(),
            "close_force_n":    self._close_force,
            "hold_force_n":     self._hold_force,
            "hold_extra_mm":    self._active_hold_extra_close * 1000,
            "base_hold_extra_mm": self._hold_extra_close * 1000,
            "squeeze_phase":    self._squeeze_phase,
            "squeeze_ticks":    self._squeeze_ticks,
            "stall_count":      self._stall_count,
            "closing_ticks":    self._closing_ticks,
           
            # added for pick failure analysis
            "had_contact": self._had_contact,
            "close_failure_reason": self._close_failure_reason,
            "expected_grip_dim_m": self._expected_grip_dim_m,
            "expected_contact_position_m": self._expected_contact_position,
            "expected_contact_window_m": (
                list(self._contact_position_window())
                if self._expected_contact_position is not None
                else None
            ),
            "contact_position_tolerance_m": self._contact_pos_tolerance,
            "compressed_contact_grace_m": float(self._compressed_contact_grace),
        }