# modules/pick_and_place_executor.py

from modules.arm_controller import UR5EController
from modules.gripper_controller import Gripper2FG7
from modules.micro_lift_validator import MicroLiftValidator
from modules.soft_object_observer import SoftObjectObserver
from modules.retry_policy import RetryPolicy
from modules.force_observer import ArticulationEffortForceObserver
from modules.adaptive_safety_monitor import AdaptiveSafetyMonitor
from modules.trial_diagnostics import (
    compact_pick_result,
    grip_feasibility,
    json_safe,
    make_event,
    pose_snapshot,
    selected_config_snapshot,
)
import omni.kit.app
from pxr import UsdGeom, Sdf, Usd    # import required usd modules

class PickAndPlaceExecutor:
    """
    Minimal Isaac-native generic grasp executor.

    First goal:
      - initialize arm controller
      - initialize 2FG7 controller
      - compute pick waypoints
      - execute a generic pick sequence

    TODO object-aware adaptive grasping. This is just a generic baseline behavior
    """

    def __init__(self, config: dict):
        self.config = config

        print("\n[PickAndPlaceExecutor] Initializing Isaac-native controllers...")

        self.arm = UR5EController(config)

        gripper_paths = config.get("paths", {}).get("gripper", {})

        left_joint = gripper_paths.get(
            "left_finger_joint",
            "/onrobot_2fg7/joints/left_finger_joint",
        )
        right_joint = gripper_paths.get(
            "right_finger_joint",
            "/onrobot_2fg7/joints/right_finger_joint",
        )

        self.gripper = Gripper2FG7(
            left_joint_path=left_joint,
            right_joint_path=right_joint,
            config=config,
        )
        self.micro_lift_validator = MicroLiftValidator(config)
        self.soft_observer = SoftObjectObserver(config)
        self.retry_policy = RetryPolicy(config)
        self.force_observer = ArticulationEffortForceObserver(config)
        self.safety_monitor = AdaptiveSafetyMonitor(config)
        # self.move_home_before_pick = config.get("move_home_before_pick", True)   #
        print("[PickAndPlaceExecutor] Ready.")

    async def _step_gripper_for_seconds(self, seconds: float):
        """
        Advance the gripper state machine while Isaac physics steps.
        """
        app = omni.kit.app.get_app()
        frames = max(1, int(seconds * 60))

        for _ in range(frames):
            self.gripper.update()
            await app.next_update_async()

    async def open_gripper(self):
        print("[Executor] Opening gripper...")
        self.gripper.open()
        await self._step_gripper_for_seconds(1.0)
        print(f"[Executor] Gripper state: {self.gripper.get_state()}")

    async def close_gripper(
        self,
        force_n=None,
        expected_grip_dim_m=None,
        hold_settle_extra_s: float = 0.0,
        hold_extra_close_m=None,
        target=None,
        adaptive_effort_target_override_sim=None,
    ):
        """Close gripper with optional object-size-aware contact expectation.

        expected_grip_dim_m is the estimated object width/diameter along the
        gripper closing direction.

        hold_settle_extra_s is added by RetryPolicy when a retry needs more
        contact stabilization before validation.
        """
        print("[Executor] Closing gripper...")

        force_before_close = None
        if hasattr(self, "force_observer"):
            # Capture an open/no-contact-ish baseline immediately before the close.
            # This does not command anything; it only makes the effort logs interpretable.
            try:
                force_before_close = self.force_observer.set_baseline_from_current(
                    stage_name="before_close_command_baseline"
                )
            except Exception as e:
                force_before_close = {
                    "available": False,
                    "stage": "before_close_command_baseline",
                    "reason": f"force_observer_exception: {e}",
                }

        self.gripper.close(
            force_n=force_n,
            expected_grip_dim_m=expected_grip_dim_m,
            hold_extra_close_m=hold_extra_close_m,
        )

        # Wait until the gripper leaves CLOSING, or until timeout.
        # This avoids validating while the gripper is still moving.
        close_resolution = await self._wait_for_gripper_close_resolution(
            float(self.config.get("gripper_close_wait_timeout_s", 6.0))
        )

        # attempt_log["gripper_close_resolution"] = json_safe(close_resolution)

        print(f"[Executor] Gripper close resolved as: {close_resolution}")

        hold_settle_time = float(
            self.config.get("post_close_hold_seconds", 1.0)
        ) + float(
            hold_settle_extra_s
        )

        await self._step_gripper_for_seconds(hold_settle_time)

        adaptive_effort_regulation = None
        if bool(self.config.get("adaptive_effort_control_enabled", False)):
            adaptive_effort_regulation = await self._regulate_gripper_effort_after_close(
                target=target,
                target_effort_override_sim=adaptive_effort_target_override_sim,
            )

        force_after_hold = None
        if hasattr(self, "force_observer"):
            try:
                force_after_hold = self.force_observer.observe(stage_name="after_close_hold_settle")
            except Exception as e:
                force_after_hold = {
                    "available": False,
                    "stage": "after_close_hold_settle",
                    "reason": f"force_observer_exception: {e}",
                }

        close_resolution["force_observation_before_close"] = force_before_close
        close_resolution["adaptive_effort_regulation"] = adaptive_effort_regulation
        close_resolution["force_observation_after_hold_settle"] = force_after_hold
        close_resolution["force_observer_diagnostics"] = (
            self.force_observer.get_diagnostics()
            if hasattr(self, "force_observer")
            else None
        )

        print(f"[Executor] Gripper state: {self.gripper.get_state()}")
        print(f"[Executor] Has object: {self.gripper.has_object()}")
        if force_after_hold and force_after_hold.get("available"):
            print(
                "[Executor] ForceObserver after hold: "
                f"source={force_after_hold.get('force_source')} "
                f"grip_effort_sim={force_after_hold.get('grip_effort_sim'):.4f} "
                f"contact_like={force_after_hold.get('contact_like_effort')}"
            )
        
        return close_resolution

    async def _regulate_gripper_effort_after_close(self, target=None, target_effort_override_sim=None) -> dict:
        """Scalar measured-effort admittance for the 2FG7 gripper.

        Phase 2 used a deadband rule: if effort was too low, close a fixed
        amount; if effort was too high, open a fixed amount. That was useful as
        a first safe feedback regulator, but it was only admittance-style.

        This Phase 2.5 loop implements an explicit 1D virtual admittance model
        on the gripper closing coordinate x [m]:

            M*x_ddot + D*x_dot + K*x = G*(F_target - F_measured)

        where F_measured is the measured prismatic finger-joint effort from the
        ForceObserver. The admittance state x is a small correction around the
        already-established HOLDING target. Positive x means "close slightly";
        negative x means "relax/open slightly".

        The effort units are Isaac simulation joint-effort units, not calibrated
        real Newtons yet. This loop is therefore an actual scalar admittance
        controller with a measured sim-side effort input, not a claim of direct
        fingertip ContactSensor force.
        """
        app = omni.kit.app.get_app()
        # Keep the object target separate from the effort target.  In Phase 3.0
        # this function accidentally reused the name ``target`` for the scalar
        # effort setpoint, so the safety monitor tried to observe a float rather
        # than the soft object.  That made soft_observation_for_safety null.
        # target_obj is the spawned object dictionary; target_effort is the scalar
        # gripper effort setpoint.
        target_obj = target

        controller_mode = str(
            self.config.get("adaptive_effort_controller_mode", "scalar_admittance")
        )

        summary = {
            "enabled": True,
            "controller": "scalar_gripper_joint_effort_admittance",
            "controller_mode": controller_mode,
            "force_source": "measured_joint_effort_sim",
            "units": "measured_prismatic_joint_effort_N_not_fingertip_calibrated",
            "formal_model": "M*x_ddot + D*x_dot + K*x = G*(F_target - F_measured)",
            "state_meaning": "x is a small correction of gripper HOLDING targets; positive closes, negative opens",
            "safety_layer": "AdaptiveSafetyMonitor combines measured effort with soft-object deformation/motion observations",
            "ran": False,
            "trace": [],
        }

        if not hasattr(self, "force_observer"):
            summary["reason"] = "force_observer_missing"
            return summary

        if not hasattr(self.gripper, "adjust_hold_targets"):
            summary["reason"] = "gripper_adjust_hold_targets_missing"
            return summary

        if self.gripper.get_state() != self.gripper.HOLDING:
            summary["reason"] = "gripper_not_holding"
            summary["state"] = self.gripper.get_state()
            return summary

        target_effort = float(
            target_effort_override_sim
            if target_effort_override_sim is not None
            else self.config.get(
                "adaptive_effort_target_sim",
                self.config.get("force_observer_target_effort_sim", 0.35),
            )
        )
        band = float(
            self.config.get(
                "adaptive_effort_target_band_sim",
                self.config.get("force_observer_target_band_sim", 0.06),
            )
        )
        max_effort = float(
            self.config.get(
                "adaptive_effort_max_sim",
                self.config.get("force_observer_max_effort_sim", 1.20),
            )
        )

        # Small physical limits around the pre-existing hold target. These keep
        # the admittance layer from destroying a grasp that the geometric close
        # stage already made plausible.
        max_total_close = float(self.config.get("adaptive_effort_max_extra_close_m", 0.00150))
        max_total_open = float(self.config.get("adaptive_effort_max_relax_open_m", 0.00100))
        max_delta_per_update = float(self.config.get("adaptive_admittance_max_delta_per_update_m", 0.00010))
        max_velocity = float(self.config.get("adaptive_admittance_max_velocity_mps", 0.0015))

        # Virtual admittance parameters. They are intentionally conservative and
        # unit-labelled as simulation gains because the effort signal is not yet
        # calibrated to real 2FG7 Newtons.
        virtual_mass = float(self.config.get("adaptive_admittance_virtual_mass", 1.0))
        virtual_damping = float(self.config.get("adaptive_admittance_virtual_damping", 12.0))
        virtual_stiffness = float(self.config.get("adaptive_admittance_virtual_stiffness", 35.0))
        effort_to_accel_gain = float(self.config.get("adaptive_admittance_effort_to_accel_gain", 0.04))
        deadband = float(self.config.get("adaptive_admittance_deadband_sim", min(0.03, band)))
        dt = float(self.config.get("adaptive_admittance_dt_s", 1.0 / 60.0))

        sample_stride = max(1, int(self.config.get("adaptive_effort_sample_stride_frames", 5)))
        max_frames = max(1, int(self.config.get("adaptive_effort_max_frames", 90)))
        required_stable = max(1, int(self.config.get("adaptive_effort_required_stable_samples", 3)))

        safety_enabled = bool(self.config.get("adaptive_safety_monitor_enabled", True))
        safety_abort_on_unsafe = bool(self.config.get("adaptive_safety_abort_on_unsafe", False))
        safety_soft_sample_stride = max(1, int(self.config.get("adaptive_safety_soft_sample_stride", sample_stride)))

        summary.update({
            "ran": True,
            "target_effort_sim": target_effort,
            "target_effort_override_sim": target_effort_override_sim,
            "target_band_sim": band,
            "deadband_sim": deadband,
            "max_effort_sim": max_effort,
            "max_total_close_m": max_total_close,
            "max_total_open_m": max_total_open,
            "max_delta_per_update_m": max_delta_per_update,
            "max_velocity_mps": max_velocity,
            "virtual_mass": virtual_mass,
            "virtual_damping": virtual_damping,
            "virtual_stiffness": virtual_stiffness,
            "effort_to_accel_gain": effort_to_accel_gain,
            "dt_s": dt,
            "sample_stride_frames": sample_stride,
            "max_frames": max_frames,
            "required_stable_samples": required_stable,
            "safety_monitor_enabled": safety_enabled,
            "safety_abort_on_unsafe": safety_abort_on_unsafe,
            "safety_soft_sample_stride": safety_soft_sample_stride,
            "target_obj_available_for_safety": target_obj is not None,
            "target_obj_label_for_safety": (target_obj.get("label") if isinstance(target_obj, dict) else None),
            "target_obj_prim_path_for_safety": (target_obj.get("prim_path") if isinstance(target_obj, dict) else None),
        })

        # Admittance state. x=0 means keep the original hold target.
        x = 0.0
        x_dot = 0.0
        applied_x = 0.0
        stable_count = 0
        final_reason = "max_frames_reached"

        for frame in range(max_frames):
            self.gripper.update()
            await app.next_update_async()

            if (frame % sample_stride) != 0:
                continue

            try:
                obs = self.force_observer.observe(
                    stage_name=f"adaptive_admittance_frame_{frame + 1}"
                )
            except Exception as e:
                obs = {
                    "available": False,
                    "reason": f"force_observer_exception: {e}",
                }

            action = "hold_no_observation"
            adjust = {"applied": False, "reason": action, "requested_delta_close_m": 0.0}
            delta_to_apply = 0.0
            effort = None
            effort_error = None
            force_input = 0.0
            x_ddot = 0.0
            soft_obs_for_safety = None
            safety_assessment = None

            if obs.get("available"):
                effort = float(obs.get("grip_effort_sim", 0.0) or 0.0)
                effort_error = target_effort - effort

                # Safety/perception layer: combine force-like effort with live
                # deformable object shape.  This is the COGAR reactive inhibitor:
                # it can request relax/open before the admittance model squeezes
                # a soft object too much.  It is deliberately conservative and
                # monitor-first; abort is optional and off by default.
                if safety_enabled and hasattr(self, "safety_monitor"):
                    if target_obj is not None and ((frame % safety_soft_sample_stride) == 0):
                        try:
                            soft_obs_for_safety = self._observe_target(
                                target_obj,
                                stage_name=f"adaptive_safety_frame_{frame + 1}",
                            )
                        except Exception as e:
                            soft_obs_for_safety = {
                                "available": False,
                                "reason": f"soft_observer_exception: {e}",
                            }
                    try:
                        safety_assessment = self.safety_monitor.assess(
                            soft_obs=soft_obs_for_safety,
                            effort_obs=obs,
                            context={
                                "frame": frame + 1,
                                "target_effort_sim": target_effort,
                                "max_effort_sim": max_effort,
                                "controller_phase": "scalar_admittance",
                            },
                        )
                    except Exception as e:
                        safety_assessment = {
                            "available": False,
                            "safe_to_continue": True,
                            "recommended_action": "continue",
                            "reason": f"safety_monitor_exception: {e}",
                        }

                safety_action = (safety_assessment or {}).get("recommended_action")
                safety_unsafe = bool((safety_assessment or {}).get("unsafe", False))

                # Safety override: if measured effort is above the maximum safe
                # value, or the soft-object safety monitor asks for relaxation,
                # force a small opening independent of the virtual dynamics.
                # This is the reactive inhibition layer.
                if effort >= max_effort or safety_action in ("relax_open", "relax_open_deformation", "relax_open_effort"):
                    x = max(x - max_delta_per_update, -max_total_open)
                    x_dot = min(x_dot, 0.0)
                    action = "safety_relax_over_max_effort" if effort >= max_effort else f"safety_{safety_action}"
                    stable_count = 0
                    if safety_abort_on_unsafe and safety_unsafe:
                        final_reason = "safety_abort_requested"
                else:
                    if abs(effort_error) <= deadband:
                        force_input = 0.0
                        stable_count += 1
                    else:
                        force_input = effort_to_accel_gain * effort_error
                        stable_count = 0

                    if virtual_mass <= 0.0:
                        virtual_mass = 1.0

                    # Actual scalar admittance dynamics.
                    x_ddot = (force_input - virtual_damping * x_dot - virtual_stiffness * x) / virtual_mass
                    x_dot = x_dot + x_ddot * dt
                    x_dot = max(-max_velocity, min(max_velocity, x_dot))
                    x = x + x_dot * dt
                    x = max(-max_total_open, min(max_total_close, x))

                    if stable_count >= required_stable:
                        action = "stable_near_target_admittance"
                        final_reason = "stable_near_target"
                    elif effort_error is not None and effort_error > deadband:
                        action = "admittance_close_from_low_effort"
                    elif effort_error is not None and effort_error < -deadband:
                        action = "admittance_relax_from_high_effort"
                    else:
                        action = "admittance_damped_hold"

                delta_to_apply = x - applied_x
                if abs(delta_to_apply) > max_delta_per_update:
                    delta_to_apply = max(-max_delta_per_update, min(max_delta_per_update, delta_to_apply))
                    x = applied_x + delta_to_apply

                if abs(delta_to_apply) > 1.0e-9:
                    adjust = self.gripper.adjust_hold_targets(
                        delta_close_m=delta_to_apply,
                        reason=action,
                    )
                    if adjust.get("applied"):
                        applied_x += delta_to_apply
                else:
                    adjust = {
                        "applied": False,
                        "reason": action,
                        "requested_delta_close_m": 0.0,
                    }

            summary["trace"].append({
                "frame": frame + 1,
                "action": action,
                "effort_sim": effort,
                "effort_error_sim": effort_error,
                "force_input_sim": force_input,
                "x_m": x,
                "x_dot_mps": x_dot,
                "x_ddot_mps2": x_ddot,
                "delta_close_m": delta_to_apply,
                "applied_x_m": applied_x,
                "stable_count": stable_count,
                "adjustment": adjust,
                "observation": obs,
                "soft_observation_for_safety": soft_obs_for_safety,
                "safety_assessment": safety_assessment,
            })

            if final_reason in ("stable_near_target", "safety_abort_requested"):
                break
            if action.startswith("safety_relax") and applied_x <= -max_total_open + 1.0e-9:
                final_reason = "safety_open_limit_reached"
                break

        # Compact Phase-3.3 safety summary for log analysis and reports.
        safety_states = {}
        safety_actions = {}
        deformation_scores = []
        combined_risk_scores = []
        max_compressions = []
        height_ratios = []
        soft_obs_count = 0
        for row in summary.get("trace", []):
            assess = row.get("safety_assessment") or {}
            if assess.get("soft_observation_available"):
                soft_obs_count += 1
            state = assess.get("safety_state")
            action_safety = assess.get("recommended_action")
            if state:
                safety_states[state] = safety_states.get(state, 0) + 1
            if action_safety:
                safety_actions[action_safety] = safety_actions.get(action_safety, 0) + 1
            for key, store in (
                ("deformation_score", deformation_scores),
                ("combined_risk_score", combined_risk_scores),
                ("compression_ratio_est", max_compressions),
                ("height_ratio", height_ratios),
            ):
                try:
                    val = assess.get(key)
                    if val is not None:
                        store.append(float(val))
                except Exception:
                    pass

        safety_summary = {
            "phase": "3.3_deformation_severity_index",
            "soft_observation_count": soft_obs_count,
            "safety_state_counts": safety_states,
            "safety_action_counts": safety_actions,
            "max_deformation_score": max(deformation_scores) if deformation_scores else None,
            "max_combined_risk_score": max(combined_risk_scores) if combined_risk_scores else None,
            "max_compression_ratio_est": max(max_compressions) if max_compressions else None,
            "min_height_ratio": min(height_ratios) if height_ratios else None,
            "interpretation": (
                "summary of AdaptiveSafetyMonitor decisions during scalar admittance; "
                "safe means effort and live deformable shape stayed inside configured thresholds"
            ),
        }

        summary.update({
            "final_reason": final_reason,
            "final_x_m": x,
            "applied_x_m": applied_x,
            "final_x_dot_mps": x_dot,
            "frames_used": (summary["trace"][-1]["frame"] if summary["trace"] else 0),
            "trace_len": len(summary["trace"]),
            "adaptive_safety_summary": safety_summary,
        })
        return summary


    # ─────────────────────────────────────────────────────────────
    # Phase 4.1f: shear/load observability during held-object transport
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _vsub(a, b):
        try:
            return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]
        except Exception:
            return None

    @staticmethod
    def _vdot(a, b):
        try:
            return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])
        except Exception:
            return None

    @staticmethod
    def _vnorm(a):
        try:
            return float((float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2) ** 0.5)
        except Exception:
            return None

    @classmethod
    def _vunit(cls, a):
        n = cls._vnorm(a)
        if n is None or n <= 1.0e-12:
            return None
        return [float(a[0]) / n, float(a[1]) / n, float(a[2]) / n]

    @classmethod
    def _vscale(cls, a, k):
        try:
            return [float(a[0]) * float(k), float(a[1]) * float(k), float(a[2]) * float(k)]
        except Exception:
            return None

    @classmethod
    def _vtangent_to_axis(cls, vec, axis_unit):
        """Return tangential component of vec with respect to candidate contact normal axis."""
        if vec is None or axis_unit is None:
            return None
        d = cls._vdot(vec, axis_unit)
        if d is None:
            return None
        normal = cls._vscale(axis_unit, d)
        return cls._vsub(vec, normal)

    def _flange_local_axis_world(self, local_axis_label: str):
        """Return a unit world vector for a flange-local axis label such as '+X' or '-Y'."""
        try:
            label = str(local_axis_label or '+X').strip().upper()
            sign = -1.0 if label.startswith('-') else 1.0
            axis_name = label[-1]
            basis = {'X': [1.0, 0.0, 0.0], 'Y': [0.0, 1.0, 0.0], 'Z': [0.0, 0.0, 1.0]}.get(axis_name)
            if basis is None:
                return None
            p0 = self.arm._transform_flange_local_point_to_world([0.0, 0.0, 0.0])
            p1 = self.arm._transform_flange_local_point_to_world([sign * basis[0], sign * basis[1], sign * basis[2]])
            return self._vunit(self._vsub(p1, p0))
        except Exception:
            return None

    def _transport_object_mass_kg(self, target: dict):
        """Best-effort mass estimate used only for shear/load audit calculations."""
        cfg_mass = self.config.get('place_transport_shear_audit_object_mass_kg', None)
        for candidate in (cfg_mass,
                          (target or {}).get('mass_kg') if isinstance(target, dict) else None,
                          (target or {}).get('mass') if isinstance(target, dict) else None):
            try:
                if candidate is not None:
                    value = float(candidate)
                    if value > 0.0:
                        return value
            except Exception:
                pass
        return float(self.config.get('place_transport_default_object_mass_kg', 0.010))

    def _estimate_required_normal_force_for_axes(self, *, mass_kg, accel_world, mu_candidates, safety_factor):
        """Estimate shear demand and required per-finger normal force for candidate flange axes.

        This is an observability model, not a calibrated force controller.  It asks:
        if this flange axis were the dominant contact-normal direction, what shear
        load would the grasp have to resist during the current transport sample?
        """
        g = float(self.config.get('place_transport_shear_audit_gravity_mps2', 9.81))
        gravity_world = [0.0, 0.0, -g]
        total_specific_load = [
            float(gravity_world[0]) - float(accel_world[0]),
            float(gravity_world[1]) - float(accel_world[1]),
            float(gravity_world[2]) - float(accel_world[2]),
        ]
        labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        assumed = str(self.config.get('place_transport_shear_audit_assumed_normal_axis_local', self.config.get('finger_capture_axis_local', '+X'))).upper()
        results = {}
        for label in labels:
            axis = self._flange_local_axis_world(label)
            tan = self._vtangent_to_axis(total_specific_load, axis)
            tan_accel_mag = self._vnorm(tan)
            if tan_accel_mag is None:
                continue
            f_shear = float(mass_kg) * float(tan_accel_mag)
            req = {}
            for mu in mu_candidates:
                try:
                    mu_f = float(mu)
                    if mu_f <= 1.0e-9:
                        continue
                    # Two-finger parallel grasp: 2 * mu * N >= F_shear.
                    req[str(mu_f)] = float(safety_factor) * f_shear / (2.0 * mu_f)
                except Exception:
                    pass
            results[label] = {
                'axis_world_unit': axis,
                'tangential_specific_load_mps2': tan_accel_mag,
                'estimated_shear_force_N': f_shear,
                'required_normal_force_per_finger_N_by_mu': req,
                'is_assumed_axis': label == assumed,
            }
        return {
            'model': 'two-finger friction grasp audit: 2*mu*N >= F_shear; N_req = safety_factor*F_shear/(2*mu)',
            'mass_kg': mass_kg,
            'gravity_mps2': g,
            'specific_load_world_mps2': total_specific_load,
            'assumed_normal_axis_local': assumed,
            'axis_candidates': results,
        }

    def _compute_transport_shear_reference(self, *, target=None, place_goal=None, transport_duration=None, transport_steps=None):
        """Compute a shear-aware gripper effort reference before transport.

        Phase 4.1f showed that transport slip is gravity/shear dominated and
        that changing policy without a load model is wrong.  This function keeps
        the model explicit:

            2 * mu * N >= F_shear
            N_req = safety_factor * F_shear / (2 * mu)

        We still do NOT claim that measured Isaac finger effort is calibrated
        fingertip normal force.  The output is therefore an effort-proxy target:
        target_effort_proxy = bias + proxy_gain * N_req, clamped by config.
        """
        enabled = bool(self.config.get("place_transport_shear_compensation_enabled", True))
        summary = {
            "enabled": enabled,
            "phase": "4.1g_calibrated_shear_aware_transport_reference",
            "model": "two-finger friction grasp: 2*mu*N >= F_shear",
            "units_warning": "target_effort_sim is a calibrated/empirical joint-effort proxy, not direct fingertip normal force",
            "success": False,
        }
        if not enabled:
            summary["reason"] = "place_transport_shear_compensation_disabled"
            return summary

        try:
            mass_kg = self._transport_object_mass_kg(target)
            mu_eff = float(self.config.get("place_transport_shear_mu_effective", 0.35))
            safety_factor = float(self.config.get("place_transport_shear_safety_factor", 2.0))
            proxy_gain = float(self.config.get("place_transport_proxy_effort_per_newton", 1.0))
            proxy_bias = float(self.config.get("place_transport_proxy_effort_bias_sim", 0.25))
            min_target = float(self.config.get("place_transport_shear_target_min_sim", 0.42))
            max_target = float(self.config.get("place_transport_shear_target_max_sim", 0.62))
            base_target = float(
                self.config.get(
                    "adaptive_effort_target_sim",
                    self.config.get("force_observer_target_effort_sim", 0.35),
                )
            )

            # Estimate a nominal transport acceleration from a smoothstep move.
            # smoothstep has max normalized acceleration about 6/T^2, so this is
            # a conservative feed-forward estimate.  Phase 4.1f showed gravity
            # dominates here, but we keep acceleration in the model for future
            # heavier/faster objects.
            cur_obj = self._get_observed_object_pos(target, stage_name="shear_reference_object")
            place_zone = (place_goal or {}).get("place_zone", {}) if isinstance(place_goal, dict) else {}
            place_center = place_zone.get("world_center") if isinstance(place_zone, dict) else None
            transport_vec = None
            transport_dist = 0.0
            accel_world = [0.0, 0.0, 0.0]
            if cur_obj is not None and place_center is not None:
                transport_vec = [
                    float(place_center[0]) - float(cur_obj[0]),
                    float(place_center[1]) - float(cur_obj[1]),
                    0.0,
                ]
                transport_dist = self._vnorm(transport_vec) or 0.0
                unit = self._vunit(transport_vec)
                T = max(1.0e-6, float(transport_duration or self.config.get("place_transport_duration", 5.0)))
                accel_mag = float(self.config.get("place_transport_shear_accel_scale", 6.0)) * transport_dist / (T * T)
                if unit is not None:
                    accel_world = self._vscale(unit, accel_mag) or [0.0, 0.0, 0.0]

            mu_candidates = self.config.get("place_transport_shear_audit_mu_candidates", [0.3, 0.5, 0.8])
            if not isinstance(mu_candidates, (list, tuple)):
                mu_candidates = [0.3, 0.5, 0.8]
            if mu_eff not in [float(m) for m in mu_candidates if str(m).strip()]:
                mu_candidates = list(mu_candidates) + [mu_eff]

            estimate = self._estimate_required_normal_force_for_axes(
                mass_kg=mass_kg,
                accel_world=accel_world,
                mu_candidates=mu_candidates,
                safety_factor=safety_factor,
            )
            axis_candidates = estimate.get("axis_candidates", {}) or {}
            mode = str(self.config.get("place_transport_contact_normal_axis_mode", "AUTO_HORIZONTAL_MAX_SHEAR")).upper()
            explicit_axis = str(self.config.get("place_transport_contact_normal_axis_local", "")).upper().strip()
            max_abs_vertical = float(self.config.get("place_transport_contact_axis_max_abs_vertical_component", 0.75))

            selected_label = None
            selected = None
            if explicit_axis in axis_candidates and not explicit_axis.startswith("AUTO"):
                selected_label = explicit_axis
                selected = axis_candidates.get(selected_label)
            else:
                candidates = []
                for label, data in axis_candidates.items():
                    axis = data.get("axis_world_unit") or [0.0, 0.0, 1.0]
                    try:
                        abs_z = abs(float(axis[2]))
                    except Exception:
                        abs_z = 1.0
                    if "HORIZONTAL" in mode and abs_z > max_abs_vertical:
                        continue
                    candidates.append((float(data.get("estimated_shear_force_N", 0.0)), label, data))
                if not candidates:
                    candidates = [
                        (float(data.get("estimated_shear_force_N", 0.0)), label, data)
                        for label, data in axis_candidates.items()
                    ]
                if candidates:
                    candidates.sort(reverse=True, key=lambda row: row[0])
                    _, selected_label, selected = candidates[0]

            f_shear = float((selected or {}).get("estimated_shear_force_N", 0.0))
            n_required = safety_factor * f_shear / max(2.0 * mu_eff, 1.0e-9)
            raw_target = proxy_bias + proxy_gain * n_required
            target_effort = max(base_target, min(max_target, max(min_target, raw_target)))

            summary.update({
                "success": True,
                "selected_contact_normal_axis_local": selected_label,
                "contact_axis_selection_mode": mode,
                "selected_axis_world_unit": (selected or {}).get("axis_world_unit"),
                "mass_kg": mass_kg,
                "transport_vector_xy_m": transport_vec,
                "transport_distance_xy_m": transport_dist,
                "nominal_accel_world_mps2": accel_world,
                "estimated_shear_force_N": f_shear,
                "mu_effective": mu_eff,
                "safety_factor": safety_factor,
                "required_normal_force_per_finger_N": n_required,
                "proxy_effort_per_newton": proxy_gain,
                "proxy_effort_bias_sim": proxy_bias,
                "base_static_target_effort_sim": base_target,
                "raw_shear_target_effort_sim": raw_target,
                "target_effort_sim": target_effort,
                "target_clamp_min_sim": min_target,
                "target_clamp_max_sim": max_target,
                "axis_candidates": axis_candidates,
                "interpretation": (
                    "Use this as a pre-transport grip effort reference. If slip still grows while this "
                    "target is reached, the limiting factor is likely contact geometry/friction/deformation, "
                    "not just scalar normal effort."
                ),
            })
        except Exception as e:
            summary.update({"success": False, "reason": f"exception: {e}"})
        return json_safe(summary)

    async def _direct_transport_preload_boost(self, *, target=None, target_effort_sim=None):
        """Direct stationary preload booster for Phase 4.1h.

        The Phase 4.1g log showed that the formal scalar admittance preload had
        the right target but did not reach it before transport.  This booster is
        still conservative, but it is direct: while stationary, close the hold
        targets in small increments until measured effort is near the shear
        target, deformation becomes too high, or the configured extra closure is
        exhausted.  It is intentionally only used before transport, not as a
        general policy during motion.
        """
        summary = {
            "enabled": bool(self.config.get("place_transport_direct_preload_enabled", True)),
            "phase": "4.1h_direct_stationary_shear_preload_booster",
            "ran": False,
            "target_effort_sim": target_effort_sim,
            "trace": [],
            "action_counts": {},
        }
        if not summary["enabled"]:
            summary["reason"] = "place_transport_direct_preload_disabled"
            return summary
        try:
            target_effort = float(target_effort_sim)
        except Exception:
            summary["reason"] = "invalid_target_effort"
            return summary

        app = omni.kit.app.get_app()
        target_band = float(self.config.get("place_transport_direct_preload_target_band_sim", 0.035))
        max_frames = max(1, int(self.config.get("place_transport_direct_preload_max_frames", 180)))
        sample_stride = max(1, int(self.config.get("place_transport_direct_preload_sample_stride_frames", 4)))
        step_m = float(self.config.get("place_transport_direct_preload_step_m", 0.00008))
        max_extra = float(self.config.get("place_transport_direct_preload_max_extra_close_m", 0.00120))
        max_effort = float(self.config.get("place_transport_direct_preload_max_effort_sim", self.config.get("adaptive_effort_max_sim", 1.20)))
        max_shape_ratio = float(self.config.get("place_transport_direct_preload_max_shape_ratio", 1.45))
        force_n = self.config.get("place_transport_direct_preload_force_n", None)
        if force_n is not None:
            try:
                force_n = float(force_n)
            except Exception:
                force_n = None

        summary.update({
            "ran": True,
            "target_band_sim": target_band,
            "max_frames": max_frames,
            "sample_stride_frames": sample_stride,
            "step_m": step_m,
            "max_extra_close_m": max_extra,
            "max_effort_sim": max_effort,
            "max_shape_ratio": max_shape_ratio,
            "force_n": force_n,
        })

        total_extra = 0.0
        stable_samples = 0
        final_reason = "max_frames_reached"

        def count(action):
            d = summary.setdefault("action_counts", {})
            d[action] = d.get(action, 0) + 1

        for frame in range(max_frames):
            self.gripper.update()
            await app.next_update_async()
            if frame % sample_stride != 0:
                continue

            try:
                effort_obs = self.force_observer.observe(stage_name=f"direct_transport_preload_frame_{frame}") if hasattr(self, "force_observer") else None
            except Exception as e:
                effort_obs = {"available": False, "reason": str(e)}
            try:
                soft_obs = self._observe_target(target, stage_name=f"direct_transport_preload_frame_{frame}")
            except Exception as e:
                soft_obs = {"available": False, "reason": str(e)}

            effort = None
            if isinstance(effort_obs, dict) and effort_obs.get("available"):
                try:
                    effort = float(effort_obs.get("grip_effort_sim"))
                except Exception:
                    effort = None
            err = None if effort is None else (target_effort - effort)

            # Prefer rotation-aware OBB ratio if available; otherwise fall back
            # to world-AABB ratio.  The booster should not overreact to simple
            # rigid rotation.
            shape_ratio_source = "none"
            shape_ratio = None
            if isinstance(soft_obs, dict):
                try:
                    if soft_obs.get("oriented_max_ratio") is not None and bool(self.config.get("place_transport_use_oriented_shape_for_deformation", True)):
                        shape_ratio = float(soft_obs.get("oriented_max_ratio"))
                        shape_ratio_source = "oriented_pca_bbox"
                    else:
                        ratios = []
                        for key in ("width_ratio_x", "width_ratio_y"):
                            if soft_obs.get(key) is not None:
                                ratios.append(float(soft_obs.get(key)))
                        h = soft_obs.get("height_m")
                        nominal_h = float(self.config.get("adaptive_safety_nominal_height_m", 0.040))
                        if h is not None and nominal_h > 1.0e-9:
                            ratios.append(float(h) / nominal_h)
                        if ratios:
                            shape_ratio = max(ratios)
                            shape_ratio_source = "world_aabb"
                except Exception:
                    pass

            action = "hold"
            adjustment = None
            if effort is not None and effort >= target_effort - target_band:
                stable_samples += 1
                action = "stable_near_shear_target"
                if stable_samples >= int(self.config.get("place_transport_direct_preload_required_stable_samples", 2)):
                    final_reason = "stable_near_shear_target"
                    count(action)
                    summary["trace"].append(json_safe({
                        "frame": frame,
                        "action": action,
                        "effort_sim": effort,
                        "error_sim": err,
                        "shape_ratio": shape_ratio,
                        "shape_ratio_source": shape_ratio_source,
                        "total_extra_close_m": total_extra,
                        "effort_observation": effort_obs,
                        "soft_observation": soft_obs,
                    }))
                    break
            elif effort is not None and effort >= max_effort:
                action = "stop_over_effort"
                final_reason = "over_max_effort"
                count(action)
                summary["trace"].append(json_safe({
                    "frame": frame,
                    "action": action,
                    "effort_sim": effort,
                    "error_sim": err,
                    "shape_ratio": shape_ratio,
                    "shape_ratio_source": shape_ratio_source,
                    "total_extra_close_m": total_extra,
                    "effort_observation": effort_obs,
                    "soft_observation": soft_obs,
                }))
                break
            elif shape_ratio is not None and shape_ratio >= max_shape_ratio:
                action = "stop_shape_ratio_limit"
                final_reason = "shape_ratio_limit"
                count(action)
                summary["trace"].append(json_safe({
                    "frame": frame,
                    "action": action,
                    "effort_sim": effort,
                    "error_sim": err,
                    "shape_ratio": shape_ratio,
                    "shape_ratio_source": shape_ratio_source,
                    "total_extra_close_m": total_extra,
                    "effort_observation": effort_obs,
                    "soft_observation": soft_obs,
                }))
                break
            else:
                stable_samples = 0
                if total_extra < max_extra:
                    delta = min(step_m, max_extra - total_extra)
                    adjustment = self.gripper.adjust_hold_targets(
                        delta_close_m=delta,
                        reason="direct_pre_transport_shear_preload",
                        force_n=force_n,
                    )
                    if isinstance(adjustment, dict) and adjustment.get("applied"):
                        total_extra += delta
                    action = "close_toward_shear_target"
                else:
                    action = "stop_extra_close_limit"
                    final_reason = "extra_close_limit"
                    count(action)
                    summary["trace"].append(json_safe({
                        "frame": frame,
                        "action": action,
                        "effort_sim": effort,
                        "error_sim": err,
                        "shape_ratio": shape_ratio,
                        "shape_ratio_source": shape_ratio_source,
                        "total_extra_close_m": total_extra,
                        "adjustment": adjustment,
                        "effort_observation": effort_obs,
                        "soft_observation": soft_obs,
                    }))
                    break

            count(action)
            summary["trace"].append(json_safe({
                "frame": frame,
                "action": action,
                "effort_sim": effort,
                "error_sim": err,
                "shape_ratio": shape_ratio,
                "shape_ratio_source": shape_ratio_source,
                "total_extra_close_m": total_extra,
                "adjustment": adjustment,
                "effort_observation": effort_obs,
                "soft_observation": soft_obs,
            }))

        summary.update(json_safe({
            "final_reason": final_reason,
            "frames_used": frame if 'frame' in locals() else 0,
            "total_extra_close_m": total_extra,
            "trace_len": len(summary.get("trace", [])),
        }))
        return json_safe(summary)

    async def _apply_transport_shear_preload(self, *, target=None, shear_reference=None):
        """Stationary pre-transport preload using the shear-aware effort target.

        The important change is timing: do not wait until the object has already
        started sliding.  Build the predicted transport normal effort while the
        arm is still stationary, then move.
        """
        summary = {
            "enabled": bool(self.config.get("place_transport_shear_preload_enabled", True)),
            "phase": "4.1g_pre_transport_shear_preload",
            "ran": False,
            "shear_reference_target_effort_sim": None,
        }
        if not summary["enabled"]:
            summary["reason"] = "place_transport_shear_preload_disabled"
            return summary
        if not isinstance(shear_reference, dict) or not shear_reference.get("success"):
            summary["reason"] = "missing_or_invalid_shear_reference"
            summary["shear_reference"] = json_safe(shear_reference)
            return summary
        target_effort = shear_reference.get("target_effort_sim")
        try:
            target_effort = float(target_effort)
        except Exception:
            summary["reason"] = "invalid_target_effort"
            return summary

        summary["ran"] = True
        summary["shear_reference_target_effort_sim"] = target_effort
        try:
            summary["before_force_observation"] = json_safe(
                self.force_observer.observe(stage_name="before_transport_shear_preload")
                if hasattr(self, "force_observer") else None
            )
        except Exception as e:
            summary["before_force_observation"] = {"available": False, "reason": str(e)}
        try:
            summary["before_soft_observation"] = json_safe(
                self._observe_target(target, stage_name="before_transport_shear_preload")
            )
        except Exception as e:
            summary["before_soft_observation"] = {"available": False, "reason": str(e)}

        preload_result = await self._regulate_gripper_effort_after_close(
            target=target,
            target_effort_override_sim=target_effort,
        )
        summary["regulation"] = json_safe(preload_result)

        # Phase 4.1h: the 4.1g log showed that the formal preload target was
        # computed correctly but not actually reached before transport.  Add a
        # direct stationary booster so the predicted shear target is achieved
        # before the object is exposed to transport shear.
        direct_boost = None
        if bool(self.config.get("place_transport_direct_preload_enabled", True)):
            direct_boost = await self._direct_transport_preload_boost(
                target=target,
                target_effort_sim=target_effort,
            )
        summary["direct_preload_boost"] = json_safe(direct_boost)

        settle_s = float(self.config.get("place_transport_shear_preload_settle_seconds", 0.25))
        if settle_s > 0.0:
            await self._step_gripper_for_seconds(settle_s)
        try:
            summary["after_force_observation"] = json_safe(
                self.force_observer.observe(stage_name="after_transport_shear_preload")
                if hasattr(self, "force_observer") else None
            )
        except Exception as e:
            summary["after_force_observation"] = {"available": False, "reason": str(e)}
        try:
            summary["after_soft_observation"] = json_safe(
                self._observe_target(target, stage_name="after_transport_shear_preload")
            )
        except Exception as e:
            summary["after_soft_observation"] = {"available": False, "reason": str(e)}
        return json_safe(summary)

    def _make_transport_shear_audit_step_callback(self, *, target=None, base_step_callback=None, dt_s=1.0/60.0, place_goal=None) -> tuple:
        """Create a non-invasive audit callback for shear-aware transport diagnosis.

        It does not change gripper targets.  It samples force proxy, soft-object pose,
        flange motion, relative object/flange drift, and a simple friction-grasp
        shear model.  The point is to decide whether the next controller should
        compensate by grip preload, trajectory acceleration reduction, wrist rotation,
        or a regrasp/contact-geometry change.
        """
        enabled = bool(self.config.get('place_transport_shear_audit_enabled', True))
        summary = {
            'enabled': enabled,
            'phase': '4.1f_shear_force_and_slip_observability_audit',
            'controller_mode': 'audit_only_no_gripper_policy_change',
            'design_intent': (
                'measure the transport shear/load problem before tuning control: '
                'estimate shear demand from mass + flange acceleration + gravity, '
                'compare it with measured gripper effort proxy and object/flange drift, '
                'and log which flange/contact direction is likely weak.'
            ),
            'trace': [],
            'action_counts': {},
            'ran': False,
            'reason': None,
        }
        if not enabled:
            summary['reason'] = 'place_transport_shear_audit_disabled'
            return base_step_callback or self.gripper.update, summary

        sample_stride = max(1, int(self.config.get('place_transport_shear_audit_sample_stride_frames', 6)))
        max_trace = max(1, int(self.config.get('place_transport_shear_audit_max_trace_samples', 120)))
        mass_kg = self._transport_object_mass_kg(target)
        safety_factor = float(self.config.get('place_transport_shear_audit_safety_factor', 2.0))
        mu_candidates = self.config.get('place_transport_shear_audit_mu_candidates', [0.3, 0.5, 0.8])
        if not isinstance(mu_candidates, (list, tuple)):
            mu_candidates = [0.3, 0.5, 0.8]
        dt = max(1.0e-6, float(dt_s) * float(sample_stride))

        initial_obj = None
        initial_flange = None
        initial_rel = None
        try:
            initial_obj = self._get_observed_object_pos(target, stage_name='shear_audit_initial_object')
            initial_flange = self.arm.get_flange_world_pos()
            if initial_obj is not None and initial_flange is not None:
                initial_rel = self._vsub(initial_obj, initial_flange)
        except Exception as e:
            summary['initial_relation_error'] = str(e)

        # Initial axis/transport geometry snapshot.
        local_axis_labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        axis_snapshot = {}
        for label in local_axis_labels:
            axis_snapshot[label] = self._flange_local_axis_world(label)

        state = {
            'frame': 0,
            'samples': 0,
            'prev_flange': None,
            'prev_obj': None,
            'prev_flange_vel': [0.0, 0.0, 0.0],
            'prev_obj_rel_drift': None,
            'max_flange_speed_mps': 0.0,
            'max_flange_accel_mps2': 0.0,
            'max_object_speed_mps': 0.0,
            'max_relative_drift_m': 0.0,
            'max_relative_drift_xy_m': 0.0,
            'max_slip_velocity_mps': 0.0,
            'max_effort_proxy': 0.0,
            'max_width_ratio': None,
            'max_deformation_score': None,
            'max_assumed_axis_shear_N': 0.0,
            'min_proxy_margin_by_mu': {},
        }

        summary.update({
            'ran': True,
            'sample_stride_frames': sample_stride,
            'dt_per_sample_s': dt,
            'object_mass_kg_used': mass_kg,
            'mu_candidates': [float(m) for m in mu_candidates if str(m).strip()],
            'safety_factor': safety_factor,
            'initial_object_pos': initial_obj,
            'initial_flange_pos': initial_flange,
            'initial_object_flange_relative_pos': initial_rel,
            'flange_axis_world_snapshot': axis_snapshot,
            'place_goal': place_goal,
            'limitations': [
                'measured gripper effort is a prismatic-joint effort proxy, not calibrated fingertip normal force',
                'candidate contact-normal axes are audited because the exact finger-pad normal/shear frame is not yet calibrated',
                'soft-object deformation can convert extra normal effort into extrusion instead of useful friction',
            ],
        })

        def _update_count(action: str):
            counts = summary.setdefault('action_counts', {})
            counts[action] = counts.get(action, 0) + 1

        def _safe_float(v):
            try:
                return float(v)
            except Exception:
                return None

        def _callback():
            base_result = None
            if base_step_callback is not None:
                base_result = base_step_callback()
            else:
                self.gripper.update()

            state['frame'] += 1
            frame = state['frame']
            if (frame % sample_stride) != 0:
                return base_result

            obs = None
            soft_obs = None
            safety = None
            effort = None
            try:
                obs = self.force_observer.observe(stage_name=f'place_transport_shear_audit_frame_{frame}')
                if obs and obs.get('available'):
                    effort = _safe_float(obs.get('grip_effort_sim'))
                    if effort is not None:
                        state['max_effort_proxy'] = max(state['max_effort_proxy'], effort)
            except Exception as e:
                obs = {'available': False, 'reason': f'force_observer_exception: {e}'}
            try:
                soft_obs = self._observe_target(target, stage_name=f'place_transport_shear_audit_frame_{frame}')
            except Exception as e:
                soft_obs = {'available': False, 'reason': f'soft_observer_exception: {e}'}

            flange = None
            obj = None
            flange_vel = None
            obj_vel = None
            flange_accel = [0.0, 0.0, 0.0]
            flange_speed = None
            flange_accel_mag = None
            obj_speed = None
            rel_drift = None
            rel_drift_xy = None
            slip_velocity = None
            transport_dir = None
            try:
                flange = self.arm.get_flange_world_pos()
                obj = (soft_obs or {}).get('center') if isinstance(soft_obs, dict) else None
                if obj is None:
                    obj = self._get_observed_object_pos(target, stage_name=f'shear_audit_object_pos_frame_{frame}')

                if state['prev_flange'] is not None and flange is not None:
                    flange_vel = [
                        (float(flange[0]) - float(state['prev_flange'][0])) / dt,
                        (float(flange[1]) - float(state['prev_flange'][1])) / dt,
                        (float(flange[2]) - float(state['prev_flange'][2])) / dt,
                    ]
                    flange_speed = self._vnorm(flange_vel)
                    if flange_speed is not None:
                        state['max_flange_speed_mps'] = max(state['max_flange_speed_mps'], flange_speed)
                    flange_accel = [
                        (flange_vel[0] - state['prev_flange_vel'][0]) / dt,
                        (flange_vel[1] - state['prev_flange_vel'][1]) / dt,
                        (flange_vel[2] - state['prev_flange_vel'][2]) / dt,
                    ]
                    flange_accel_mag = self._vnorm(flange_accel)
                    if flange_accel_mag is not None:
                        state['max_flange_accel_mps2'] = max(state['max_flange_accel_mps2'], flange_accel_mag)
                    state['prev_flange_vel'] = flange_vel

                if state['prev_obj'] is not None and obj is not None:
                    obj_vel = [
                        (float(obj[0]) - float(state['prev_obj'][0])) / dt,
                        (float(obj[1]) - float(state['prev_obj'][1])) / dt,
                        (float(obj[2]) - float(state['prev_obj'][2])) / dt,
                    ]
                    obj_speed = self._vnorm(obj_vel)
                    if obj_speed is not None:
                        state['max_object_speed_mps'] = max(state['max_object_speed_mps'], obj_speed)

                if initial_rel is not None and obj is not None and flange is not None:
                    cur_rel = self._vsub(obj, flange)
                    rel_delta = self._vsub(cur_rel, initial_rel)
                    if rel_delta is not None:
                        rel_drift = self._vnorm(rel_delta)
                        rel_drift_xy = self._vnorm([rel_delta[0], rel_delta[1], 0.0])
                        state['max_relative_drift_m'] = max(state['max_relative_drift_m'], rel_drift or 0.0)
                        state['max_relative_drift_xy_m'] = max(state['max_relative_drift_xy_m'], rel_drift_xy or 0.0)
                        if state['prev_obj_rel_drift'] is not None and rel_drift is not None:
                            slip_velocity = (rel_drift - state['prev_obj_rel_drift']) / dt
                            state['max_slip_velocity_mps'] = max(state['max_slip_velocity_mps'], abs(slip_velocity))
                        state['prev_obj_rel_drift'] = rel_drift

                if initial_flange is not None and flange is not None:
                    transport_dir = self._vunit(self._vsub(flange, initial_flange))

                state['prev_flange'] = list(flange) if flange is not None else state['prev_flange']
                state['prev_obj'] = list(obj) if obj is not None else state['prev_obj']
            except Exception as e:
                summary.setdefault('kinematic_errors', []).append(str(e))

            try:
                width_ratio_x = _safe_float((soft_obs or {}).get('width_x_m')) / float(self.config.get('adaptive_safety_nominal_width_m', 0.040))
                width_ratio_y = _safe_float((soft_obs or {}).get('width_y_m')) / float(self.config.get('adaptive_safety_nominal_depth_m', 0.040))
                max_width_ratio = max(width_ratio_x, width_ratio_y)
                old = state.get('max_width_ratio')
                state['max_width_ratio'] = max_width_ratio if old is None else max(old, max_width_ratio)
            except Exception:
                width_ratio_x = width_ratio_y = max_width_ratio = None

            try:
                safety = self.safety_monitor.assess(
                    soft_obs=soft_obs,
                    effort_obs=obs,
                    context={
                        'frame': frame,
                        'controller_phase': 'place_transport_shear_audit',
                        'relative_object_flange_drift_m': rel_drift,
                        'relative_object_flange_drift_xy_m': rel_drift_xy,
                    },
                )
                ds = _safe_float((safety or {}).get('deformation_score'))
                if ds is not None:
                    old = state.get('max_deformation_score')
                    state['max_deformation_score'] = ds if old is None else max(old, ds)
            except Exception as e:
                safety = {'available': False, 'reason': f'safety_monitor_exception: {e}'}

            shear_est = self._estimate_required_normal_force_for_axes(
                mass_kg=mass_kg,
                accel_world=flange_accel,
                mu_candidates=mu_candidates,
                safety_factor=safety_factor,
            )
            assumed = shear_est.get('assumed_normal_axis_local')
            assumed_entry = (shear_est.get('axis_candidates') or {}).get(assumed)
            assumed_shear = None
            assumed_req = None
            if isinstance(assumed_entry, dict):
                assumed_shear = _safe_float(assumed_entry.get('estimated_shear_force_N'))
                assumed_req = assumed_entry.get('required_normal_force_per_finger_N_by_mu') or {}
                if assumed_shear is not None:
                    state['max_assumed_axis_shear_N'] = max(state['max_assumed_axis_shear_N'], assumed_shear)
                if effort is not None and assumed_shear is not None and assumed_shear > 1.0e-9:
                    for mu in mu_candidates:
                        try:
                            mu_f = float(mu)
                            # Proxy only: assumes grip_effort_sim behaves like normal force per finger.
                            margin = (2.0 * mu_f * effort) / assumed_shear
                            key = str(mu_f)
                            prev = state['min_proxy_margin_by_mu'].get(key)
                            state['min_proxy_margin_by_mu'][key] = margin if prev is None else min(prev, margin)
                        except Exception:
                            pass

            action = 'audit_sample'
            if slip_velocity is not None and abs(slip_velocity) > float(self.config.get('place_transport_shear_audit_warn_slip_velocity_mps', 0.003)):
                action = 'audit_slip_velocity_warning'
            if rel_drift is not None and rel_drift > float(self.config.get('place_transport_shear_audit_warn_relative_drift_m', 0.004)):
                action = 'audit_relative_drift_warning'
            _update_count(action)

            sample = {
                'frame': frame,
                'action': action,
                'flange_pos': flange,
                'object_pos': obj,
                'flange_velocity_mps': flange_vel,
                'flange_speed_mps': flange_speed,
                'flange_accel_mps2': flange_accel,
                'flange_accel_mag_mps2': flange_accel_mag,
                'object_velocity_mps': obj_vel,
                'object_speed_mps': obj_speed,
                'transport_dir_world_unit': transport_dir,
                'relative_object_flange_drift_m': rel_drift,
                'relative_object_flange_drift_xy_m': rel_drift_xy,
                'slip_velocity_mps': slip_velocity,
                'grip_effort_proxy_sim': effort,
                'width_ratio_x': width_ratio_x,
                'width_ratio_y': width_ratio_y,
                'max_width_ratio': max_width_ratio,
                'soft_observation_table_gap_m': (soft_obs or {}).get('table_gap_m') if isinstance(soft_obs, dict) else None,
                'shear_estimate': shear_est,
                'assumed_axis_shear_force_N': assumed_shear,
                'assumed_axis_required_normal_N_by_mu': assumed_req,
                'force_observation': obs,
                'soft_observation': soft_obs,
                'safety_assessment': safety,
            }
            if len(summary['trace']) < max_trace:
                summary['trace'].append(json_safe(sample))
            summary.update(json_safe({
                'frames_seen': state['frame'],
                'samples': state['samples'] + 1,
                'max_flange_speed_mps': state['max_flange_speed_mps'],
                'max_flange_accel_mps2': state['max_flange_accel_mps2'],
                'max_object_speed_mps': state['max_object_speed_mps'],
                'max_relative_object_flange_drift_m': state['max_relative_drift_m'],
                'max_relative_object_flange_drift_xy_m': state['max_relative_drift_xy_m'],
                'max_slip_velocity_mps': state['max_slip_velocity_mps'],
                'max_grip_effort_proxy_sim': state['max_effort_proxy'],
                'max_width_ratio': state['max_width_ratio'],
                'max_deformation_score': state['max_deformation_score'],
                'max_assumed_axis_shear_force_N': state['max_assumed_axis_shear_N'],
                'min_proxy_margin_by_mu': state['min_proxy_margin_by_mu'],
            }))
            state['samples'] += 1
            return base_result

        return _callback, summary

    def _make_transport_admittance_step_callback(self, target=None, target_effort_override_sim=None, shear_reference=None) -> tuple:
        """Create an active gripper-admittance callback for held-object transport.

        Phase 4.1/4.1b transported the held foam using a fixed gripper HOLD target.
        That is not enough for soft objects: during sideways motion the object can
        slip or stretch inside the fingers even if the pre-transport grasp was
        valid.  This callback keeps the gripper feedback loop alive while the arm
        moves.

        It is deliberately scalar and conservative:
          - measured finger-joint effort comes from ForceObserver;
          - live deformable pose/shape comes from SoftObjectObserver;
          - AdaptiveSafetyMonitor can request relaxation if deformation is unsafe;
          - small target corrections are applied through gripper.adjust_hold_targets().

        Positive delta closes the gripper slightly; negative delta relaxes it.
        The callback is synchronous because UR5EController.move_to() calls the
        step_callback once per simulation update.
        """
        enabled = bool(self.config.get("place_transport_admittance_enabled", True))
        summary = {
            "enabled": enabled,
            "phase": "4.1c_transport_gripper_admittance",
            "controller": "transport_active_scalar_gripper_admittance",
            "force_source": "measured_joint_effort_sim",
            "pose_source": "soft_simulation_mesh_points_when_available",
            "design_intent": (
                "keep regulating the gripper during arm transport; increase gentle squeeze "
                "when effort is low or object/flange relative drift grows; during transport, "
                "do not open on deformation alone because opening a moving grasp can drop the object; "
                "instead request an arm abort on critical slip/fall"
            ),
            "trace": [],
            "action_counts": {},
            "ran": False,
            "reason": None,
        }
        if not enabled:
            summary["reason"] = "place_transport_admittance_disabled"
            return self.gripper.update, summary
        if not hasattr(self, "force_observer"):
            summary["reason"] = "force_observer_missing"
            return self.gripper.update, summary
        if not hasattr(self.gripper, "adjust_hold_targets"):
            summary["reason"] = "gripper_adjust_hold_targets_missing"
            return self.gripper.update, summary

        base_target = float(
            self.config.get(
                "adaptive_effort_target_sim",
                self.config.get("force_observer_target_effort_sim", 0.35),
            )
        )
        target_effort = float(
            target_effort_override_sim
            if target_effort_override_sim is not None
            else self.config.get(
                "place_transport_admittance_target_effort_sim",
                base_target + float(self.config.get("place_transport_effort_boost_sim", 0.05)),
            )
        )
        max_effort = float(
            self.config.get(
                "place_transport_admittance_max_effort_sim",
                self.config.get("adaptive_effort_max_sim", 1.20),
            )
        )
        deadband = float(self.config.get("place_transport_admittance_deadband_sim", 0.035))
        sample_stride = max(1, int(self.config.get("place_transport_admittance_sample_stride_frames", 12)))
        delta_step = float(self.config.get("place_transport_admittance_delta_step_m", 0.00005))
        max_total_close = float(self.config.get("place_transport_admittance_max_extra_close_m", 0.00075))
        max_total_open = float(self.config.get("place_transport_admittance_max_relax_open_m", 0.00060))
        slip_warn_m = float(self.config.get("place_transport_slip_warn_relative_drift_m", 0.0040))
        slip_close_m = float(self.config.get("place_transport_slip_close_relative_drift_m", 0.0060))
        slip_critical_m = float(self.config.get("place_transport_slip_critical_relative_drift_m", 0.0120))
        deformation_warn_score = float(self.config.get("place_transport_admittance_warn_deformation_score", 0.20))
        slip_close_effort_margin = float(self.config.get("place_transport_slip_close_effort_margin_sim", 0.040))
        slip_close_max_shape_ratio = float(self.config.get("place_transport_slip_close_max_shape_ratio", 1.34))
        disable_relax_during_motion = bool(self.config.get("place_transport_disable_relax_during_motion", True))
        abort_on_critical_slip = bool(self.config.get("place_transport_abort_on_critical_slip", True))
        abort_on_effort_loss = bool(self.config.get("place_transport_abort_on_effort_loss", True))
        abort_on_table_drop = bool(self.config.get("place_transport_abort_on_table_drop", True))
        effort_loss_threshold = float(self.config.get("place_transport_effort_loss_threshold_sim", 0.10))
        min_table_gap_during_hold = float(self.config.get("place_transport_min_table_gap_during_hold_m", 0.030))
        emergency_close_step = float(self.config.get("place_transport_emergency_close_step_m", max(delta_step, 0.00010)))

        # Initial object/flange relation: if this changes during transport, the
        # object is sliding/stretching relative to the gripper rather than just
        # moving rigidly with the arm.
        initial_obj = None
        initial_flange = None
        initial_rel = None
        try:
            initial_obj = self._get_observed_object_pos(target, stage_name="transport_admittance_initial_object")
            initial_flange = self.arm.get_flange_world_pos()
            if initial_obj is not None and initial_flange is not None:
                initial_rel = [
                    float(initial_obj[0]) - float(initial_flange[0]),
                    float(initial_obj[1]) - float(initial_flange[1]),
                    float(initial_obj[2]) - float(initial_flange[2]),
                ]
        except Exception as e:
            summary["initial_relation_error"] = str(e)

        state = {
            "frame": 0,
            "applied_x_m": 0.0,
            "samples": 0,
            "max_relative_drift_m": 0.0,
            "max_relative_drift_xy_m": 0.0,
            "max_deformation_score": None,
            "max_combined_risk_score": None,
            "max_width_ratio": None,
            "unsafe_seen": False,
            "warning_seen": False,
            "slip_warning_seen": False,
            "slip_critical_seen": False,
            "effort_loss_seen": False,
            "table_drop_seen": False,
            "abort_requested": False,
            "abort_reason": None,
        }

        summary.update({
            "ran": True,
            "target_effort_sim": target_effort,
            "target_effort_source": "shear_reference_override" if target_effort_override_sim is not None else "config_or_base_plus_boost",
            "shear_reference": json_safe(shear_reference),
            "base_target_effort_sim": base_target,
            "deadband_sim": deadband,
            "max_effort_sim": max_effort,
            "sample_stride_frames": sample_stride,
            "delta_step_m": delta_step,
            "max_total_close_m": max_total_close,
            "max_total_open_m": max_total_open,
            "slip_warn_relative_drift_m": slip_warn_m,
            "slip_close_relative_drift_m": slip_close_m,
            "slip_critical_relative_drift_m": slip_critical_m,
            "deformation_warn_score": deformation_warn_score,
            "slip_close_effort_margin_sim": slip_close_effort_margin,
            "slip_close_max_shape_ratio": slip_close_max_shape_ratio,
            "disable_relax_during_motion": disable_relax_during_motion,
            "abort_on_critical_slip": abort_on_critical_slip,
            "abort_on_effort_loss": abort_on_effort_loss,
            "abort_on_table_drop": abort_on_table_drop,
            "effort_loss_threshold_sim": effort_loss_threshold,
            "min_table_gap_during_hold_m": min_table_gap_during_hold,
            "emergency_close_step_m": emergency_close_step,
            "initial_object_pos": initial_obj,
            "initial_flange_pos": initial_flange,
            "initial_object_flange_relative_pos": initial_rel,
        })

        def _safe_float(v):
            try:
                return float(v)
            except Exception:
                return None

        def _update_count(action: str):
            counts = summary.setdefault("action_counts", {})
            counts[action] = counts.get(action, 0) + 1

        def _callback():
            # Always keep the gripper state machine alive.
            self.gripper.update()
            state["frame"] += 1
            frame = state["frame"]

            if (frame % sample_stride) != 0:
                return None

            obs = None
            soft_obs = None
            safety = None
            effort = None
            effort_error = None
            relative_drift = None
            relative_drift_xy = None
            width_ratio_x = None
            width_ratio_y = None
            max_width_ratio = None
            transport_shape_ratio = None
            transport_shape_ratio_source = None
            deformation_score = None
            combined_risk_score = None
            table_gap_m = None
            action = "hold"
            requested_delta = 0.0
            adjust = {"applied": False, "reason": "hold", "requested_delta_close_m": 0.0}

            try:
                obs = self.force_observer.observe(stage_name=f"place_transport_admittance_frame_{frame}")
            except Exception as e:
                obs = {"available": False, "reason": f"force_observer_exception: {e}"}

            try:
                soft_obs = self._observe_target(target, stage_name=f"place_transport_admittance_frame_{frame}")
            except Exception as e:
                soft_obs = {"available": False, "reason": f"soft_observer_exception: {e}"}

            # Relative object/gripper drift.  This is not the absolute transport
            # distance; it is motion of the held object inside/with respect to the
            # gripper frame proxy.
            try:
                cur_obj = (soft_obs or {}).get("center") if isinstance(soft_obs, dict) else None
                cur_flange = self.arm.get_flange_world_pos()
                if initial_rel is not None and cur_obj is not None and cur_flange is not None:
                    cur_rel = [
                        float(cur_obj[0]) - float(cur_flange[0]),
                        float(cur_obj[1]) - float(cur_flange[1]),
                        float(cur_obj[2]) - float(cur_flange[2]),
                    ]
                    dx = cur_rel[0] - initial_rel[0]
                    dy = cur_rel[1] - initial_rel[1]
                    dz = cur_rel[2] - initial_rel[2]
                    relative_drift = (dx * dx + dy * dy + dz * dz) ** 0.5
                    relative_drift_xy = (dx * dx + dy * dy) ** 0.5
                    state["max_relative_drift_m"] = max(state["max_relative_drift_m"], relative_drift)
                    state["max_relative_drift_xy_m"] = max(state["max_relative_drift_xy_m"], relative_drift_xy)
            except Exception:
                pass

            try:
                table_gap_m = _safe_float((soft_obs or {}).get("table_gap_m"))
            except Exception:
                table_gap_m = None

            try:
                width_ratio_x = _safe_float((soft_obs or {}).get("width_x_m")) / float(self.config.get("adaptive_safety_nominal_width_m", 0.040))
                width_ratio_y = _safe_float((soft_obs or {}).get("width_y_m")) / float(self.config.get("adaptive_safety_nominal_depth_m", 0.040))
                max_width_ratio = max(width_ratio_x, width_ratio_y)
                old = state.get("max_width_ratio")
                state["max_width_ratio"] = max_width_ratio if old is None else max(old, max_width_ratio)
                if bool(self.config.get("place_transport_use_oriented_shape_for_deformation", True)) and (soft_obs or {}).get("oriented_max_ratio") is not None:
                    transport_shape_ratio = _safe_float((soft_obs or {}).get("oriented_max_ratio"))
                    transport_shape_ratio_source = "pca_oriented_bbox"
                else:
                    transport_shape_ratio = max_width_ratio
                    transport_shape_ratio_source = "world_aabb"
            except Exception:
                pass

            if obs and obs.get("available"):
                effort = _safe_float(obs.get("grip_effort_sim")) or 0.0
                effort_error = target_effort - effort
                try:
                    safety = self.safety_monitor.assess(
                        soft_obs=soft_obs,
                        effort_obs=obs,
                        context={
                            "frame": frame,
                            "target_effort_sim": target_effort,
                            "max_effort_sim": max_effort,
                            "controller_phase": "place_transport_admittance",
                            "relative_object_flange_drift_m": relative_drift,
                            "relative_object_flange_drift_xy_m": relative_drift_xy,
                        },
                    )
                except Exception as e:
                    safety = {
                        "available": False,
                        "safe_to_continue": True,
                        "recommended_action": "continue",
                        "reason": f"safety_monitor_exception: {e}",
                    }

                deformation_score = _safe_float((safety or {}).get("deformation_score"))
                combined_risk_score = _safe_float((safety or {}).get("combined_risk_score"))
                if deformation_score is not None:
                    old = state.get("max_deformation_score")
                    state["max_deformation_score"] = deformation_score if old is None else max(old, deformation_score)
                if combined_risk_score is not None:
                    old = state.get("max_combined_risk_score")
                    state["max_combined_risk_score"] = combined_risk_score if old is None else max(old, combined_risk_score)

                safety_action = (safety or {}).get("recommended_action")
                unsafe = bool((safety or {}).get("unsafe", False))
                warning = bool((safety or {}).get("warning", False))
                state["unsafe_seen"] = state["unsafe_seen"] or unsafe
                state["warning_seen"] = state["warning_seen"] or warning

                if relative_drift is not None and relative_drift >= slip_warn_m:
                    state["slip_warning_seen"] = True
                if relative_drift is not None and relative_drift >= slip_critical_m:
                    state["slip_critical_seen"] = True

                # Phase 4.1e decision order: transport is different from static grasping.
                # Opening while the arm is moving can turn a small slip into a drop.
                # Therefore critical slip/fall requests an arm abort and emergency hold,
                # while normal deformation warnings are logged but do not automatically
                # relax the gripper during transport.
                effort_loss = bool(effort <= effort_loss_threshold and frame > sample_stride * 3)
                table_drop = bool(table_gap_m is not None and table_gap_m < min_table_gap_during_hold)
                critical_slip = bool(relative_drift is not None and relative_drift >= slip_critical_m)

                if effort_loss:
                    state["effort_loss_seen"] = True
                if table_drop:
                    state["table_drop_seen"] = True

                should_abort = (abort_on_critical_slip and critical_slip) or (abort_on_effort_loss and effort_loss) or (abort_on_table_drop and table_drop)

                if should_abort:
                    state["abort_requested"] = True
                    if table_drop:
                        state["abort_reason"] = "object_dropped_or_touched_table_during_transport"
                    elif effort_loss:
                        state["abort_reason"] = "transport_effort_loss"
                    else:
                        state["abort_reason"] = "critical_relative_slip"
                    requested_delta = emergency_close_step
                    action = "transport_abort_emergency_hold"
                elif effort >= max_effort:
                    # Too much force while moving: stop squeezing and request abort.
                    state["abort_requested"] = True
                    state["abort_reason"] = "transport_effort_over_max"
                    requested_delta = 0.0
                    action = "transport_abort_over_effort"
                elif relative_drift is not None and relative_drift >= slip_close_m:
                    # Phase 4.1i: do not keep squeezing just because drift exists.
                    # If the effort is already above the shear target, or the
                    # object is already visibly elongated, more closure tends to
                    # create extrusion rather than useful anti-slip friction.
                    effort_ok_for_extra_close = effort <= (target_effort + slip_close_effort_margin)
                    shape_ok_for_extra_close = (
                        transport_shape_ratio is None or transport_shape_ratio <= slip_close_max_shape_ratio
                    )
                    if effort_ok_for_extra_close and shape_ok_for_extra_close:
                        requested_delta = emergency_close_step
                        action = "transport_close_from_relative_slip"
                    else:
                        requested_delta = 0.0
                        if not effort_ok_for_extra_close:
                            action = "transport_slip_high_effort_no_extra_close"
                        else:
                            action = "transport_slip_shape_limit_no_extra_close"
                elif effort_error is not None and effort_error > deadband:
                    requested_delta = delta_step
                    action = "transport_close_from_low_effort"
                elif effort_error is not None and effort_error < -deadband:
                    if disable_relax_during_motion:
                        requested_delta = 0.0
                        action = "transport_hold_high_effort_no_open_while_moving"
                    else:
                        requested_delta = -delta_step
                        action = "transport_relax_from_high_effort"
                else:
                    requested_delta = 0.0
                    action = "transport_hold_near_target"

                # Clamp cumulative correction.
                desired = state["applied_x_m"] + requested_delta
                desired = max(-max_total_open, min(max_total_close, desired))
                requested_delta = desired - state["applied_x_m"]
                if abs(requested_delta) > 1.0e-9:
                    adjust = self.gripper.adjust_hold_targets(
                        delta_close_m=requested_delta,
                        reason=action,
                    )
                    if adjust.get("applied"):
                        state["applied_x_m"] += requested_delta
                else:
                    adjust = {
                        "applied": False,
                        "reason": action,
                        "requested_delta_close_m": 0.0,
                    }
            else:
                action = "transport_hold_no_force_observation"

            state["samples"] += 1
            _update_count(action)
            summary["trace"].append(json_safe({
                "frame": frame,
                "action": action,
                "effort_sim": effort,
                "effort_error_sim": effort_error,
                "target_effort_sim": target_effort,
                "delta_close_m": requested_delta,
                "applied_x_m": state["applied_x_m"],
                "relative_object_flange_drift_m": relative_drift,
                "relative_object_flange_drift_xy_m": relative_drift_xy,
                "width_ratio_x": width_ratio_x,
                "width_ratio_y": width_ratio_y,
                "max_width_ratio": max_width_ratio,
                "transport_shape_ratio": transport_shape_ratio,
                "transport_shape_ratio_source": transport_shape_ratio_source,
                "deformation_score": deformation_score,
                "combined_risk_score": combined_risk_score,
                "table_gap_m": table_gap_m,
                "abort_requested": state["abort_requested"],
                "abort_reason": state["abort_reason"],
                "adjustment": adjust,
                "observation": obs,
                "soft_observation": soft_obs,
                "safety_assessment": safety,
            }))
            summary.update(json_safe({
                "frames_seen": state["frame"],
                "samples": state["samples"],
                "final_applied_x_m": state["applied_x_m"],
                "max_relative_object_flange_drift_m": state["max_relative_drift_m"],
                "max_relative_object_flange_drift_xy_m": state["max_relative_drift_xy_m"],
                "max_deformation_score": state["max_deformation_score"],
                "max_combined_risk_score": state["max_combined_risk_score"],
                "max_width_ratio": state["max_width_ratio"],
                "unsafe_seen": state["unsafe_seen"],
                "warning_seen": state["warning_seen"],
                "slip_warning_seen": state["slip_warning_seen"],
                "slip_critical_seen": state["slip_critical_seen"],
                "effort_loss_seen": state["effort_loss_seen"],
                "table_drop_seen": state["table_drop_seen"],
                "abort_requested": state["abort_requested"],
                "abort_reason": state["abort_reason"],
            }))
            if state["abort_requested"]:
                return {"abort": True, "reason": state["abort_reason"]}
            return None

        return _callback, summary


    async def _hold_for_inspection(self, seconds: float = None):
        if not self.config.get("debug_hold_after_stage", False):
            return


        if seconds is None:
            seconds = float(self.config.get("inspection_hold_seconds", 2.0))

        app = omni.kit.app.get_app()
        frames = max(1, int(seconds * 60))

        for _ in range(frames):
            self.gripper.update()
            await app.next_update_async()
            
    # helper to get object position from prim path
    def _get_prim_world_pos(self, prim_path: str):
        stage = omni.usd.get_context().get_stage()    # get the stage
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path)) # get the prim from prim path

        if not prim.IsValid(): # if the prim is not valid, print an error message and return None
            print(f"[Executor] ⚠️ Prim not found: {prim_path}")
            return None

        # Imported deformable USD assets are wrapped in an Xform whose root
        # transform is bottom-centre, not object-centre.  Their visual/simulation
        # mesh can also deform without a rigid-body pose update.  For these
        # targets we use the composed bounding-box centre as the best available
        # runtime observation.
        try:
            pose_attr = prim.GetAttribute("b2b:poseSource")
            pose_source = pose_attr.Get() if pose_attr and pose_attr.IsValid() else None
            if pose_source == "bbox_center":
                bbox_cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    useExtentsHint=False,
                )
                box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                mn = box.GetMin()
                mx = box.GetMax()
                return [
                    float((mn[0] + mx[0]) / 2.0),
                    float((mn[1] + mx[1]) / 2.0),
                    float((mn[2] + mx[2]) / 2.0),
                ]
        except Exception as e:
            print(f"[Executor] ⚠️ bbox_center pose read failed for {prim_path}: {e}")

        xf = UsdGeom.Xformable(prim) # get the xformable interface from the prim
        mtx = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) # compute the local to world transform
        p = mtx.ExtractTranslation() # extract the translation from the transform

        return [float(p[0]), float(p[1]), float(p[2])]

    def _is_soft_target(self, target: dict) -> bool:
        return self.soft_observer.is_soft_target(target)

    def _observe_target(self, target: dict, stage_name: str = "observe"):
        """Return live soft-object observation when available.

        For deformable USD objects this is the perceptual state used by
        planning and validation.  Rigid targets return None and keep the old
        root-transform behaviour.
        """
        if not self._is_soft_target(target):
            return None
        obs = self.soft_observer.observe(target)
        if obs:
            print(
                f"[SoftObserver:{stage_name}] "
                f"source={obs.get('pose_source')} "
                f"center=({obs['center'][0]:.4f},{obs['center'][1]:.4f},{obs['center'][2]:.4f}) "
                f"bottom_z={obs.get('bottom_z'):.4f} top_z={obs.get('top_z'):.4f} "
                f"h={obs.get('height_m'):.4f} table_gap={obs.get('table_gap_m')} "
                f"warnings={obs.get('warnings', [])}"
            )
        else:
            print(f"[SoftObserver:{stage_name}] unavailable; falling back to root transform")
        return obs

    def _get_observed_object_pos(self, target: dict, stage_name: str = "object"):
        """Return grasp-relevant object centre.

        Soft objects use SoftObjectObserver's visible-bbox centre.  Rigid
        objects use the original root transform.
        """
        obs = self._observe_target(target, stage_name=stage_name)
        if obs and obs.get("center") is not None:
            return list(obs["center"])
        prim_path = target.get("prim_path") if isinstance(target, dict) else None
        return self._get_prim_world_pos(prim_path)

    @staticmethod
    def _vec_delta_z(after, before):
        try:
            return float(after[2]) - float(before[2])
        except Exception:
            return None

    @staticmethod
    def _point_dist(a, b):
        try:
            return float(((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2) ** 0.5)
        except Exception:
            return None

    def _make_soft_micro_lift_step_callback(self, target: dict, trace: list):
        """Return a step callback that updates the gripper and samples soft state.

        This is intentionally diagnostic.  For PhysX deformables, USD BBoxCache
        may remain static while the object visibly moves.  Sampling during the
        motion lets the log prove whether our current observation source is
        dynamically updating or not.
        """
        sample_every = max(1, int(self.config.get("soft_motion_trace_sample_every_steps", 10)))
        counter = {"i": 0}

        def _callback():
            self.gripper.update()
            i = counter["i"]
            counter["i"] += 1
            if i % sample_every != 0:
                return

            obs = None
            try:
                obs = self._observe_target(target, stage_name=f"micro_lift_trace_{i}")
            except Exception as e:
                obs = {"error": str(e)}

            capture = None
            try:
                capture = self.arm.get_calibrated_capture_geometry_world()
            except Exception as e:
                capture = {"error": str(e)}

            gripper_diag = None
            try:
                gripper_diag = self.gripper.get_diagnostics()
            except Exception as e:
                gripper_diag = {"error": str(e)}

            trace.append(json_safe({
                "step": i,
                "soft_center": (obs or {}).get("center") if isinstance(obs, dict) else None,
                "soft_bottom_z": (obs or {}).get("bottom_z") if isinstance(obs, dict) else None,
                "soft_top_z": (obs or {}).get("top_z") if isinstance(obs, dict) else None,
                "pose_source": (obs or {}).get("pose_source") if isinstance(obs, dict) else None,
                "capture_geometry": capture,
                "gripper_has_object": self.gripper.has_object(),
                "gripper_diagnostics": gripper_diag,
            }))

        return _callback

    def _assess_soft_motion_observer_reliability(
        self,
        *,
        target: dict,
        soft_obs_before: dict,
        soft_obs_after: dict,
        geometry_before: dict,
        geometry_after: dict,
        gripper_has_object: bool,
        trace: list,
    ) -> dict:
        """Classify whether USD-bbox motion observation is usable for soft objects."""
        is_soft = self._is_soft_target(target)
        object_before = (soft_obs_before or {}).get("center")
        object_after = (soft_obs_after or {}).get("center")
        flange_before = (geometry_before or {}).get("flange_world_pos")
        flange_after = (geometry_after or {}).get("flange_world_pos")
        grasp_before = (geometry_before or {}).get("grasp_centre_world_pos")
        grasp_after = (geometry_after or {}).get("grasp_centre_world_pos")

        object_dz = self._vec_delta_z(object_after, object_before)
        flange_dz = self._vec_delta_z(flange_after, flange_before)
        grasp_dz = self._vec_delta_z(grasp_after, grasp_before)

        centers = [s.get("soft_center") for s in trace if isinstance(s, dict) and s.get("soft_center") is not None]
        trace_z_values = []
        for c in centers:
            try:
                trace_z_values.append(float(c[2]))
            except Exception:
                pass
        trace_z_span = (max(trace_z_values) - min(trace_z_values)) if trace_z_values else None

        min_commanded = float(self.config.get("min_commanded_micro_lift_delta_m", 0.015))
        static_eps = float(self.config.get("soft_observer_static_motion_epsilon_m", 0.0015))
        reliable = True
        status = "RELIABLE_OR_NOT_SOFT"
        reasons = []

        if is_soft:
            status = "USD_BBOX_MOTION_OBSERVER_OK"
            if object_before is None or object_after is None:
                reliable = False
                status = "SOFT_OBJECT_OBSERVER_MISSING_MEASUREMENTS"
                reasons.append("soft object observation before/after micro-lift is missing")
            elif flange_dz is not None and flange_dz >= min_commanded:
                if object_dz is not None and abs(object_dz) <= static_eps:
                    # The gripper/flange moved, but USD bbox did not.  When the gripper
                    # also reports object capture, treat this as an observer limitation
                    # rather than physical proof that the object failed to follow.
                    reliable = False
                    status = "SOFT_USD_BBOX_STATIC_DURING_LIFT"
                    reasons.append(
                        f"soft USD bbox dz={object_dz:.4f} m while flange dz={flange_dz:.4f} m"
                    )
                    if gripper_has_object:
                        reasons.append("gripper_has_object true while USD bbox remained static")
                if trace_z_span is not None and trace_z_span <= static_eps:
                    reasons.append(f"soft motion trace z-span {trace_z_span:.4f} m indicates static USD bbox source")

        return json_safe({
            "is_soft_target": is_soft,
            "motion_reliable": reliable,
            "observer_status": status,
            "reasons": reasons,
            "object_dz_m": object_dz,
            "flange_dz_m": flange_dz,
            "grasp_centre_dz_m": grasp_dz,
            "trace_sample_count": len(trace or []),
            "trace_z_span_m": trace_z_span,
            "static_motion_epsilon_m": static_eps,
            "min_commanded_micro_lift_delta_m": min_commanded,
        })

    # ─────────────────────────────────────────────────────────────────────
    # Helper to always  reset robot before the trial starts
    # ─────────────────────────────────────────────────────────────────────
    async def reset_robot_for_trial(self):
        """
        Clean robot/gripper reset for repeated Isaac Script Editor runs.

        Important:
        Do NOT step physics for the gripper before commanding the arm.
        Otherwise old UR5e drive targets from the previous run may move the arm.
        """
        print("\n[Executor] Resetting robot for trial...")

        # 1. Tell gripper to open, but do NOT step it alone yet.
        # The gripper will update while the arm goes home.
        print("[Executor] Reset: command gripper open, no pre-home stepping.")
        
        print("[TRACE_RESET] before gripper.open")
        self.gripper.open()
        print("[TRACE_RESET] after gripper.open, before arm.move_home")

        # 2. Immediately command arm home slowly.
        print("[Executor] Reset: moving arm home immediately...")
        await self.arm.move_home(
            duration=float(self.config.get("startup_home_duration", 6.0)),
            steps=int(self.config.get("startup_home_steps", 360)),
        )

        # 3. Now let the gripper finish opening and physics settle.
        print("[Executor] Reset: settling after home...")

        print("[TRACE_RESET] after arm.move_home, before post-home settle")
        await self._step_gripper_for_seconds(
            float(self.config.get("post_home_settle_seconds", 0.8))
        )
        print("[TRACE_RESET] after post-home settle")

        print("[Executor] Robot reset complete.")
        
    # ─────────────────────────────────────────────────────────────────────────
    # helper to reset object to its original position
    # ─────────────────────────────────────────────────────────────────────────
    async def reset_object_for_trial(self):
        """
        Smooth reset at the beginning of a trial.

        Assumes preplay sync already prevented stale-drive wake-up.
        """
        print("\n[Executor] Resetting robot for trial...")

        # Command gripper open but do not do long pre-home stepping.
        print("[Executor] Reset: command gripper open.")
        self.gripper.open()

        print("[Executor] Reset: moving arm home smoothly...")
        await self.arm.move_home(
            duration=float(self.config.get("startup_home_duration", 6.0)),
            steps=int(self.config.get("startup_home_steps", 360)),
        )

        print("[Executor] Reset: settling after home...")
        await self._step_gripper_for_seconds(
            float(self.config.get("post_home_settle_seconds", 0.8))
        )

        print("[Executor] Robot reset complete.")
        
    # ─────────────────────────────────────────────────────────────────────────
    # helper to park robot after trial
    # ─────────────────────────────────────────────────────────────────────────
    async def park_robot_after_trial(self):
        """
        Park robot safely at the end of a trial.

        This reduces stale target problems in the next run.
        """
        if not self.config.get("park_robot_after_trial", True):
            return

        print("\n[Executor] Parking robot after trial...")

        self.gripper.open()
        await self._step_gripper_for_seconds(
            float(self.config.get("park_gripper_open_seconds", 0.8))
        )

        await self.arm.move_home(
            duration=float(self.config.get("park_home_duration", 5.0)),
            steps=int(self.config.get("park_home_steps", 300)),
        )

        await self._step_gripper_for_seconds(
            float(self.config.get("park_settle_seconds", 0.5))
        )

        print("[Executor] Robot parked.")
    
    #-------------------------------
    #  helpers to check grasp success:-
    # -------------------------------
    # 1. calculate distance between two points
    def _distance(self, a, b):
        import math
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def _last_known_object_pos_from_attempt(self, attempt_log: dict):
        """Return the last object position observed inside a previous attempt."""
        if not attempt_log:
            return None

        micro = attempt_log.get("micro_lift_validation") or {}
        if micro.get("object_pos_after") is not None:
            return micro.get("object_pos_after")

        close = attempt_log.get("close_validation") or {}
        if close.get("object_pos_after") is not None:
            return close.get("object_pos_after")

        if attempt_log.get("actual_object_pos") is not None:
            return attempt_log.get("actual_object_pos")

        return None

    def _object_retry_displacement_summary(self, current_pos, previous_attempt: dict):
        """Log how far the object moved between retry attempts."""
        previous_pos = self._last_known_object_pos_from_attempt(previous_attempt)

        summary = {
            "previous_last_known_object_pos": previous_pos,
            "current_refreshed_object_pos": current_pos,
            "displacement_since_previous_attempt_m": None,
            "horizontal_displacement_since_previous_attempt_m": None,
            "retry_used_refreshed_pose": current_pos is not None,
        }

        if previous_pos is None or current_pos is None:
            return summary

        summary["displacement_since_previous_attempt_m"] = self._distance(
            previous_pos,
            current_pos,
        )
        summary["horizontal_displacement_since_previous_attempt_m"] = (
            (
                (float(previous_pos[0]) - float(current_pos[0])) ** 2
                + (float(previous_pos[1]) - float(current_pos[1])) ** 2
            ) ** 0.5
        )
        return summary

    # 2. helper to validate grasp just after closing gripper
    def validate_after_close(self, target, object_pos_before_close):
        object_pos_after = self._get_observed_object_pos(target, stage_name="after_close")
        flange_pos = self.arm.get_flange_world_pos()
        soft_obs_after = self._observe_target(target, stage_name="after_close_validation")

        result = {
            "stage": "after_close",
            "validation_mode": "soft_bbox" if self._is_soft_target(target) else "rigid_root",
            "gripper_has_object": self.gripper.has_object(),
            "object_pos_before": object_pos_before_close,
            "object_pos_after": object_pos_after,
            "soft_observation_after": soft_obs_after,
            "flange_pos": flange_pos,
            "object_shift": None,
            "flange_object_distance": None,
            "transport_shape_safety": None,
            "success": False,
            "reasons": [],
            "warnings": [],
        }

        if object_pos_before_close is None or object_pos_after is None:
            result["reasons"].append("missing object pose")
            return result

        shift = self._distance(object_pos_before_close, object_pos_after)
        dist = self._distance(object_pos_after, flange_pos)

        result["object_shift"] = shift
        result["flange_object_distance"] = dist

        if self._is_soft_target(target) and soft_obs_after:
            result["soft_metrics"] = {
                "height_m": soft_obs_after.get("height_m"),
                "deformation_ratio_z": soft_obs_after.get("deformation_ratio_z"),
                "table_gap_m": soft_obs_after.get("table_gap_m"),
                "collision_visible_bottom_offset_m": soft_obs_after.get("collision_visible_bottom_offset_m"),
                "warnings": soft_obs_after.get("warnings", []),
            }

        max_shift = float(self.config.get("max_object_shift_during_close_m", 0.03))
        max_dist = float(self.config.get("max_object_flange_distance_after_close_m", 0.25))

        if not result["gripper_has_object"]:
            result["reasons"].append("gripper_has_object false")

        if shift > max_shift:
            result["reasons"].append(
                f"object shifted too much during close: {shift:.3f} > {max_shift:.3f}"
            )

        if dist > max_dist:
            result["reasons"].append(
                f"object too far from flange after close: {dist:.3f} > {max_dist:.3f}"
            )

        if self._is_soft_target(target) and soft_obs_after:
            max_table_gap = float(self.config.get("max_soft_table_gap_m", 0.003))
            table_gap = soft_obs_after.get("table_gap_m")
            if table_gap is not None and table_gap > max_table_gap:
                result["reasons"].append(
                    f"soft object not in table/contact-consistent pose: table_gap={table_gap:.4f} > {max_table_gap:.4f}"
                )

        result["success"] = len(result["reasons"]) == 0
        return result

    # 3. helper to validate grasp after lift
    def validate_after_lift(self, target, object_pos_before_lift, table_height):
        object_pos_after = self._get_observed_object_pos(target, stage_name="after_full_lift")
        flange_pos = self.arm.get_flange_world_pos()
        soft_obs_after = self._observe_target(target, stage_name="after_full_lift_validation")

        result = {
            "stage": "after_lift",
            "validation_mode": "soft_bbox" if self._is_soft_target(target) else "rigid_root",
            "gripper_has_object": self.gripper.has_object(),
            "object_pos_before_lift": object_pos_before_lift,
            "object_pos_after_lift": object_pos_after,
            "soft_observation_after": soft_obs_after,
            "flange_pos": flange_pos,
            "object_lift_delta_z": None,
            "flange_object_distance": None,
            "thresholds": {},
            "warnings": [],
            "success": False,
            "reasons": [],
        }

        if object_pos_before_lift is None or object_pos_after is None:
            result["reasons"].append("missing object pose")
            return result

        dz = object_pos_after[2] - object_pos_before_lift[2]
        dist = self._distance(object_pos_after, flange_pos)

        result["object_lift_delta_z"] = dz
        result["flange_object_distance"] = dist

        min_lift_delta = float(self.config.get("min_success_lift_delta_m", 0.04))
        min_above_table = float(self.config.get("min_object_above_table_m", 0.03))
        max_dist = float(self.config.get("max_object_flange_distance_after_lift_m", 0.25))

        if not result["gripper_has_object"]:
            result["reasons"].append("gripper_has_object false")

        if dz < min_lift_delta:
            result["reasons"].append(
                f"object did not lift enough: dz={dz:.3f} < {min_lift_delta:.3f}"
            )

        if object_pos_after[2] < table_height + min_above_table:
            result["reasons"].append(
                f"object not above table enough: z={object_pos_after[2]:.3f}"
            )

        if dist > max_dist:
            result["reasons"].append(
                f"object too far from flange after lift: {dist:.3f} > {max_dist:.3f}"
            )

        result["success"] = len(result["reasons"]) == 0
        return result

    def _get_table_zone(self, scene_info: dict, zone_key: str) -> dict:
        """Return a semantic table zone record from SceneBuilder output.

        Phase 4 uses the zone as a goal region for placing, not as the grasp
        target.  Grasping remains object-observation based through
        SoftObjectObserver / simulation_mesh points.
        """
        if not isinstance(scene_info, dict):
            return {}
        zones = scene_info.get("table_zones") or (scene_info.get("table_info") or {}).get("zones") or {}
        return zones.get(zone_key, {}) if isinstance(zones, dict) else {}

    def _estimate_object_height_for_place(self, target: dict) -> float:
        """Estimate object height for center-on-table placement target."""
        obs = self._observe_target(target, stage_name="place_height_estimate")
        if obs and obs.get("height_m") is not None:
            return float(obs.get("height_m"))
        for key in ("height", "height_m", "size"):
            if isinstance(target, dict) and target.get(key) is not None:
                try:
                    return float(target.get(key))
                except Exception:
                    pass
        return float(self.config.get("place_default_object_height_m", 0.04))

    def _compute_place_object_center(self, scene_info: dict, target: dict, table_height: float) -> dict:
        """Compute the desired object-center position over the place zone.

        The place zone is a semantic goal region.  The desired final object
        center uses the place marker XY and the current/estimated object height.
        """
        place_zone = self._get_table_zone(scene_info, "place_zone")
        if not place_zone:
            return {
                "available": False,
                "reason": "missing_place_zone",
                "place_zone": {},
                "place_object_center": None,
            }

        world_center = place_zone.get("world_center") or place_zone.get("world_center_m")
        if not world_center or len(world_center) < 2:
            return {
                "available": False,
                "reason": "place_zone_missing_world_center",
                "place_zone": place_zone,
                "place_object_center": None,
            }

        object_height = self._estimate_object_height_for_place(target)
        place_center = [
            float(world_center[0]),
            float(world_center[1]),
            float(table_height) + 0.5 * object_height,
        ]
        return {
            "available": True,
            "reason": None,
            "place_zone": place_zone,
            "object_height_m": object_height,
            "place_object_center": place_center,
            "semantic_note": "place zone defines desired release region; grasp target remains live object pose",
        }

    def validate_after_place_transport(
        self,
        *,
        target: dict,
        place_zone: dict,
        object_pos_before_transport,
        object_pos_after_transport,
        table_height: float,
    ) -> dict:
        """Validate Phase 4.1 transport while still holding the object.

        This does not validate release yet.  It only verifies that the object is
        still held and its XY center has moved above/near the place zone.
        """
        flange_pos = self.arm.get_flange_world_pos()
        soft_obs_after = self._observe_target(target, stage_name="after_place_transport_validation")
        result = {
            "stage": "after_place_transport",
            "phase": "4.1_transport_to_place_zone_no_release",
            "validation_mode": "soft_bbox" if self._is_soft_target(target) else "rigid_root",
            "gripper_has_object": self.gripper.has_object(),
            "place_zone": place_zone,
            "object_pos_before_transport": object_pos_before_transport,
            "object_pos_after_transport": object_pos_after_transport,
            "soft_observation_after": soft_obs_after,
            "flange_pos": flange_pos,
            "place_zone_xy_error_m": None,
            "object_transport_xy_delta_m": None,
            "flange_object_distance": None,
            "thresholds": {},
            "warnings": [],
            "success": False,
            "reasons": [],
        }

        if object_pos_after_transport is None:
            result["reasons"].append("missing object pose after place transport")
            return result

        world_center = place_zone.get("world_center") or []
        if len(world_center) < 2:
            result["reasons"].append("place zone world center unavailable")
            return result

        dx = float(object_pos_after_transport[0]) - float(world_center[0])
        dy = float(object_pos_after_transport[1]) - float(world_center[1])
        xy_err = (dx * dx + dy * dy) ** 0.5
        result["place_zone_xy_error_m"] = xy_err

        if object_pos_before_transport is not None:
            tx = float(object_pos_after_transport[0]) - float(object_pos_before_transport[0])
            ty = float(object_pos_after_transport[1]) - float(object_pos_before_transport[1])
            result["object_transport_xy_delta_m"] = (tx * tx + ty * ty) ** 0.5

        if flange_pos is not None:
            result["flange_object_distance"] = self._distance(object_pos_after_transport, flange_pos)

        zone_size = place_zone.get("size_xy_m", [0.08, 0.08])
        half_zone = min(float(zone_size[0]), float(zone_size[1])) * 0.5 if len(zone_size) >= 2 else 0.04
        tolerance = float(self.config.get("place_transport_xy_tolerance_m", half_zone + 0.025))
        max_flange_dist = float(self.config.get("max_object_flange_distance_after_transport_m", 0.28))
        min_above_table = float(self.config.get("min_object_above_table_after_transport_m", 0.03))

        max_width_ratio = float(self.config.get("place_transport_max_width_ratio", 1.35))
        max_deformation_score = float(self.config.get("place_transport_max_deformation_score", 0.25))
        fail_on_deformation = bool(self.config.get("place_transport_fail_on_excessive_deformation", True))
        result["thresholds"].update({
            "place_transport_max_width_ratio": max_width_ratio,
            "place_transport_max_deformation_score": max_deformation_score,
            "place_transport_fail_on_excessive_deformation": fail_on_deformation,
        })

        # Phase 4.1h: shape validation should not confuse rigid rotation with
        # true soft-body deformation.  Prefer the PCA-oriented bbox metric when
        # available; log both world-AABB and OBB metrics for research analysis.
        if isinstance(soft_obs_after, dict):
            wx_ratio = soft_obs_after.get("width_ratio_x")
            wy_ratio = soft_obs_after.get("width_ratio_y")
            height = soft_obs_after.get("height_m")
            nominal_h = float(self.config.get("adaptive_safety_nominal_height_m", 0.040))
            height_ratio = None
            try:
                if height is not None and nominal_h > 1.0e-9:
                    height_ratio = float(height) / nominal_h
            except Exception:
                height_ratio = None

            world_ratios = []
            for v in (wx_ratio, wy_ratio, height_ratio):
                try:
                    if v is not None:
                        world_ratios.append(float(v))
                except Exception:
                    pass
            world_max_ratio = max(world_ratios) if world_ratios else None
            world_deformation_score = max(abs(r - 1.0) for r in world_ratios) if world_ratios else None

            use_oriented = bool(self.config.get("place_transport_use_oriented_shape_for_deformation", True))
            oriented_max_ratio = soft_obs_after.get("oriented_max_ratio")
            oriented_ratio_sorted = soft_obs_after.get("oriented_ratio_sorted")
            oriented_bbox = soft_obs_after.get("oriented_bbox")

            max_ratio = world_max_ratio
            deformation_score = soft_obs_after.get("deformation_score")
            shape_metric_source = "world_aabb"
            if use_oriented and oriented_max_ratio is not None:
                try:
                    max_ratio = float(oriented_max_ratio)
                    deformation_score = abs(max_ratio - 1.0)
                    shape_metric_source = "pca_oriented_bbox"
                except Exception:
                    pass
            if deformation_score is None:
                deformation_score = world_deformation_score

            result["transport_shape_safety"] = {
                "shape_metric_source": shape_metric_source,
                "width_ratio_x_world_aabb": wx_ratio,
                "width_ratio_y_world_aabb": wy_ratio,
                "height_ratio_world_aabb": height_ratio,
                "world_aabb_max_dimension_ratio": world_max_ratio,
                "world_aabb_deformation_score_proxy": world_deformation_score,
                "oriented_max_ratio": oriented_max_ratio,
                "oriented_ratio_sorted": oriented_ratio_sorted,
                "oriented_bbox": oriented_bbox,
                "max_dimension_ratio": max_ratio,
                "deformation_score_proxy": deformation_score,
                "interpretation": "PCA-oriented bbox helps separate cube rotation from real soft-object deformation/extrusion.",
            }
            if max_ratio is not None and max_ratio > max_width_ratio:
                msg = f"soft object stretched during transport: max_ratio={max_ratio:.3f} > {max_width_ratio:.3f}"
                if fail_on_deformation:
                    result["reasons"].append(msg)
                else:
                    result["warnings"].append(msg)
            if deformation_score is not None:
                try:
                    ds = float(deformation_score)
                    if ds > max_deformation_score:
                        msg = f"transport deformation score high: {ds:.3f} > {max_deformation_score:.3f}"
                        if fail_on_deformation:
                            result["reasons"].append(msg)
                        else:
                            result["warnings"].append(msg)
                except Exception:
                    pass

        if not result["gripper_has_object"]:
            result["reasons"].append("gripper_has_object false after place transport")
        if xy_err > tolerance:
            result["reasons"].append(
                f"object not above place zone: xy_error={xy_err:.3f} > {tolerance:.3f}"
            )
        if result["flange_object_distance"] is not None and result["flange_object_distance"] > max_flange_dist:
            result["reasons"].append(
                f"object too far from flange after transport: {result['flange_object_distance']:.3f} > {max_flange_dist:.3f}"
            )
        if float(object_pos_after_transport[2]) < float(table_height) + min_above_table:
            result["reasons"].append("object not safely above table after transport")

        result["success"] = len(result["reasons"]) == 0
        return result

    # Method to get last trial log as .json file and write it to the output directory with a timestamp and create the dir if not exists
    def get_last_trial_log(self):
        return json_safe(getattr(self, "_last_trial_log", None))

    def _adaptive_safety_outcome_from_close_resolution(self, close_resolution: dict) -> dict:
        """Classify whether adaptive safety requested attempt termination.

        Phase 3.4 proved that unsafe deformation can trigger reactive
        relaxation.  Phase 3.5 turns that event into supervisory control:
        once safety has actively opened/relaxed the gripper, the current
        attempt must not continue to micro-lift as if a secure grasp still
        existed.
        """
        close_resolution = close_resolution or {}
        reg = close_resolution.get("adaptive_effort_regulation") or {}
        safety_summary = reg.get("adaptive_safety_summary") or {}
        trace = reg.get("trace") or []

        unsafe_rows = []
        warning_rows = []
        relax_rows = []
        for row in trace:
            assess = row.get("safety_assessment") or {}
            state = assess.get("safety_state")
            rec = assess.get("recommended_action")
            action = row.get("action")
            if state == "unsafe":
                unsafe_rows.append(row)
            if state == "warning":
                warning_rows.append(row)
            if (rec and str(rec).startswith("relax_open")) or (action and str(action).startswith("safety_relax")):
                relax_rows.append(row)

        final_reason = reg.get("final_reason")
        safety_open_limit = final_reason == "safety_open_limit_reached"
        safety_abort = final_reason == "safety_abort_requested"
        requires_stop = bool(unsafe_rows or relax_rows or safety_open_limit or safety_abort)

        return {
            "phase": "3.5_safety_aware_termination",
            "available": bool(reg),
            "requires_attempt_stop": requires_stop,
            "reason": (
                "adaptive_safety_relaxation_or_unsafe_state"
                if requires_stop else
                "no_safety_termination_requested"
            ),
            "regulation_final_reason": final_reason,
            "unsafe_count": len(unsafe_rows),
            "warning_count": len(warning_rows),
            "relax_action_count": len(relax_rows),
            "safety_summary": safety_summary,
            "last_unsafe_assessment": (unsafe_rows[-1].get("safety_assessment") if unsafe_rows else None),
            "last_relax_action": (relax_rows[-1].get("action") if relax_rows else None),
            "supervisory_policy": (
                "If unsafe deformation/relaxation occurs after close, stop the current attempt, "
                "recover safely, and let RetryPolicy choose a safer retry instead of proceeding to micro-lift."
            ),
        }
    
    def _apply_retry_adjustments_to_pick_result(
        self,
        pick_result: dict,
        adjustments: dict,
    ) -> dict:
        """Apply retry-time command adjustments to a planned pick result.

        The arm planner computes the geometric plan.
        RetryPolicy may then scale the commanded grasp force for the next attempt.

        Geometry adjustment, such as grasp_z_delta_m, is handled separately
        inside arm_controller through set_runtime_grasp_z_delta().
        """
        if not pick_result:
            return pick_result

        adjustments = adjustments or {}

        force_scale = float(adjustments.get("force_scale", 1.0))

        old_force = float(pick_result.get("target_force_n", 80.0))
        min_force = float(self.config.get("min_grip_force", 40.0))
        max_force = float(self.config.get("max_grip_force", 140.0))

        new_force = max(
            min_force,
            min(max_force, old_force * force_scale),
        )

        pick_result["target_force_n_before_retry_scale"] = old_force
        pick_result["retry_force_scale"] = force_scale
        pick_result["target_force_n"] = new_force

        # Phase 3.5: safety-aware retry may ask for a less aggressive hold
        # target after an unsafe deformation relaxation.  This affects the
        # gripper contact geometry directly, while scalar admittance target
        # scaling is passed separately to close_gripper.
        if "hold_extra_close_delta_m" in adjustments:
            old_extra = float(pick_result.get("gripper_hold_extra_close_m", 0.0) or 0.0)
            delta_extra = float(adjustments.get("hold_extra_close_delta_m", 0.0) or 0.0)
            new_extra = max(0.0, old_extra + delta_extra)
            pick_result["gripper_hold_extra_close_m_before_retry_delta"] = old_extra
            pick_result["retry_hold_extra_close_delta_m"] = delta_extra
            pick_result["gripper_hold_extra_close_m"] = new_extra

        return pick_result

    async def _recover_to_safe_for_retry(
        self,
        pre_grasp,
        safe_above,
        from_micro_lift: bool = False,
    ):
        """Return to a safe configuration before another grasp attempt.

        This is a recovery fixed-action pattern.

        If failure happened after micro-lift, the arm is already above the
        object and the gripper may still be holding or partially holding
        something. In that case, move upward safely first, then open.

        If failure happened during/after close at grasp pose, open and retreat
        through pre_grasp and safe_above.
        """
        print("[Executor] Recovering safely before retry...")

        # Case A:
        # Failure happened after micro-lift.
        # The safest behaviour is to keep hold pressure while moving upward,
        # then open only at a safer height.
        if from_micro_lift:
            if safe_above is not None:
                await self.arm.move_via_safe_height(
                    safe_above,
                    duration=float(self.config.get("retry_to_safe_duration", 4.0)),
                    steps=int(self.config.get("retry_to_safe_steps", 240)),
                    step_callback=self.gripper.update,
                )

            self.gripper.open()
            await self._step_gripper_for_seconds(
                float(self.config.get("post_open_settle_seconds", 0.8))
            )
            return

        # Case B:
        # Failure happened before/during close or during close validation.
        # First give the simulator a small pause, then open the gripper.
        await self._step_gripper_for_seconds(
            float(self.config.get("post_failed_close_pause_seconds", 0.5))
        )

        self.gripper.open()
        await self._step_gripper_for_seconds(
            float(self.config.get("post_open_settle_seconds", 0.8))
        )

        # Move back upward along the approach chain.
        if pre_grasp is not None:
            await self.arm.move_to(
                pre_grasp,
                duration=float(self.config.get("retry_to_pregrasp_duration", 3.0)),
                steps=int(self.config.get("retry_to_pregrasp_steps", 180)),
                check_table_collision=True,
                step_callback=self.gripper.update,
            )

        await self._step_gripper_for_seconds(
            float(self.config.get("retry_mid_settle_seconds", 0.3))
        )

        if safe_above is not None:
            await self.arm.move_via_safe_height(
                safe_above,
                duration=float(self.config.get("retry_to_safe_duration", 4.0)),
                steps=int(self.config.get("retry_to_safe_steps", 240)),
                step_callback=self.gripper.update,
            )

    async def _wait_for_gripper_close_resolution(
        self,
        max_seconds: float = None,
    ) -> dict:
        """Wait until the gripper close action reaches a terminal state.

        We should not validate grasp success while the gripper is still CLOSING.

        The close is considered resolved when:
          - the gripper enters HOLDING after plausible contact/squeeze, or
          - the gripper enters HOLDING after a no-object timeout/full close, or
          - the timeout expires.

        Returns a diagnostic dictionary for logging.
        """
        app = omni.kit.app.get_app()

        if max_seconds is None:
            max_seconds = float(
                self.config.get("gripper_close_wait_timeout_s", 6.0)
            )

        # We use frame count because gripper.update() is called once per frame.
        # 6 seconds × 60 Hz = 360 updates, enough for:
        # max_closing_ticks ≈ 180 + squeeze_ticks ≈ 50 + margin.
        updates_per_second = float(
            self.config.get("gripper_close_wait_updates_per_second", 60.0)
        )

        max_frames = max(
            1,
            int(max_seconds * updates_per_second),
        )

        last_state = None
        frames_used = 0
        force_trace = []
        force_stride = max(1, int(self.config.get("force_observer_trace_stride_frames", 10)))

        for i in range(max_frames):
            last_state = self.gripper.update()
            frames_used = i + 1

            await app.next_update_async()

            if (i % force_stride) == 0 and hasattr(self, "force_observer"):
                try:
                    force_trace.append(
                        self.force_observer.observe(stage_name=f"during_close_frame_{frames_used}")
                    )
                except Exception as e:
                    force_trace.append({
                        "available": False,
                        "stage": f"during_close_frame_{frames_used}",
                        "reason": f"force_observer_exception: {e}",
                    })

            # The key guard:
            # Do not continue waiting once close has resolved.
            if last_state != self.gripper.CLOSING:
                diagnostics = self.gripper.get_diagnostics()
                return {
                    "resolved": True,
                    "frames_used": frames_used,
                    "max_frames": max_frames,
                    "final_state": last_state,
                    "timeout_s": max_seconds,
                    "gripper_diagnostics": diagnostics,
                    "force_trace_during_close": force_trace,
                }

        # Timeout: close did not resolve.
        diagnostics = self.gripper.get_diagnostics()
        return {
            "resolved": False,
            "frames_used": frames_used,
            "max_frames": max_frames,
            "final_state": last_state,
            "timeout_s": max_seconds,
            "gripper_diagnostics": diagnostics,
            "force_trace_during_close": force_trace,
            "reason": "gripper_close_wait_timeout",
        }
    # implement move to pregrasp_approach position (a point above target pos with constant height of pregrasp_height)
    async def run_generic_pick(self, scene_info: dict) -> bool:
        """
        Run a first generic pick attempt using the Isaac-native
        arm and gripper controllers.

        """
        # Get the pick target and table info from the scene_info, but first i need to know the structure of scene_info
        target = scene_info["pick_target"]
        table_info = scene_info["table_info"]

        # extract the object world position and table height and pan to object angle
        object_world_pos = target["world_pos"]
        table_height = scene_info.get("table_height", table_info["table_size"][2])
        pan_to_object_deg = scene_info.get("pan_to_table_deg", 0.0)

        # print information about the pick target
        print("\n[Executor] Generic pick target:")
        print(f"  label: {target.get('label')}")
        print(f"  shape: {target.get('shape')}")
        print(f"  material: {target.get('material_name')}")
        print(f"  world_pos: {object_world_pos}")
        print(f"  table_height: {table_height:.3f}")
        print(f"  pan_to_object_deg: {pan_to_object_deg:.2f}")

        self.arm.set_table_info(table_info)

        print("\n[Executor] Arm status before planning:")
        print(self.arm.get_status())

        trial_log = {
            "schema_version": "b2b_validation_v2_analysis",
            "target": {
                "label": target.get("label", "unknown"),
                "shape": target.get("shape", "unknown"),
                "material": target.get("material_name", "unknown"),
                "mass": target.get("mass", "unknown"),
                "grip_dim_mm": target.get("grip_dim_mm", "unknown"),
                "prim_path": target.get("prim_path", "unknown"),
            },
            "target_full": json_safe(target),
            "initial_soft_observation": json_safe(self._observe_target(target, stage_name="trial_start")),
            "target_feasibility": grip_feasibility(target, self.config),
            "table_info": json_safe(table_info),
            "table_zones": json_safe(scene_info.get("table_zones", table_info.get("zones", {}))),
            "config_snapshot": selected_config_snapshot(self.config),
            "events": [
                make_event(
                    "trial_log_created",
                    target_label=target.get("label", "unknown"),
                    target_shape=target.get("shape", "unknown"),
                )
            ],
            "attempts": [],
            "trial_success": False,
            "final_reason": None,
        }

        # ─────────────────────────────────────────────────────────────
        # Early affordance/range guard.
        # If the object is clearly outside the hard 2FG7 range, do not spend
        # time planning and do not pretend a pick primitive is reasonable.
        # This is action selection rejecting an unavailable motor schema.
        target_feasibility = trial_log.get("target_feasibility", {}) or {}
        if (not target_feasibility.get("within_hard_range", True)) or str(
            target_feasibility.get("grip_feasibility_class", "")
        ).startswith("infeasible"):
            print("[Executor] ❌ Target rejected before planning: outside hard gripper range.")
            trial_log["trial_success"] = False
            trial_log["final_reason"] = "target_infeasible_for_gripper_range"
            trial_log["events"].append(
                make_event(
                    "target_infeasible_for_gripper_range",
                    grip_dim_m=target_feasibility.get("estimated_object_grip_dim_m"),
                    grip_range_min_m=target_feasibility.get("grip_range_min_m"),
                    grip_range_max_m=target_feasibility.get("grip_range_max_m"),
                    grip_feasibility_class=target_feasibility.get("grip_feasibility_class"),
                )
            )
            self._last_trial_log = trial_log
            return False

        print("\n[Executor] Robot reset already handled by TrialRunner.")

        #-------------------------
        # implemented a simple pick_result using compute pick joints for now 
        # TODO:  (if possible)need to improve for robust picking using ML based model later
        max_attempts = int(self.config.get("max_grasp_attempts", 1))
        current_retry_adjustments = {}

        for attempt in range(max_attempts):
            print(f"\n[Executor] Grasp attempt {attempt + 1}/{max_attempts}")
            attempt_log = {
                "attempt": attempt + 1,
                "stored_object_pos": json_safe(target.get("world_pos")),
                "actual_object_pos": None,
                "object_displacement_since_previous_attempt": None,
                "target_feasibility": grip_feasibility(target, self.config),
                "safe_above_ok": False,
                "pre_grasp_ok": False,
                "grasp_ok": False,
                "planning_initial": None,
                "planning_replanned_after_safe_above": None,
                "preclose_diagnostics": None,
                "preclose_geometry_gate": None,
                "close_validation": None,
                "micro_lift_validation": None,
                "lift_validation": None,
                "place_transport_plan": None,
                "place_transport_shear_reference": None,
                "place_transport_shear_preload": None,
                "place_transport_validation": None,
                "gripper_diagnostics_after_close": None,
                "gripper_diagnostics_after_micro_lift": None,
                "gripper_diagnostics_after_lift": None,
                "snapshots": {},
                "events": [make_event("attempt_started", attempt=attempt + 1)],
                "success": False,
                "failure_reason": None,
            }
            attempt_log["retry_adjustments_applied"] = json_safe(
                current_retry_adjustments
            )
            self.arm.set_runtime_grasp_z_delta(
                float(current_retry_adjustments.get("grasp_z_delta_m", 0.0))
            )

            actual_object_pos = self._get_observed_object_pos(target, stage_name="before_planning")

            if actual_object_pos is None:
                actual_object_pos = target["world_pos"]

            attempt_log["actual_object_pos"] = json_safe(actual_object_pos)
            if attempt > 0 and trial_log.get("attempts"):
                attempt_log["object_displacement_since_previous_attempt"] = json_safe(
                    self._object_retry_displacement_summary(
                        actual_object_pos,
                        trial_log["attempts"][-1],
                    )
                )
                attempt_log["events"].append(
                    make_event(
                        "retry_object_pose_refreshed",
                        object_displacement_since_previous_attempt_m=(
                            attempt_log["object_displacement_since_previous_attempt"].get(
                                "displacement_since_previous_attempt_m"
                            )
                            if attempt_log.get("object_displacement_since_previous_attempt")
                            else None
                        ),
                    )
                )
            attempt_log["snapshots"]["before_planning"] = pose_snapshot(
                self, target, "before_planning"
            )

            print("[Executor] Stored object pos:", target["world_pos"])
            print("[Executor] Actual object pos:", actual_object_pos)

            # TODO(mihret): make this smarter. 
            # Instead of always computing a new grasp for every attempt, 
            # we could cache the first result and try it multiple times, 
            # only recomputing if the first attempt fails (e.g. after a failed pick or after moving home). TO BE IMPROVED LATER

            # Recompute each attempt because the arm controller may select
            # a different valid grasp orientation candidate.
            pick_result = self.arm.compute_pick_joints(
                object_world_pos=actual_object_pos,    # pass actual_object_pos instead of the stale object_world_pos in target to make it robust to small errors in target position during runtime    
                pan_to_object_deg=pan_to_object_deg,
                table_height=table_height,
                object_metadata=target,
                prim_path=target.get("prim_path"),
            )
            pick_result = self._apply_retry_adjustments_to_pick_result(
                pick_result,
                current_retry_adjustments,
            )

            # Added a break with a False return if the IK fails to produce a valid plan.
            # This ensures that the robot does not attempt to execute a failed plan.
            attempt_log["planning_initial"] = compact_pick_result(pick_result)
            attempt_log["events"].append(
                make_event(
                    "planning_initial_done",
                    planning_failed=pick_result.get("planning_failed", False),
                    target_force_n=pick_result.get("target_force_n"),
                    waypoints=list((pick_result.get("joints", {}) or {}).keys()),
                )
            )

            if pick_result.get("planning_failed", False):
                print("[Executor] ❌ Planner rejected the grasp before execution.")
                print("[Executor] IK diagnostics:")
                print(pick_result.get("ik_meta", {}))
                attempt_log["failure_reason"] = "planning_failed_before_safe_above"
                trial_log["attempts"].append(attempt_log)
                trial_log["final_reason"] = "planning_failed_before_safe_above"
                self._last_trial_log = trial_log
                return False
            joints = pick_result["joints"]

            # Print-out of the joints. This is useful for debugging
            # and understanding what the robot is doing.
            print("\n[Executor] Pick planning result:")
            print(f"  target_force_n: {pick_result.get('target_force_n')}")
            print(f"  grasp_strategy: {pick_result.get('grasp_strategy')}")
            print(f"  object_height: {pick_result.get('object_height')}")
            print(f"  IK meta: {pick_result.get('ik_meta')}")
            print(f"  waypoints: {list(pick_result.get('joints', {}).keys())}")

            joints = pick_result.get("joints", {})
            safe_above = joints.get("safe_above")
            pre_grasp = joints.get("pre_grasp")
            grasp = joints.get("grasp")


            if safe_above is None or pre_grasp is None or grasp is None:
                print("[Executor] ❌ Missing one or more required waypoints.")
                return False

            print("\n[Executor] Moving to safe_above...")
            ok = await self.arm.move_via_safe_height(
                safe_above,
                duration=3.0,
                steps=150,
            )
            if not ok:
                print("[Executor] ❌ Failed to reach safe_above.")
                attempt_log["failure_reason"] = "failed_to_reach_safe_above"
                trial_log["attempts"].append(attempt_log)
                return False

            print("[Executor] ✅ Reached safe_above.")
            attempt_log["safe_above_ok"] = True
            attempt_log["snapshots"]["after_safe_above"] = pose_snapshot(
                self, target, "after_safe_above"
            )
            attempt_log["events"].append(make_event("safe_above_reached"))

            ## read actual object pose again 
            actual_object_pos = self._get_observed_object_pos(target, stage_name="after_safe_above")
            if actual_object_pos is None:
                actual_object_pos = target["world_pos"]
            
            attempt_log["actual_object_pos"] = json_safe(actual_object_pos)
            attempt_log["retry_used_refreshed_pose_after_safe_above"] = True
            attempt_log["snapshots"]["after_safe_above_object_refresh"] = pose_snapshot(
                self, target, "after_safe_above_object_refresh"
            )
            
            pick_result = self.arm.compute_pick_joints(
                object_world_pos=actual_object_pos,    # pass actual_object_pos instead of the stale object_world_pos in target to make it robust to small errors in target position during runtime    
                pan_to_object_deg=pan_to_object_deg,
                table_height=table_height,
                object_metadata=target,
                prim_path=target.get("prim_path"),
            )
            pick_result = self._apply_retry_adjustments_to_pick_result(
                pick_result,
                current_retry_adjustments,
            )
            attempt_log["planning_replanned_after_safe_above"] = compact_pick_result(
                pick_result
            )
            attempt_log["events"].append(
                make_event(
                    "planning_replanned_after_safe_above_done",
                    planning_failed=pick_result.get("planning_failed", False),
                    target_force_n=pick_result.get("target_force_n"),
                    waypoints=list((pick_result.get("joints", {}) or {}).keys()),
                )
            )

            if pick_result.get("planning_failed", False):
                print("[Executor] ❌ Replanning from safe_above was rejected.")
                print("[Executor] IK diagnostics:")
                print(pick_result.get("ik_meta", {}))
                attempt_log["failure_reason"] = "planning_failed_after_safe_above"
                trial_log["attempts"].append(attempt_log)
                trial_log["final_reason"] = "planning_failed_after_safe_above"
                self._last_trial_log = trial_log
                return False

            ## get the computed joints again
            joints = pick_result.get("joints", {})
            # safe_above = joints.get("safe_above")
            pre_grasp = joints.get("pre_grasp")
            grasp = joints.get("grasp")

            ## if pre grasp is None, then recompute pick joints
            if pre_grasp is None:
                print("[Executor] ❌ Missing pre_grasp. Recomputing pick joints...")
                continue

            print("\n[Executor] Moving to pre_grasp...")
            ok = await self.arm.move_to(
                pre_grasp,
                duration=3.0,
                steps=150,
                check_table_collision=True,
            )
            if not ok:
                print("[Executor] ❌ Failed to reach pre_grasp.")
                await self.arm.move_via_safe_height(safe_above, duration=3.0, steps=150)
                attempt_log["failure_reason"] = "failed_to_reach_pre_grasp"
                trial_log["attempts"].append(attempt_log)
                return False

            print("[Executor] ✅ Reached pre_grasp.")
            attempt_log["pre_grasp_ok"] = True

            print("\n[Executor] Moving to grasp pose...")
            ok = await self.arm.move_to(
                grasp,
                duration=3.0,
                steps=150,
                check_table_collision=True,
            )
            if not ok:
                print("[Executor] ❌ Failed to reach grasp pose.")
                await self.arm.move_to(pre_grasp, duration=2.0, steps=100)
                await self.arm.move_via_safe_height(safe_above, duration=3.0, steps=150)
                attempt_log["failure_reason"] = "failed_to_reach_grasp_pose"
                trial_log["attempts"].append(attempt_log)
                return False

            print("[Executor] ✅ Reached grasp pose.")
            attempt_log["grasp_ok"] = True
            attempt_log["snapshots"]["at_grasp_before_preclose"] = pose_snapshot(
                self, target, "at_grasp_before_preclose"
            )
            attempt_log["events"].append(make_event("grasp_pose_reached"))

            # ──── Diagnostic-only snapshot before gripper closure ────
            # We do not reject a grasp yet.  First verify the professor USD's
            # actual flange-to-finger axis convention using measured evidence.
            if self.config.get("enable_preclose_diagnostics", True):
                preclose_object_pos = self._get_observed_object_pos(
                    target, stage_name="preclose"
                )

                if preclose_object_pos is not None:
                    preclose_diag = self.arm.get_preclose_geometry_diagnostics(
                        object_world_pos=preclose_object_pos,
                        object_metadata=target,
                        planned_flange_world_pos=(
                            pick_result.get("flange_targets", {}).get("grasp")
                        ),
                    )
                    attempt_log["preclose_diagnostics"] = json_safe(preclose_diag)

                    print("\n[Executor] Pre-close geometry diagnostics (no rejection):")
                    print(
                        "  planned flange tracking error: "
                        f"{preclose_diag['planned_flange_tracking_error_m']}"
                    )
                    print(
                        "  flange-object delta: "
                        f"{preclose_diag['flange_object_delta_m']}"
                    )
                    print(
                        "  closest local flange-axis candidate: "
                        f"{preclose_diag['closest_axis_candidate']}"
                    )
                    print(
                        "  closest candidate grasp-centre distance: "
                        f"{preclose_diag['closest_axis_distance_m']:.4f} m"
                    )

                    if self.config.get(
                        "preclose_diagnostics_print_all_axes", True
                    ):
                        for axis_name, axis_info in (
                            preclose_diag["candidate_axes"].items()
                        ):
                            print(
                                f"    {axis_name}: "
                                f"centre={axis_info['grasp_centre_world_pos']} "
                                f"distance={axis_info['grasp_centre_object_distance_m']:.4f} m"
                            )

                    # ──── Calibrated pre-close geometry gate ────
                    preclose_gate = self.arm.evaluate_preclose_geometry_gate(
                        preclose_diag
                    )
                    attempt_log["preclose_geometry_gate"] = preclose_gate

                    print("\n[Executor] Calibrated pre-close geometry gate:")
                    print(
                        "  calibrated finger axis:    "
                        f"{preclose_gate['calibrated_finger_axis_local']}"
                    )
                    print(
                        "  geometry_ok:               "
                        f"{preclose_gate['geometry_ok']}"
                    )
                    print(
                        "  grasp-centre XY error:     "
                        f"{preclose_gate['grasp_centre_xy_error_m']:.4f} m"
                    )
                    print(
                        "  vertical finger overlap:   "
                        f"{preclose_gate['vertical_overlap_m']:.4f} m"
                    )
                    print(
                        "  flange tracking error:     "
                        f"{preclose_gate['planned_flange_tracking_error_m']}"
                    )

                    if (
                        self.config.get("enable_preclose_geometry_gate", True)
                        and not preclose_gate["geometry_ok"]
                    ):
                        print(
                            "[Executor] ❌ Rejecting close: pre-close geometry "
                            "is not plausible."
                        )
                        for reason in preclose_gate["reasons"]:
                            print(f"  - {reason}")

                        attempt_log["failure_reason"] = "preclose_geometry_gate_failed"
                        decision = self.retry_policy.decide(trial_log, attempt_log)
                        attempt_log["retry_decision"] = json_safe(decision)
                        trial_log["attempts"].append(attempt_log)

                        if decision["retry"] and attempt < max_attempts - 1:
                            print(f"[Executor] RetryPolicy: {decision['reason']}")
                            current_retry_adjustments = decision.get("adjustments", {})
                            await self._recover_to_safe_for_retry(
                                pre_grasp=pre_grasp,
                                safe_above=safe_above,
                                from_micro_lift=False,
                            )
                            continue

                        trial_log["trial_success"] = False
                        trial_log["final_reason"] = "preclose_geometry_gate_failed"
                        self._last_trial_log = trial_log

                        # The fingers are still open. Retreat gently instead of
                        # issuing a meaningless close command.
                        await self.arm.move_to(
                            pre_grasp,
                            duration=2.0,
                            steps=100,
                            check_table_collision=True,
                        )
                        await self.arm.move_via_safe_height(
                            safe_above,
                            duration=3.0,
                            steps=150,
                        )
                        return False
                else:
                    print(
                        "[Executor] ⚠️ Pre-close diagnostics skipped: "
                        "object pose unavailable"
                    )

            # ──── Now try to close the gripper and validate the grasp. ────
            print("\n[Executor] Closing gripper at grasp pose...")

            # Object pose just before closing; used for close/contact validation, NOT lift validation.
            object_pos_before_close = self._get_observed_object_pos(target, stage_name="before_close")

            # TODO: use the same approach as in compute_pick_joints to get target force
            target_force = pick_result.get("target_force_n", None)

            target_feasibility_for_close = (
                attempt_log.get("target_feasibility")
                or trial_log.get("target_feasibility")
                or {}
            )

            expected_grip_dim_m = target_feasibility_for_close.get(
                "estimated_object_grip_dim_m"
            )

            print(
                "[Executor] Expected grip dimension for close: "
                f"{expected_grip_dim_m}"
            )

            adaptive_effort_target_override = None
            if "adaptive_effort_target_scale" in current_retry_adjustments or "adaptive_effort_target_delta_sim" in current_retry_adjustments:
                base_effort_target = float(
                    self.config.get(
                        "adaptive_effort_target_sim",
                        self.config.get("force_observer_target_effort_sim", 0.35),
                    )
                )
                adaptive_effort_target_override = (
                    base_effort_target * float(current_retry_adjustments.get("adaptive_effort_target_scale", 1.0))
                    + float(current_retry_adjustments.get("adaptive_effort_target_delta_sim", 0.0))
                )
                adaptive_effort_target_override = max(
                    0.05,
                    min(float(self.config.get("adaptive_effort_max_sim", 1.20)), adaptive_effort_target_override),
                )

            close_resolution = await self.close_gripper(
                force_n=target_force,
                expected_grip_dim_m=expected_grip_dim_m,
                hold_settle_extra_s=float(
                    current_retry_adjustments.get("hold_settle_extra_s", 0.0)
                ),
                hold_extra_close_m=pick_result.get("gripper_hold_extra_close_m"),
                target=target,
                adaptive_effort_target_override_sim=adaptive_effort_target_override,
            )
            attempt_log["gripper_close_resolution"] = json_safe(close_resolution)

            adaptive_safety_outcome = self._adaptive_safety_outcome_from_close_resolution(close_resolution)
            attempt_log["adaptive_safety_outcome"] = json_safe(adaptive_safety_outcome)

            # Phase 3.5 supervisory stop: if the reactive safety layer had to
            # relax/open because deformation was unsafe, do not continue to
            # close validation + micro-lift as if the grasp were still secure.
            if adaptive_safety_outcome.get("requires_attempt_stop", False):
                print("[Executor] ⚠️ Adaptive safety requested attempt stop before micro-lift.")
                attempt_log["failure_reason"] = "adaptive_safety_relaxation_triggered"
                attempt_log["adaptive_safety_failure_reason"] = adaptive_safety_outcome.get("reason")

                decision = self.retry_policy.decide(trial_log, attempt_log)
                attempt_log["retry_decision"] = json_safe(decision)
                trial_log["attempts"].append(attempt_log)

                if decision.get("retry") and attempt < max_attempts - 1:
                    print(f"[Executor] RetryPolicy: {decision['reason']}")
                    current_retry_adjustments = decision.get("adjustments", {})
                    await self._recover_to_safe_for_retry(
                        pre_grasp=pre_grasp,
                        safe_above=safe_above,
                        from_micro_lift=False,
                    )
                    continue

                trial_log["trial_success"] = False
                trial_log["final_reason"] = "adaptive_safety_relaxation_triggered"
                self._last_trial_log = trial_log
                return False

            # get diagnostics from the gripper after close
            diag = self.gripper.get_diagnostics()
            print("\n[Executor] Gripper diagnostics after close:")
            # print(diag)

            # Refuse close-validation if close did not resolve
            if not close_resolution.get("resolved", False):
                print("[Executor] Close did not resolve before validation timeout.")

                attempt_log["close_validation"] = {
                    "stage": "after_close",
                    "success": False,
                    "reasons": ["gripper_close_not_resolved_before_validation"],
                    "gripper_state": close_resolution.get("final_state"),
                }

                attempt_log["gripper_diagnostics_after_close"] = (
                    self.gripper.get_diagnostics()
                )

                attempt_log["failure_reason"] = "close_validation_failed"
                attempt_log["close_failure_reasons"] = [
                    "gripper_close_not_resolved_before_validation"
                ]

                decision = self.retry_policy.decide(trial_log, attempt_log)
                attempt_log["retry_decision"] = json_safe(decision)

                trial_log["attempts"].append(attempt_log)

                if decision["retry"] and attempt < max_attempts - 1:
                    print(f"[Executor] RetryPolicy: {decision['reason']}")
                    current_retry_adjustments = decision.get("adjustments", {})

                    await self._recover_to_safe_for_retry(
                        pre_grasp=pre_grasp,
                        safe_above=safe_above,
                        from_micro_lift=False,
                    )
                    continue

                trial_log["trial_success"] = False
                trial_log["final_reason"] = "gripper_close_not_resolved_before_validation"
                self._last_trial_log = trial_log
                return False
            # 

            # ──── Validate grasp/contact after closing ────
            close_validation = self.validate_after_close(
                target=target,
                object_pos_before_close=object_pos_before_close,
            )


            print("\n[Executor] Close validation:")
            print(close_validation)
            attempt_log["close_validation"] = json_safe(close_validation)
            attempt_log["gripper_diagnostics_after_close"] = json_safe(
                self.gripper.get_diagnostics()
            )
            attempt_log["snapshots"]["after_close_validation"] = pose_snapshot(
                self, target, "after_close_validation"
            )
            attempt_log["events"].append(
                make_event(
                    "close_validation_done",
                    success=close_validation.get("success"),
                    reasons=close_validation.get("reasons", []),
                )
            )

            # ──── Handle close validation failure: retry or fail ────
            if not close_validation["success"]:
                print("[Executor] ❌ Close validation failed. Will retry if attempts remain.")
                for reason in close_validation["reasons"]:
                    print(f"  - {reason}")

                attempt_log["failure_reason"] = "close_validation_failed"
                attempt_log["close_failure_reasons"] = json_safe(
                    close_validation.get("reasons", [])
                )
                decision = self.retry_policy.decide(trial_log, attempt_log)
                attempt_log["retry_decision"] = json_safe(decision)
                trial_log["attempts"].append(attempt_log)

                if decision["retry"] and attempt < max_attempts - 1:
                    print(f"[Executor] RetryPolicy: {decision['reason']}")
                    current_retry_adjustments = decision.get("adjustments", {})
                    await self._recover_to_safe_for_retry(
                        pre_grasp=pre_grasp,
                        safe_above=safe_above,
                        from_micro_lift=False,
                    )
                    continue

                trial_log["trial_success"] = False
                trial_log["final_reason"] = "all_attempts_failed_close_validation"
                self._last_trial_log = trial_log
                return False

            else:
                print("[Executor] ✅ Close validation passed.")

                # ──── Micro-lift proof-of-hold checkpoint ────
                # A partially closed gripper can report contact even after the
                # object has slipped.  Before committing to a full lift, move
                # only a few centimetres and verify that the object follows the
                # calibrated grasp centre.
                if self.config.get("enable_micro_lift_test", False):
                    micro_lift = joints.get("micro_lift")
                    if micro_lift is None:
                        print("[Executor] ❌ No micro_lift waypoint available.")
                        attempt_log["failure_reason"] = "missing_micro_lift_waypoint"
                        trial_log["attempts"].append(attempt_log)
                        trial_log["final_reason"] = "missing_micro_lift_waypoint"
                        self._last_trial_log = trial_log
                        return False

                    soft_obs_before_micro = self._observe_target(
                        target, stage_name="before_micro_lift_observation"
                    )
                    object_before_micro = (
                        list(soft_obs_before_micro["center"])
                        if soft_obs_before_micro and soft_obs_before_micro.get("center") is not None
                        else self._get_observed_object_pos(target, stage_name="before_micro_lift")
                    )
                    geometry_before_micro = (
                        self.arm.get_calibrated_capture_geometry_world()
                    )
                    attempt_log["soft_observation_before_micro_lift"] = json_safe(soft_obs_before_micro)

                    print("\n[Executor] Performing micro-lift verification checkpoint...")
                    plan_micro_lift_speed_scale = float(
                        pick_result.get("micro_lift_speed_scale", 1.0) or 1.0
                    )
                    retry_micro_lift_speed_scale = float(
                        current_retry_adjustments.get("micro_lift_speed_scale", 1.0)
                    )
                    # Use the slower/more cautious speed when either the object
                    # strategy or retry policy requests it.
                    micro_lift_speed_scale = min(
                        plan_micro_lift_speed_scale,
                        retry_micro_lift_speed_scale,
                    )
                    base_micro_lift_duration = float(
                        self.config.get("micro_lift_duration", 2.0)
                    )
                    micro_lift_duration = base_micro_lift_duration / max(
                        0.1,
                        micro_lift_speed_scale,
                    )
                    attempt_log["micro_lift_speed_profile"] = json_safe({
                        "plan_micro_lift_speed_scale": plan_micro_lift_speed_scale,
                        "retry_micro_lift_speed_scale": retry_micro_lift_speed_scale,
                        "effective_micro_lift_speed_scale": micro_lift_speed_scale,
                        "duration_s": micro_lift_duration,
                    })

                    soft_motion_trace = []
                    step_callback = (
                        self._make_soft_micro_lift_step_callback(target, soft_motion_trace)
                        if self._is_soft_target(target)
                        else self.gripper.update
                    )

                    ok = await self.arm.move_to(
                        micro_lift,
                        duration=micro_lift_duration,
                        steps=int(self.config.get("micro_lift_steps", 120)),
                        check_table_collision=True,
                        step_callback=step_callback,
                    )
                    attempt_log["soft_motion_trace"] = json_safe(soft_motion_trace)

                    if not ok:
                        print("[Executor] ❌ Micro-lift motion failed.")
                        attempt_log["failure_reason"] = "micro_lift_motion_failed"
                        trial_log["attempts"].append(attempt_log)
                        trial_log["final_reason"] = "micro_lift_motion_failed"
                        self._last_trial_log = trial_log
                        return False

                    await self._step_gripper_for_seconds(
                        float(self.config.get("post_micro_lift_settle_seconds", 0.4))
                    )

                    soft_obs_after_micro = self._observe_target(
                        target, stage_name="after_micro_lift_observation"
                    )
                    object_after_micro = (
                        list(soft_obs_after_micro["center"])
                        if soft_obs_after_micro and soft_obs_after_micro.get("center") is not None
                        else self._get_observed_object_pos(target, stage_name="after_micro_lift")
                    )
                    geometry_after_micro = (
                        self.arm.get_calibrated_capture_geometry_world()
                    )
                    attempt_log["soft_observation_after_micro_lift"] = json_safe(soft_obs_after_micro)

                    if object_before_micro is not None and object_after_micro is not None:
                        dz_obj = float(object_after_micro[2]) - float(object_before_micro[2])
                        attempt_log["soft_micro_lift_motion"] = json_safe({
                            "object_center_before": object_before_micro,
                            "object_center_after": object_after_micro,
                            "object_center_dz_m": dz_obj,
                            "bottom_z_before": (soft_obs_before_micro or {}).get("bottom_z"),
                            "bottom_z_after": (soft_obs_after_micro or {}).get("bottom_z"),
                            "bottom_z_delta_m": (
                                float((soft_obs_after_micro or {}).get("bottom_z"))
                                - float((soft_obs_before_micro or {}).get("bottom_z"))
                                if soft_obs_before_micro and soft_obs_after_micro
                                and soft_obs_before_micro.get("bottom_z") is not None
                                and soft_obs_after_micro.get("bottom_z") is not None
                                else None
                            ),
                            "height_before_m": (soft_obs_before_micro or {}).get("height_m"),
                            "height_after_m": (soft_obs_after_micro or {}).get("height_m"),
                            "deformation_ratio_z_before": (soft_obs_before_micro or {}).get("deformation_ratio_z"),
                            "deformation_ratio_z_after": (soft_obs_after_micro or {}).get("deformation_ratio_z"),
                            "table_gap_before_m": (soft_obs_before_micro or {}).get("table_gap_m"),
                            "table_gap_after_m": (soft_obs_after_micro or {}).get("table_gap_m"),
                        })

                    soft_motion_observer_reliability = self._assess_soft_motion_observer_reliability(
                        target=target,
                        soft_obs_before=soft_obs_before_micro,
                        soft_obs_after=soft_obs_after_micro,
                        geometry_before=geometry_before_micro,
                        geometry_after=geometry_after_micro,
                        gripper_has_object=self.gripper.has_object(),
                        trace=attempt_log.get("soft_motion_trace", []),
                    )
                    attempt_log["soft_motion_observer_reliability"] = json_safe(soft_motion_observer_reliability)

                    micro_validation = self.micro_lift_validator.evaluate(
                        object_pos_before=object_before_micro,
                        object_pos_after=object_after_micro,
                        flange_pos_before=geometry_before_micro["flange_world_pos"],
                        flange_pos_after=geometry_after_micro["flange_world_pos"],
                        grasp_centre_before=geometry_before_micro["grasp_centre_world_pos"],
                        grasp_centre_after=geometry_after_micro["grasp_centre_world_pos"],
                        gripper_has_object=self.gripper.has_object(),
                        observer_reliability=soft_motion_observer_reliability,
                    )

                    print("\n[Executor] Micro-lift validation:")
                    print(micro_validation)
                    attempt_log["micro_lift_validation"] = json_safe(micro_validation)
                    attempt_log["gripper_diagnostics_after_micro_lift"] = json_safe(
                        self.gripper.get_diagnostics()
                    )
                    attempt_log["snapshots"]["after_micro_lift_validation"] = pose_snapshot(
                        self, target, "after_micro_lift_validation"
                    )
                    attempt_log["events"].append(
                        make_event(
                            "micro_lift_validation_done",
                            success=micro_validation.get("success"),
                            reasons=micro_validation.get("reasons", []),
                        )
                    )

                    if not micro_validation["success"]:
                        print("[Executor] ❌ Micro-lift validation failed.")
                        for reason in micro_validation["reasons"]:
                            print(f"  - {reason}")
                        attempt_log["failure_reason"] = "micro_lift_validation_failed"

                        decision = self.retry_policy.decide(trial_log, attempt_log)
                        attempt_log["retry_decision"] = json_safe(decision)
                        trial_log["attempts"].append(attempt_log)

                        if decision["retry"] and attempt < max_attempts - 1:
                            print(f"[Executor] RetryPolicy: {decision['reason']}")
                            current_retry_adjustments = decision.get("adjustments", {})
                            await self._recover_to_safe_for_retry(
                                pre_grasp=pre_grasp,
                                safe_above=safe_above,
                                from_micro_lift=True,
                            )
                            continue

                        trial_log["final_reason"] = "micro_lift_validation_failed"
                        self._last_trial_log = trial_log

                        # Safe reactive recovery: keep hold pressure while
                        # retreating upward, then release only at safe height.
                        await self.arm.move_via_safe_height(
                            safe_above,
                            duration=float(self.config.get("retry_to_safe_duration", 4.0)),
                            steps=int(self.config.get("retry_to_safe_steps", 240)),
                            step_callback=self.gripper.update,
                        )
                        self.gripper.open()
                        await self._step_gripper_for_seconds(
                            float(self.config.get("post_open_settle_seconds", 0.8))
                        )
                        return False

                    print("[Executor] ✅ Micro-lift passed: object followed gripper.")

                    if not self.config.get("enable_lift_test", False):
                        print("[Executor] Full lift disabled. Ending after validated micro-lift.")
                        attempt_log["success"] = True
                        trial_log["trial_success"] = True
                        trial_log["final_reason"] = (
                            "micro_lift_validation_passed_full_lift_disabled"
                        )
                        trial_log["attempts"].append(attempt_log)
                        self._last_trial_log = trial_log
                        return True

                elif not self.config.get("enable_lift_test", False):
                    print("[Executor] Lift disabled for now. Ending after successful close validation.")
                    attempt_log["success"] = True
                    trial_log["trial_success"] = True
                    trial_log["final_reason"] = "close_validation_passed_lift_disabled"
                    trial_log["attempts"].append(attempt_log)
                    self._last_trial_log = trial_log
                    return True

                # ──── Perform the full lift ────
                lift = joints.get("lift")
                if lift is None:
                    print("[Executor] ❌ No lift waypoint available.")
                    return False

                # Object pose immediately before lift.
                # This is the reference for lift delta.
                object_pos_before_lift = self._get_observed_object_pos(target, stage_name="before_full_lift")

                print("\n[Executor] Lifting object...")

                ok = await self.arm.move_to(
                    lift,
                    duration=float(self.config.get("lift_duration", 5.0)),
                    steps=int(self.config.get("lift_steps", 300)),
                    check_table_collision=True,
                    step_callback=self.gripper.update,
                )

                if not ok:
                    print("[Executor] ❌ Lift motion failed.")
                    return False

                # Let the gripper/object settle briefly after lift.
                await self._step_gripper_for_seconds(0.5)

                print("\n[Executor] Gripper diagnostics after lift:")
                # print(self.gripper.get_diagnostics())

                # ──── Validate actual object lift ────
                lift_validation = self.validate_after_lift(
                    target=target,
                    object_pos_before_lift=object_pos_before_lift,
                    table_height=table_height,
                )

                print("\n[Executor] Lift validation:")
                print(lift_validation)

                attempt_log["lift_validation"] = lift_validation
                attempt_log["gripper_diagnostics_after_lift"] = self.gripper.get_diagnostics()

                if lift_validation["success"]:
                    print("[Executor] ✅ Validated lift: object moved with gripper.")

                    # ──── Phase 4.1: transport held object above the semantic place zone ────
                    if bool(self.config.get("enable_place_transport_test", False)):
                        print("\n[Executor] Phase 4.1: transporting held object to place zone...")
                        place_goal = self._compute_place_object_center(
                            scene_info=scene_info,
                            target=target,
                            table_height=table_height,
                        )
                        place_plan_log = {
                            "phase": "4.1_transport_to_place_zone_no_release",
                            "place_goal": place_goal,
                            "planning_failed": False,
                            "reason": None,
                            "waypoint_names": [],
                            "joints": None,
                        }
                        attempt_log["place_transport_plan"] = json_safe(place_plan_log)

                        if not place_goal.get("available"):
                            print("[Executor] ❌ Place zone unavailable for transport.")
                            attempt_log["failure_reason"] = "place_zone_unavailable"
                            place_plan_log["planning_failed"] = True
                            place_plan_log["reason"] = place_goal.get("reason")
                            attempt_log["place_transport_plan"] = json_safe(place_plan_log)
                            trial_log["attempts"].append(attempt_log)
                            trial_log["final_reason"] = "place_zone_unavailable"
                            self._last_trial_log = trial_log
                            return False

                        place_joints = self.arm.compute_place_joints(
                            place_world_pos=place_goal["place_object_center"],
                            pan_to_place_deg=pan_to_object_deg,
                            table_height=table_height,
                            object_metadata=target,
                        ) or {}
                        place_plan_log["waypoint_names"] = list(place_joints.keys())
                        place_plan_log["joints"] = place_joints
                        attempt_log["place_transport_plan"] = json_safe(place_plan_log)

                        # Phase 4.1b: avoid unnecessary safe_above lift during
                        # transport.  The previous target used safe_above, which lifted
                        # the object roughly 10 cm more while moving laterally; that can
                        # stretch/slide a deformable object.  Default to the place "lift"
                        # waypoint, closer to the current carrying height, and move more
                        # slowly.
                        transport_waypoint_name = str(
                            self.config.get("place_transport_waypoint", "lift")
                        )
                        place_transport_target = place_joints.get(transport_waypoint_name)
                        if place_transport_target is None:
                            place_transport_target = place_joints.get("safe_above")
                            transport_waypoint_name = "safe_above_fallback"

                        place_plan_log["transport_waypoint_name"] = transport_waypoint_name
                        place_plan_log["use_safe_height_wrapper"] = bool(
                            self.config.get("place_transport_use_safe_height_wrapper", False)
                        )
                        attempt_log["place_transport_plan"] = json_safe(place_plan_log)

                        if place_transport_target is None:
                            print("[Executor] ❌ Place transport planning failed: missing transport waypoint.")
                            attempt_log["failure_reason"] = "place_transport_planning_failed"
                            place_plan_log["planning_failed"] = True
                            place_plan_log["reason"] = "missing_place_transport_waypoint"
                            attempt_log["place_transport_plan"] = json_safe(place_plan_log)
                            trial_log["attempts"].append(attempt_log)
                            trial_log["final_reason"] = "place_transport_planning_failed"
                            self._last_trial_log = trial_log
                            return False

                        object_pos_before_transport = self._get_observed_object_pos(
                            target, stage_name="before_place_transport"
                        )

                        transport_duration = float(self.config.get("place_transport_duration", 6.0))
                        transport_steps = int(self.config.get("place_transport_steps", 360))

                        # Phase 4.1g: compute a shear-aware effort reference before moving.
                        # The Phase 4.1f audit showed that gravity/shear is the main
                        # load and that reacting after slip starts is too late.  Here
                        # we estimate required normal load from 2*mu*N >= F_shear,
                        # convert it to our empirical effort proxy, preload while
                        # stationary, then keep a small feedback loop alive during
                        # transport.
                        transport_shear_reference = None
                        transport_preload = None
                        if bool(self.config.get("place_transport_shear_compensation_enabled", True)):
                            transport_shear_reference = self._compute_transport_shear_reference(
                                target=target,
                                place_goal=place_goal,
                                transport_duration=transport_duration,
                                transport_steps=transport_steps,
                            )
                            attempt_log["place_transport_shear_reference"] = json_safe(transport_shear_reference)
                            if bool(self.config.get("place_transport_shear_preload_enabled", True)):
                                transport_preload = await self._apply_transport_shear_preload(
                                    target=target,
                                    shear_reference=transport_shear_reference,
                                )
                                attempt_log["place_transport_shear_preload"] = json_safe(transport_preload)

                        transport_target_effort_override = None
                        if isinstance(transport_shear_reference, dict) and transport_shear_reference.get("success"):
                            transport_target_effort_override = transport_shear_reference.get("target_effort_sim")

                        transport_step_callback = self.gripper.update
                        transport_admittance = None
                        if bool(self.config.get("place_transport_admittance_enabled", False)):
                            transport_step_callback, transport_admittance = (
                                self._make_transport_admittance_step_callback(
                                    target=target,
                                    target_effort_override_sim=transport_target_effort_override,
                                    shear_reference=transport_shear_reference,
                                )
                            )
                            attempt_log["place_transport_admittance"] = json_safe(transport_admittance)

                        transport_shear_audit = None
                        if bool(self.config.get("place_transport_shear_audit_enabled", True)):
                            transport_step_callback, transport_shear_audit = (
                                self._make_transport_shear_audit_step_callback(
                                    target=target,
                                    base_step_callback=transport_step_callback,
                                    dt_s=(transport_duration / max(1, transport_steps)),
                                    place_goal=place_goal,
                                )
                            )
                            attempt_log["place_transport_shear_audit"] = json_safe(transport_shear_audit)

                        if bool(self.config.get("place_transport_use_safe_height_wrapper", False)):
                            ok = await self.arm.move_via_safe_height(
                                place_transport_target,
                                duration=transport_duration,
                                steps=transport_steps,
                                step_callback=transport_step_callback,
                            )
                        else:
                            ok = await self.arm.move_to(
                                place_transport_target,
                                duration=transport_duration,
                                steps=transport_steps,
                                check_table_collision=True,
                                step_callback=transport_step_callback,
                            )

                        if transport_admittance is not None:
                            transport_admittance["move_completed"] = bool(ok)
                            transport_admittance["duration_s"] = transport_duration
                            transport_admittance["steps"] = transport_steps
                            attempt_log["place_transport_admittance"] = json_safe(transport_admittance)
                        if transport_shear_audit is not None:
                            transport_shear_audit["move_completed"] = bool(ok)
                            transport_shear_audit["duration_s"] = transport_duration
                            transport_shear_audit["steps"] = transport_steps
                            attempt_log["place_transport_shear_audit"] = json_safe(transport_shear_audit)

                        if not ok:
                            transport_abort_reason = None
                            if isinstance(transport_admittance, dict):
                                transport_abort_reason = transport_admittance.get("abort_reason")
                            if transport_abort_reason:
                                failure_reason = "place_transport_aborted_due_to_slip_or_drop"
                                print(f"[Executor] ❌ Place transport aborted by reactive monitor: {transport_abort_reason}")
                            else:
                                failure_reason = "place_transport_motion_failed"
                                print("[Executor] ❌ Place transport motion failed.")
                            attempt_log["failure_reason"] = failure_reason
                            attempt_log["place_transport_abort_reason"] = transport_abort_reason
                            trial_log["attempts"].append(attempt_log)
                            trial_log["final_reason"] = failure_reason
                            self._last_trial_log = trial_log
                            return False

                        await self._step_gripper_for_seconds(
                            float(self.config.get("post_place_transport_settle_seconds", 0.5))
                        )
                        if transport_admittance is not None:
                            try:
                                transport_admittance["post_transport_force_observation"] = json_safe(
                                    self.force_observer.observe(stage_name="after_place_transport_settle")
                                    if hasattr(self, "force_observer") else None
                                )
                            except Exception as e:
                                transport_admittance["post_transport_force_observation"] = {
                                    "available": False,
                                    "reason": f"force_observer_exception: {e}",
                                }
                            try:
                                transport_admittance["post_transport_soft_observation"] = json_safe(
                                    self._observe_target(target, stage_name="after_place_transport_settle")
                                )
                            except Exception as e:
                                transport_admittance["post_transport_soft_observation"] = {
                                    "available": False,
                                    "reason": f"soft_observer_exception: {e}",
                                }
                            attempt_log["place_transport_admittance"] = json_safe(transport_admittance)

                        if transport_shear_audit is not None:
                            try:
                                transport_shear_audit["post_transport_force_observation"] = json_safe(
                                    self.force_observer.observe(stage_name="after_place_transport_shear_audit_settle")
                                    if hasattr(self, "force_observer") else None
                                )
                            except Exception as e:
                                transport_shear_audit["post_transport_force_observation"] = {
                                    "available": False,
                                    "reason": f"force_observer_exception: {e}",
                                }
                            try:
                                transport_shear_audit["post_transport_soft_observation"] = json_safe(
                                    self._observe_target(target, stage_name="after_place_transport_shear_audit_settle")
                                )
                            except Exception as e:
                                transport_shear_audit["post_transport_soft_observation"] = {
                                    "available": False,
                                    "reason": f"soft_observer_exception: {e}",
                                }
                            attempt_log["place_transport_shear_audit"] = json_safe(transport_shear_audit)

                        object_pos_after_transport = self._get_observed_object_pos(
                            target, stage_name="after_place_transport"
                        )
                        place_transport_validation = self.validate_after_place_transport(
                            target=target,
                            place_zone=place_goal.get("place_zone", {}),
                            object_pos_before_transport=object_pos_before_transport,
                            object_pos_after_transport=object_pos_after_transport,
                            table_height=table_height,
                        )
                        print("\n[Executor] Place transport validation:")
                        print(place_transport_validation)
                        attempt_log["place_transport_validation"] = json_safe(place_transport_validation)
                        attempt_log["events"].append(
                            make_event(
                                "place_transport_validation_done",
                                success=place_transport_validation.get("success"),
                                reasons=place_transport_validation.get("reasons", []),
                            )
                        )

                        if not place_transport_validation.get("success"):
                            print("[Executor] ❌ Place transport validation failed.")
                            attempt_log["failure_reason"] = "place_transport_validation_failed"
                            trial_log["attempts"].append(attempt_log)
                            trial_log["final_reason"] = "place_transport_validation_failed"
                            self._last_trial_log = trial_log
                            return False

                        print("[Executor] ✅ Phase 4.1 transport passed: object held above place zone.")
                        attempt_log["success"] = True
                        trial_log["trial_success"] = True
                        trial_log["final_reason"] = "place_transport_validation_passed_release_disabled"
                        trial_log["attempts"].append(attempt_log)
                        self._last_trial_log = trial_log
                        return True

                    attempt_log["success"] = True
                    trial_log["trial_success"] = True
                    trial_log["final_reason"] = "lift_validation_passed"
                    trial_log["attempts"].append(attempt_log)

                    self._last_trial_log = trial_log
                    return True

                print("[Executor] ❌ Lift validation failed.")
                for reason in lift_validation["reasons"]:
                    print(f"  - {reason}")
                    attempt_log["failure_reason"] = "lift_validation_failed"
                    trial_log["attempts"].append(attempt_log)
                    self._last_trial_log = trial_log
                    return False
 
                ###

            # ──── Fallback: no object detected after close. ────
            print("[Executor] ❌ No object detected after close.")
            attempt_log["failure_reason"] = "close_validation_failed"
            decision = self.retry_policy.decide(trial_log, attempt_log)
            attempt_log["retry_decision"] = json_safe(decision)
            trial_log["attempts"].append(attempt_log)

            if decision["retry"] and attempt < max_attempts - 1:
                print(f"[Executor] RetryPolicy: {decision['reason']}")
                current_retry_adjustments = decision.get("adjustments", {})
                await self._recover_to_safe_for_retry(
                    pre_grasp=pre_grasp,
                    safe_above=safe_above,
                    from_micro_lift=False,
                )
                continue

            print("[Executor] ❌ All grasp attempts failed.")
            trial_log["trial_success"] = False
            trial_log["final_reason"] = "all_attempts_failed_no_object_after_close"
            self._last_trial_log = trial_log
            await self._hold_for_inspection()
            return False