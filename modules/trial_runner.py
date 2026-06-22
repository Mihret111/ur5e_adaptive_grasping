from modules.scene_builder import SceneBuilder
from modules.pick_and_place_executor import PickAndPlaceExecutor
from modules.manipulation_primitive_executor import ManipulationPrimitiveExecutor
import os
import json
from modules.trial_diagnostics import json_safe
from datetime import datetime
import copy


class TrialRunner:
    def __init__(self, config, table_materials, table_slots, step_fn, step_seconds_fn):
        self.config = config
        self.table_materials = table_materials
        self.table_slots = table_slots

        self.step_fn = step_fn
        self.step_seconds_fn = step_seconds_fn

        self._total_attempts = 0
        self._total_successes = 0

        print("[TRACE] TrialRunner: before PickAndPlaceExecutor")
        self.pick_executor = PickAndPlaceExecutor(self.config)
        print("[TRACE] TrialRunner: after PickAndPlaceExecutor")

        self.primitive_executor = ManipulationPrimitiveExecutor(
            config=self.config,
            pick_executor=self.pick_executor,
        )

        print("[TRACE] TrialRunner: before SceneBuilder")
        self.scene_builder = SceneBuilder(
            config=self.config,
            table_materials=self.table_materials,
            table_seat_slots=self.table_slots,
        )
        print("[TRACE] TrialRunner: after SceneBuilder")
        print("[TrialRunner] Ready.")

    def _run_dir(self):
        return self.config.get("paths", {}).get("run_outputs_dir", "runs/isaac_run")

    def _target_slug(self, target):
        label = str((target or {}).get("label", "target"))
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in label)
        return safe[:64] or "target"

    def _ordered_phase7_targets(self, objects):
        order = self.config.get("phase7_object_run_order", []) or []
        if order:
            rank = {str(label): i for i, label in enumerate(order)}
            return sorted(
                list(objects),
                key=lambda o: (rank.get(str(o.get("label")), 999), int(o.get("batch_order", 999)), str(o.get("label", ""))),
            )
        return sorted(list(objects), key=lambda o: (int(o.get("batch_order", 999)), str(o.get("label", ""))))

    def _scene_for_target(self, scene_info, target, object_index):
        scene_copy = dict(scene_info)
        target_copy = dict(target)
        place_zone_key = target_copy.get("place_zone_key") or f"place_zone_{object_index}"
        target_copy["place_zone_key"] = place_zone_key
        scene_copy["pick_target"] = target_copy
        scene_copy["active_place_zone_key"] = place_zone_key
        scene_copy["phase7_object_index"] = object_index
        scene_copy["phase7_object_label"] = target_copy.get("label")
        return scene_copy

    def _save_trial_log(self, *, trial_log, trial_index, ok, object_index=None, target=None):
        if trial_log is None:
            return None
        trial_log["trial_index"] = trial_index
        trial_log["ok_returned"] = ok
        if object_index is not None:
            trial_log["phase7_object_index"] = object_index
            trial_log["phase7_object_label"] = (target or {}).get("label")
            trial_log["phase7_place_zone_key"] = (target or {}).get("place_zone_key")

        run_dir = self._run_dir()
        trial_dir = os.path.join(run_dir, f"trial_{trial_index}")
        os.makedirs(trial_dir, exist_ok=True)

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if object_index is None:
            log_name = f"validation_log_{run_stamp}.json"
        else:
            log_name = f"object_{object_index:02d}_{self._target_slug(target)}_validation_log_{run_stamp}.json"
        log_path = os.path.join(trial_dir, log_name)

        with open(log_path, "w") as f:
            json.dump(json_safe(trial_log), f, indent=2)
        print(f"[TrialRunner] Validation log saved to: {log_path}")
        return log_path

    def _save_primitive_log(self, *, primitive_log, trial_index):
        if primitive_log is None:
            return None
        run_dir = self._run_dir()
        trial_dir = os.path.join(run_dir, f"trial_{trial_index}")
        os.makedirs(trial_dir, exist_ok=True)

        path = os.path.join(
            trial_dir,
            f"phase8_manipulation_primitives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(path, "w") as f:
            json.dump(json_safe(primitive_log), f, indent=2)
        print(f"[TrialRunner] Phase 8 primitive log saved to: {path}")
        return path

    async def _maybe_run_phase8_primitives(self, trial_index, scene_info, *, when: str):
        if not bool(self.config.get("phase8_manipulation_primitives_enabled", False)):
            return None

        configured_when = str(self.config.get("phase8_run_timing", "standalone")).lower()
        allowed = {"standalone", "after_phase7", "before_phase7", "after_pick_place", "after_batch"}
        if configured_when not in allowed:
            print(f"[TrialRunner] ⚠ Unknown phase8_run_timing={configured_when}; skipping primitives")
            return None

        # Protect the frozen pick-place benchmark by default.  Phase-8 primitives
        # can be run standalone on a fresh scene, or after the Phase-7 batch when
        # explicitly requested.
        should_run = (
            (configured_when == "standalone" and when == "standalone")
            or (configured_when in ("after_phase7", "after_pick_place", "after_batch") and when == "after_phase7")
            or (configured_when == "before_phase7" and when == "before_phase7")
        )
        if not should_run:
            return None

        primitive_log = await self.primitive_executor.run_from_scene(scene_info)
        self._save_primitive_log(primitive_log=primitive_log, trial_index=trial_index)

        # Phase 8 is a manipulation primitive benchmark, not a Phase-7 pick-place
        # attempt.  Still, it should contribute to the final run counters so the
        # terminal does not misleadingly print Attempts: 0 Successes: 0.
        primitive_results_for_total = (primitive_log or {}).get("results", []) or []
        primitive_attempts_for_total = len(primitive_results_for_total)
        primitive_successes_for_total = int((primitive_log or {}).get("success_count", 0) or 0)
        self._total_attempts += primitive_attempts_for_total
        self._total_successes += primitive_successes_for_total
        self._print_phase8_summary(primitive_log)

        return primitive_log

    def _print_phase8_summary(self, primitive_log):
        """Print a compact terminal summary for Phase-8 primitives.

        This is separate from the Phase-7 pick-place validation summary, but it
        updates the same TrialRunner counters through _maybe_run_phase8_primitives
        so main.py reports non-zero Attempts/Successes in standalone primitive mode.
        """
        if not primitive_log:
            return

        results = primitive_log.get("results", []) or []
        success_count = int(primitive_log.get("success_count", 0) or 0)
        failure_count = int(primitive_log.get("failure_count", 0) or 0)

        print("\n[TrialRunner] Phase 8 primitive summary:")
        print(f"  primitive_attempts: {len(results)}")
        print(f"  primitive_successes: {success_count}")
        print(f"  primitive_failures: {failure_count}")

        for idx, r in enumerate(results, start=1):
            primitive = r.get("primitive", "unknown")
            label = r.get("target_label", "unknown_target")
            ok = bool(r.get("success", False))
            reason = r.get("reason")
            metrics = r.get("metrics") or {}
            along = metrics.get("along_command_m")
            along_txt = "n/a" if along is None else f"{float(along):.4f} m"
            mark = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {idx}. {mark}  {primitive} → {label}  along={along_txt}  reason={reason}")

    def _print_validation_summary(self, trial_log):
        if not trial_log:
            return
        print("\n[TrialRunner] Validation summary:")
        print(f"  target: {trial_log['target']['label']}")
        print(f"  shape: {trial_log['target']['shape']}")
        print(f"  material: {trial_log['target']['material']}")
        print(f"  success: {trial_log['trial_success']}")
        print(f"  final_reason: {trial_log['final_reason']}")

        for a in trial_log.get("attempts", []):
            print(f"  attempt {a['attempt']}:")
            print(f"    safe_above_ok: {a['safe_above_ok']}")
            print(f"    pre_grasp_ok: {a['pre_grasp_ok']}")
            print(f"    grasp_ok: {a['grasp_ok']}")
            print(f"    failure_reason: {a['failure_reason']}")

            if a.get("preclose_diagnostics"):
                d = a["preclose_diagnostics"]
                print("    preclose_closest_axis: " f"{d['closest_axis_candidate']}")
                print("    preclose_axis_distance_m: " f"{d['closest_axis_distance_m']:.4f}")
                print("    preclose_tracking_error_m: " f"{d['planned_flange_tracking_error_m']}")

            if a.get("preclose_geometry_gate"):
                g = a["preclose_geometry_gate"]
                print("    preclose_geometry_ok: " f"{g['geometry_ok']}")
                print("    preclose_xy_error_m: " f"{g['grasp_centre_xy_error_m']:.4f}")
                print("    preclose_vertical_overlap_m: " f"{g['vertical_overlap_m']:.4f}")
                print("    preclose_gate_reasons: " f"{g['reasons']}")

            if a.get("close_validation"):
                print(f"    close_success: {a['close_validation']['success']}")
                print(f"    close_reasons: {a['close_validation']['reasons']}")

            if a.get("micro_lift_validation"):
                m = a["micro_lift_validation"]
                print(f"    micro_lift_success: {m['success']}")
                print(f"    micro_lift_object_dz_m: {m['object_lift_delta_z_m']}")
                print(f"    micro_lift_following_ratio: {m['following_ratio']}")
                print(f"    micro_lift_relative_drift_m: {m['relative_grasp_drift_m']}")
                print(f"    micro_lift_reasons: {m['reasons']}")

            if a.get("lift_validation"):
                print(f"    lift_success: {a['lift_validation']['success']}")
                print(f"    lift_reasons: {a['lift_validation']['reasons']}")

            if a.get("place_release_result"):
                v = (a["place_release_result"].get("validation") or {})
                print(f"    release_success: {v.get('success')}")
                print(f"    release_reasons: {v.get('reasons')}")

    async def run_all(self):
        num_trials = int(self.config.get("num_trials", 1))

        for i in range(num_trials):
            print(f"\n[TrialRunner] Trial {i + 1}/{num_trials}")

            if self.config.get("reset_robot_before_trial", True):
                await self.pick_executor.reset_robot_for_trial()
            else:
                print("\n[TrialRunner] Skipping home reset; starting from current pose.")
                await self.pick_executor.open_gripper()

            scene_info = self.scene_builder.build_trial(i)
            if not scene_info["all_objects"]:
                print("❌ No objects spawned in this trial. Moving to next trial.")
                continue

            settle_seconds = float(self.config.get("post_spawn_settle_seconds", 1.0))
            print(f"[TrialRunner] Settling spawned objects for {settle_seconds:.2f}s...")
            await self.step_seconds_fn(settle_seconds)

            phase7_enabled = bool(self.config.get("phase7_multi_object_batch_enabled", False))
            phase8_enabled = bool(self.config.get("phase8_manipulation_primitives_enabled", False))
            phase8_timing = str(self.config.get("phase8_run_timing", "standalone")).lower()

            if phase7_enabled:
                await self._maybe_run_phase8_primitives(i, scene_info, when="before_phase7")
                await self._run_phase7_multi_object_batch(i, scene_info)
                await self._maybe_run_phase8_primitives(i, scene_info, when="after_phase7")
            elif phase8_enabled and phase8_timing == "standalone":
                await self._maybe_run_phase8_primitives(i, scene_info, when="standalone")
            else:
                self._total_attempts += 1
                ok = await self.pick_executor.run_generic_pick(scene_info)
                if ok:
                    self._total_successes += 1
                    print(f"[TrialRunner] Trial {i + 1} pick success")
                else:
                    print(f"[TrialRunner] Trial {i + 1} pick failed")

                trial_log = self.pick_executor.get_last_trial_log()
                self._save_trial_log(trial_log=trial_log, trial_index=i, ok=ok)
                self._print_validation_summary(trial_log)

            await self.pick_executor.park_robot_after_trial()

    async def _run_phase7_multi_object_batch(self, trial_index, scene_info):
        objects = self._ordered_phase7_targets(scene_info.get("all_objects", []))
        max_objects = self.config.get("phase7_max_objects_per_batch", None)
        if max_objects is not None:
            objects = objects[: int(max_objects)]

        batch_log = {
            "schema_version": "phase7_multi_object_batch_v1",
            "trial_index": trial_index,
            "objects_total": len(objects),
            "objects": [],
            "success_count": 0,
            "failure_count": 0,
        }

        print("\n[TrialRunner] Phase 7 multi-object batch enabled")
        print(f"  objects: {[o.get('label') for o in objects]}")

        for object_index, target in enumerate(objects):
            print("\n" + "─" * 60)
            print(f"[TrialRunner] Phase 7 object {object_index + 1}/{len(objects)}: {target.get('label')}")
            print("─" * 60)
            self._total_attempts += 1

            if object_index > 0 and bool(self.config.get("phase7_reset_robot_between_objects", True)):
                await self.pick_executor.reset_robot_for_trial()
                await self.step_seconds_fn(float(self.config.get("phase7_between_object_settle_seconds", 0.50)))

            scene_for_target = self._scene_for_target(scene_info, target, object_index)
            ok = await self.pick_executor.run_generic_pick(scene_for_target)
            if ok:
                self._total_successes += 1
                batch_log["success_count"] += 1
            else:
                batch_log["failure_count"] += 1

            trial_log = self.pick_executor.get_last_trial_log()
            self._save_trial_log(
                trial_log=trial_log,
                trial_index=trial_index,
                ok=ok,
                object_index=object_index,
                target=scene_for_target["pick_target"],
            )
            self._print_validation_summary(trial_log)
            batch_log["objects"].append(json_safe({
                "object_index": object_index,
                "label": target.get("label"),
                "shape": target.get("shape"),
                "place_zone_key": scene_for_target["pick_target"].get("place_zone_key"),
                "success": bool(ok),
                "final_reason": trial_log.get("final_reason") if trial_log else None,
            }))

            if (not ok) and bool(self.config.get("phase7_stop_batch_on_first_failure", False)):
                print("[TrialRunner] Stopping Phase 7 batch after first failure.")
                break

        run_dir = self._run_dir()
        trial_dir = os.path.join(run_dir, f"trial_{trial_index}")
        os.makedirs(trial_dir, exist_ok=True)
        batch_path = os.path.join(
            trial_dir,
            f"phase7_batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(batch_path, "w") as f:
            json.dump(json_safe(batch_log), f, indent=2)
        print(f"[TrialRunner] Phase 7 batch summary saved to: {batch_path}")