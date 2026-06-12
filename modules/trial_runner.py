from modules.scene_builder import SceneBuilder
from modules.pick_and_place_executor import PickAndPlaceExecutor
import os
import json

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

        print("[TRACE] TrialRunner: before SceneBuilder")
        self.scene_builder = SceneBuilder(
            config=self.config,
            table_materials=self.table_materials,
            table_seat_slots=self.table_slots,
        )
        print("[TRACE] TrialRunner: after SceneBuilder")
        print("[TrialRunner] Ready.")

    async def run_all(self):
        num_trials = int(self.config.get("num_trials", 1))

        for i in range(num_trials):
            print(f"\n[TrialRunner] Trial {i + 1}/{num_trials}")
            self._total_attempts += 1

            # 1. Reset robot before spawning a new trial
            # This prevents the arm/gripper from colliding with newly spawned objects
            # and removes stale drive states from previous runs.
            if self.config.get("reset_robot_before_trial", True):
                # resetting the arm before spawning new objects to reduce accidental contacts with newly spawned objects
                await self.pick_executor.reset_robot_for_trial()
            else:
                print("\n[TrialRunner] Skipping home reset; starting from current pose.")
                await self.pick_executor.open_gripper()

            # 2. Build randomized trial scene
            scene_info = self.scene_builder.build_trial(i)

            # Check if any objects were spawned successfully
            if not scene_info["all_objects"]:
                print("❌ No objects spawned in this trial. Moving to next trial.")
                continue

            # 3. Let spawned objects settle before reading actual prim poses
            settle_seconds = float(self.config.get("post_spawn_settle_seconds", 1.0))
            print(f"[TrialRunner] Settling spawned objects for {settle_seconds:.2f}s...")
            await self.step_seconds_fn(settle_seconds)

            # 4. Run pick and place for the current trial
            ok = await self.pick_executor.run_generic_pick(scene_info)

            ## Update trial statistics and print
            if ok:
                self._total_successes += 1
                print(f"[TrialRunner] Trial {i + 1} pick success")
            else:
                print(f"[TrialRunner] Trial {i + 1} pick failed")

            # 5. Save validation log if available.
            trial_log = self.pick_executor.get_last_trial_log()
            if trial_log is not None:
                trial_log["trial_index"] = i
                trial_log["ok_returned"] = ok

                run_dir = self.config.get("paths", {}).get(
                    "run_outputs_dir",
                    "runs/isaac_run",
                )

                trial_dir = os.path.join(run_dir, f"trial_{i}")
                os.makedirs(trial_dir, exist_ok=True)

                log_path = os.path.join(trial_dir, "validation_log.json")

                with open(log_path, "w") as f:
                    json.dump(trial_log, f, indent=2)

                print(f"[TrialRunner] Validation log saved to: {log_path}")

                print("\n[TrialRunner] Validation summary:")
                print(f"  target: {trial_log['target']['label']}")
                print(f"  shape: {trial_log['target']['shape']}")
                print(f"  material: {trial_log['target']['material']}")
                print(f"  success: {trial_log['trial_success']}")
                print(f"  final_reason: {trial_log['final_reason']}")

                for a in trial_log["attempts"]:
                    print(f"  attempt {a['attempt']}:")
                    print(f"    safe_above_ok: {a['safe_above_ok']}")
                    print(f"    pre_grasp_ok: {a['pre_grasp_ok']}")
                    print(f"    grasp_ok: {a['grasp_ok']}")
                    print(f"    failure_reason: {a['failure_reason']}")

                    if a.get("preclose_diagnostics"):
                        d = a["preclose_diagnostics"]
                        print(
                            "    preclose_closest_axis: "
                            f"{d['closest_axis_candidate']}"
                        )
                        print(
                            "    preclose_axis_distance_m: "
                            f"{d['closest_axis_distance_m']:.4f}"
                        )
                        print(
                            "    preclose_tracking_error_m: "
                            f"{d['planned_flange_tracking_error_m']}"
                        )

                    if a["close_validation"]:
                        print(f"    close_success: {a['close_validation']['success']}")
                        print(f"    close_reasons: {a['close_validation']['reasons']}")

                    if a["lift_validation"]:
                        print(f"    lift_success: {a['lift_validation']['success']}")
                        print(f"    lift_reasons: {a['lift_validation']['reasons']}")

            # 6. Park robot after trial.
            await self.pick_executor.park_robot_after_trial()