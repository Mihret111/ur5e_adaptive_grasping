

from ament_index_python import constants
from launch.actions import reset_launch_configurations
from modules.object_context import build_object_profile
from modules.strategy_selector import select_strategy
from modules.execution_manager import ExecutionManager
from modules.scene_builder import SceneBuilder
from modules.target_exporter import export_moveit_target_command
from modules.pick_and_place_executor import PickAndPlaceExecutor


class TrialRunner:
    def __init__(self, config, table_materials, table_slots, step_fn, step_seconds_fn):
        self.config = config
        self.table_materials = table_materials
        self.table_slots = table_slots

        self.step_fn = step_fn
        self.step_seconds_fn = step_seconds_fn
        self._total_attempts = 0
        self._total_successes = 0

        self.pick_executor = PickAndPlaceExecutor(self.config)

        print(len(table_materials))
        # builds scene 
        self.scene_builder = SceneBuilder(
            config=self.config,
            table_materials=self.table_materials,
            table_seat_slots=self.table_slots,
        )
        print(self.scene_builder)
    async def run_all(self):
        num_trials = self.config.get("num_trials", 1)

        manager = ExecutionManager()

        for i in range(num_trials):
            print(f"[TrialRunner] Trial {i+1}/{num_trials}")

            scene_info = self.scene_builder.build_trial(i)

            ## pick 
            ok = await self.pick_executor.run_generic_pick(scene_info)

            if ok:
                self._total_successes += 1
                print(f"[TrialRunner] Trial {i} safe_above success")
            else:
                print(f"[TrialRunner] Trial {i} safe_above failed")

            ## Target selection 
            target = scene_info["pick_target"]
            target_local_base = scene_info["target_local_base"]

            export_moveit_target_command(
                output_path="/tmp/cogar_b2b/target_command.json",
                target=target,
                target_local_base=target_local_base,
                hover_height=0.40,
            )

            ## -------------------------------------------------------------
            ## Export the first target as a MoveIt command JSON file.
            ## -------------------------------------------------------------
            target = scene_info["pick_target"]

            # Construct a path under the run outputs directory.
            run_dir = self.config.get("paths", {}).get(
                "run_outputs_dir",
                "runs/isaac_run",
            )

            target_path = os.path.join(
                run_dir,
                f"trial_{i}",
                "target.json",
            )

            _ = export_moveit_target_command(
                output_path=target_path,
                target=target,
                target_local_base=scene_info["target_local_base"],
                hover_height=0.25,   # you can tune this
            )
            # -------------------------------------------------------------
            target = scene_info["pick_target"]
            print(f"[TrialRunner] Target: {target['label']}")

            # convert to profile
            object_profile = build_object_profile(target)
            print(f"[TrialRunner] Object Profile: {object_profile}")

            # print(f"[TrialRunner] Object: {object_profile['name']}")

            strategy = select_strategy(object_profile)

            await manager.run_strategy(strategy)