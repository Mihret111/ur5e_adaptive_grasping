

from modules.object_context import build_object_profile
from modules.strategy_selector import select_strategy
from modules.execution_manager import ExecutionManager
from modules.scene_builder import SceneBuilder

class TrialRunner:
    def __init__(self, config, table_materials, table_slots, step_fn, step_seconds_fn):
        self.config = config
        self.table_materials = table_materials
        self.table_slots = table_slots

        self.step_fn = step_fn
        self.step_seconds_fn = step_seconds_fn
        self._total_attempts = 0
        self._total_successes = 0

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
            target = scene_info["pick_target"]
            print(f"[TrialRunner] Target: {target['label']}")

            # convert to profile
            object_profile = build_object_profile(target)
            print(f"[TrialRunner] Object Profile: {object_profile}")

            # print(f"[TrialRunner] Object: {object_profile['name']}")

            strategy = select_strategy(object_profile)

            await manager.run_strategy(strategy)