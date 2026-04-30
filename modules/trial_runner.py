"""
To be written
"""
from modules.object_context import get_object_profile
from modules.strategy_selector import select_strategy
from modules.execution_manager import ExecutionManager

class TrialRunner:
    def __init__(self, config, step_fn, step_seconds_fn):
        self.config = config
        self.step_fn = step_fn
        self.step_seconds_fn = step_seconds_fn
        self._total_attempts = 0
        self._total_successes = 0

    async def run_all(self):
        num_trials = self.config.get("num_trials", 1)

        manager = ExecutionManager()

        for i in range(num_trials):
            print(f"[TrialRunner] Trial {i+1}/{num_trials}")

            object_profile = get_object_profile()
            print(f"[TrialRunner] Object: {object_profile['name']}")

            strategy = select_strategy(object_profile)

            await manager.run_strategy(strategy)