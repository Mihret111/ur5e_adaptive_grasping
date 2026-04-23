"""
To be written
"""

class TrialRunner:
    def __init__(self, config, step_fn, step_seconds_fn):
        self.config = config
        self.step_fn = step_fn
        self.step_seconds_fn = step_seconds_fn
        self._total_attempts = 0
        self._total_successes = 0

    async def run_all(self):
        num_trials = self.config.get("num_trials", 1)

        print(f"  [TrialRunner] Running {num_trials} dummy trial(s)...")

        for i in range(num_trials):
            self._total_attempts += 1
            print(f"  [TrialRunner] Trial {i+1}/{num_trials}")
            await self.step_seconds_fn(1.0)
            self._total_successes += 1

        print("  [TrialRunner] Dummy run complete.")