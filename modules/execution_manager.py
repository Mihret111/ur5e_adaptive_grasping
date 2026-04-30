# Runs state machine and executes strategy

from modules import primitive_library as primitives

PRIMITIVE_MAP = {
    "approach": primitives.approach,
    "close": primitives.close,
    "close_gentle": primitives.close_gentle,
    "lift": primitives.lift,
    "place": primitives.place,
}

class ExecutionManager:
    async def run_strategy(self, strategy):
        print(f"[Execution] Running strategy: {strategy['name']}")

        for step in strategy["sequence"]:
            primitive_fn = PRIMITIVE_MAP.get(step)

            if primitive_fn:
                await primitive_fn()
            else:
                print(f"[Execution] Unknown primitive: {step}")

        print("[Execution] Strategy complete.")