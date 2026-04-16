"""
Thin wrappers around Isaac Sim timeline and frame-stepping APIs.
"""

import omni.kit.app
import omni.timeline


# ═══════════════════════════════════════════════════════════════
# FRAME STEPPING
# ═══════════════════════════════════════════════════════════════

async def step_simulation(num_steps: int = 1) -> None:
    """
    Advance Isaac Sim by `num_steps` frames.

    Args:
        num_steps: Number of update frames to await.
                   Minimum clamped to 1.
    """
    app = omni.kit.app.get_app()
    num_steps = max(1, num_steps)
    for _ in range(num_steps):
        await app.next_update_async()


async def step_simulation_seconds(
    seconds: float,
    dt: float = 1.0 / 60.0,
) -> None:
    """
    Step the simulation for approximately `seconds` of sim-time.

    Args:
        seconds: Target duration in simulated seconds.
        dt:      Duration of one frame in seconds.
                 Defaults to 1/60 (60 Hz).
    """
    num_steps = max(1, int(seconds / dt))
    print(
        f"  [sim_utils] Stepping {num_steps} frames "
        f"(~{seconds:.1f}s at {1.0 / dt:.0f} Hz)..."
    )
    await step_simulation(num_steps)


# ═══════════════════════════════════════════════════════════════
# TIMELINE CONTROL
# ═══════════════════════════════════════════════════════════════

def start_simulation() -> None:
    """
    Start (play) the Isaac Sim timeline.
    Safe to call if already playing — logs current state.
    """
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
        print("  [sim_utils] ▶️  Timeline PLAY")
    else:
        print("  [sim_utils] ▶️  Timeline already playing")


def stop_simulation() -> None:
    """
    Stop the Isaac Sim timeline.
    Safe to call if already stopped — logs current state.
    """
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.stop()
        print("  [sim_utils] ⏹️  Timeline STOP")
    else:
        print("  [sim_utils] ⏹️  Timeline already stopped")