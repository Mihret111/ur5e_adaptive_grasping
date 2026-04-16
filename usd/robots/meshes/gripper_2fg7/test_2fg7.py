import omni.usd
import asyncio
from pxr import UsdPhysics

stage = omni.usd.get_context().get_stage()

# ══════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════
GRIP_FORCE = 80.0
STIFFNESS = 1e6
DAMPING = 1e4
STEPS_PER_MOVE = 180
NUM_CYCLES = 3

# ── Find joints ──
left_joint = None
right_joint = None
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.PrismaticJoint):
        pp = str(prim.GetPath()).lower()
        if "left_finger" in pp:
            left_joint = prim
        elif "right_finger" in pp:
            right_joint = prim

# ── FIX: Clamp right joint limits so it can't go negative ──
if right_joint:
    jp = UsdPhysics.PrismaticJoint(right_joint)
    jp.GetLowerLimitAttr().Set(0.0)
    jp.GetUpperLimitAttr().Set(0.019)
    print("Fixed right joint limits: [0.0, 0.019]")

# ── Apply force ──
for joint in [left_joint, right_joint]:
    if joint:
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        if drive:
            drive.GetMaxForceAttr().Set(GRIP_FORCE)
            drive.GetStiffnessAttr().Set(STIFFNESS)
            drive.GetDampingAttr().Set(DAMPING)

print("Force: " + str(GRIP_FORCE) + "N")


def set_gripper(value):
    """
    value=0.0   -> OPEN  (fingers spread apart)
    value=0.019 -> CLOSED (fingers together)
    Both joints get the SAME positive value.
    """
    for joint in [left_joint, right_joint]:
        if joint:
            UsdPhysics.DriveAPI.Get(joint, "linear").GetTargetPositionAttr().Set(value)


async def animate():
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
        for i in range(30):
            await omni.kit.app.get_app().next_update_async()

    print("\n=== Gripper Animation ===")
    print("0.0   = OPEN")
    print("0.019 = CLOSED")
    print("Force = " + str(GRIP_FORCE) + "N")
    print("")

    for cycle in range(NUM_CYCLES):
        print("[Cycle " + str(cycle + 1) + "] OPEN")
        set_gripper(0.0)
        for i in range(STEPS_PER_MOVE):
            await omni.kit.app.get_app().next_update_async()

        print("[Cycle " + str(cycle + 1) + "] CLOSE")
        set_gripper(0.019)
        for i in range(STEPS_PER_MOVE):
            await omni.kit.app.get_app().next_update_async()

    # End open
    set_gripper(0.0)
    for i in range(60):
        await omni.kit.app.get_app().next_update_async()

    print("\n=== Done! ===")

asyncio.ensure_future(animate())
