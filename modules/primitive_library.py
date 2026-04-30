# Implements primitives (executable building blocks)

import asyncio
from modules.motion_interface import MotionInterface


motion = None

def init_motion(flange_path): 
    global motion
    motion = MotionInterface(flange_path)


async def approach():
    print("[Primitive] Approaching object...")
    motion.move_up(-0.05)   # move down
    await asyncio.sleep(1)

async def close():
    print("[Primitive] Closing gripper (normal)...")
    await asyncio.sleep(1)   # will later try to close the gripper


async def close_gentle():
    print("[Primitive] Closing gripper (gentle)...")
    await asyncio.sleep(1)


async def lift():
    print("[Primitive] Lifting object...")
    motion.move_up(0.1)
    await asyncio.sleep(1)


async def place():
    print("[Primitive] Placing object...")
    motion.move_up(-0.1)
    await asyncio.sleep(1)