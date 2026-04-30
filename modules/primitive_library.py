# Implements primitives (executable building blocks)

import asyncio

async def approach():
    print("[Primitive] Approaching object...")
    await asyncio.sleep(1)


async def close():
    print("[Primitive] Closing gripper (normal)...")
    await asyncio.sleep(1)


async def close_gentle():
    print("[Primitive] Closing gripper (gentle)...")
    await asyncio.sleep(1)


async def lift():
    print("[Primitive] Lifting object...")
    await asyncio.sleep(1)


async def place():
    print("[Primitive] Placing object...")
    await asyncio.sleep(1)