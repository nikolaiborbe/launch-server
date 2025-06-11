import asyncio
import random

state = {"estimate": 0.0, "points": 0}

async def simulate_mc_loop():
    global state
    inside = 0
    total = 0
    while True:
        for _ in range(1000):
            x, y = random.random(), random.random()
            total += 1
            if x * x + y * y <= 1.0:
                inside += 1
        state = {"estimate": 4 * inside / total, "points": total}
        await asyncio.sleep(1)
