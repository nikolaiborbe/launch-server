import asyncio
from fastapi import FastAPI
from rocketpy import Flight
import random
from contextlib import asynccontextmanager

#from rocket import get_rocket

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off the continuous simulation loop as soon as the app starts
    asyncio.create_task(simulate_loop())
    yield

app = FastAPI(lifespan=lifespan)

# Shared state: updated once per second by the background task
state = {
    "landing": {"x": None, "y": None},
    "max_altitude": None,
    "max_speed": None,
    "wind": None
}

#rocket = get_rocket()

async def simulate_loop():
    t = 0
    while True:
        # Dummy simulation data (replace with real Flight call later)
        state["landing"]["x"] = t + 1
        state["landing"]["y"] = t * 2
        state["max_altitude"]   = 100 + t * 20
        state["max_speed"]      = 50 + t * 2
        state["wind"]           = random.uniform(0, 10)

        t += 1
        await asyncio.sleep(1)


@app.get("/")
def root():
    return {
        "message": "Rocket simulation API. Use GET /status to fetch current metrics."
    }

@app.get("/status")
def get_status():
    # Clients call this once per second (or however often you like)
    return state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)