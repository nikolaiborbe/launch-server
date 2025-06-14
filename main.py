import asyncio
import gc
from urllib import response
from fastapi import FastAPI
from contextlib import asynccontextmanager

from sim import get_data
from models import Day


#from rocket import get_rocket

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off the continuous simulation loop as soon as the app starts
    asyncio.create_task(simulate_loop())
    yield

app = FastAPI(lifespan=lifespan)


# Shared state: updated once per second by the background task
state: list[Day] = []

async def simulate_loop():
    global state
    while True:
        state = get_data()
        # Force a full garbage collection cycle after each simulation run
        # to ensure heavy objects from RocketPy are released promptly.
        gc.collect()
        await asyncio.sleep(1)


@app.get("/")
def root():
    return {
        "message": "Rocket simulation API. Use GET /status to fetch current metrics."
    }

@app.get("/status", response_model=list[Day])
def get_status():
    # Clients call this once per second (or however often you like)
    return state


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

