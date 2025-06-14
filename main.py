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
        # Run the full‑fidelity simulation
        raw_state = get_data()

        # ----------------------------------------------------------
        # Strip heavy per‑point time‑series before exposing via API
        # so that concurrent /status calls don't duplicate megabytes
        # of arrays in RAM.
        # ----------------------------------------------------------
        slim_state: list[Day] = []
        for day in raw_state:
            # Mutate in place: drop the large FlightData arrays but keep scalars
            try:
                fd = day.data.flight_data
                # Remove big arrays to minimise memory footprint
                fd.t = []
                fd.coords = []
            except AttributeError:
                # Model layout changed; ignore gracefully
                pass
            slim_state.append(day)

        # Atomically swap the shared reference
        state = slim_state

        # Force a GC cycle to promptly release orphaned arrays
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

