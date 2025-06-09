import asyncio
from dataclasses import dataclass
from fastapi import FastAPI
import random
from contextlib import asynccontextmanager
from sim import Data, Weather, get_data


#from rocket import get_rocket

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off the continuous simulation loop as soon as the app starts
    asyncio.create_task(simulate_loop())
    yield

app = FastAPI(lifespan=lifespan)


@dataclass
class API:
    data: Data
    weather: Weather
# Shared state: updated once per second by the background task
state = API(
    data=Data(0, 0, 0, 0, 0, 0),
    weather=Weather(0, 0, 0, 0, 0)
)

async def simulate_loop():
    global flight_index
    while True:
        global state
        state = get_data()
        print(state)
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