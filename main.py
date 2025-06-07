import asyncio
from fastapi import FastAPI
from rocketpy import Flight
import random
from contextlib import asynccontextmanager

from rocketpy import Environment, SolidMotor, Rocket, Flight

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

# --- RocketPy simulation setup ---
# Initialize environment at launch coordinates and date
env = Environment(
    latitude=63.80263794391954,
    longitude=9.413957500356199,
    date=(2025, 6, 8, 12, 0, 0)
)

# Define motor using inline thrust curve data
# Simple inline thrust curve: list of [time (s), thrust (N)]
thrust_data = [
    [0.0, 0.0],
    [0.5, 500.0],
    [1.0, 1000.0],
    [2.0, 2000.0],
    [3.0, 1500.0],
    [4.0, 1000.0],
    [5.0, 0.0]
]
motor = SolidMotor(
    thrust_source=thrust_data,
    burn_time=5,
    grain_number=5,
    grain_separation=5e-3,
    grain_density=1800,
    grain_outer_radius=0.075,
    grain_initial_inner_radius=0.035,
    grain_initial_height=0.230,
    nozzle_radius=0.05
)

# Define rocket geometry and mass properties
rocket = Rocket(
    motor=motor,
    radius=0.1,
    mass=19.7,
    inertiaI=0.1,
    inertiaZ=0.1
)

# Create and run the flight simulation
flight = Flight(rocket=rocket, environment=env, rail_length=5, inclination=90, heading=0)
flight.run()

# Extract trajectory and performance data
times = flight.all_info["t"]
x_data = flight.all_info["x"]
z_data = flight.all_info["z"]     # altitude
v_data = flight.all_info["v"]     # speed
flight_index = 0
# ----------------------------------

#rocket = get_rocket()

async def simulate_loop():
    global flight_index
    while True:
        if flight_index < len(times):
            # Update position and performance up to this time-step
            state["landing"]["x"] = x_data[flight_index]
            state["landing"]["y"] = z_data[flight_index]
            state["max_altitude"] = max(z_data[:flight_index+1])
            state["max_speed"] = max(v_data[:flight_index+1])
            state["wind"] = random.uniform(0, 10)
            flight_index += 1
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