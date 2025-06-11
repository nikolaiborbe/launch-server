import asyncio
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from rocketpy import Environment, Rocket, Flight, LiquidMotor, CylindricalTank, MassFlowRateBasedTank, Fluid as RPFluid
from pyfluids import FluidsList, Mixture, Input, Fluid as PyFluid

from models import Day, Data, Weather, FlightData
from weather import construct_environment

# ---------------------------------------------------------------------------
# Read input parameters exactly like sim.py
# ---------------------------------------------------------------------------
df = pd.read_excel("Input_values.xlsx", index_col=1)
headers = df.iloc[0].values
df.columns = headers

analysis_parameters = {
    "rocket_length": df.loc["rocket_length"][1],
    "radius": df.loc["rocket_radius"][1],
    "cg": df.loc["center_gravity"][1],
    "dry_mass": df.loc["rocket_mass"][1],
    "Ixx": df.loc["inertia_xx"][1],
    "Iyy": df.loc["inertia_yy"][1],
    "Izz": df.loc["inertia_zz"][1],
    "Ixy": df.loc["inertia_xy"][1],
    "Ixz": df.loc["inertia_xz"][1],
    "Iyz": df.loc["inertia_yz"][1],

    "burnout_time": 9,
    "tank_length": df.loc["fuel_length"][1],
    "fuel_tank_radius": df.loc["fuel_radius"][1],
    "ox_tank_length": df.loc["ox_length"][1],
    "ox_tank_radius": df.loc["ox_eff_radius"][1],
    "n2_tank_radius": df.loc["n2_radius"][1],
    "n2_tank_length": df.loc["n2_length"][1],
    "fuel_tank_position": df.loc["fuel_position"][1],
    "ox_tank_position": df.loc["ox_position"][1],
    "n2_tank_position": df.loc["n2_position"][1],
    "fuel_mass": df.loc["fuel_mass"][1],
    "n2_volume": df.loc["n2_volume"][1],
    "ox_mass": df.loc["ox_mass"][1],
    "of": df.loc["OF_ratio"][1],
    "fuel_pressure": df.loc["fuel_pressure"][1],
    "ox_pressure": df.loc["ox_pressure"][1],
    "n2_pressure": df.loc["n2_pressure"][1],
    "ox_temp": df.loc["ox_temp"][1],
    "fuel_temp": df.loc["fuel_temp"][1],
    "ambient_temp": df.loc["ambient_temp"][1],
    "ethanol_perc": df.loc["ethanol_perc"][1],
    "water_perc": df.loc["water_perc"][1],
    "mdot": df.loc["Massflowrate"][1],

    "nozzle_position": 0,
    "power_off_drag": 1,
    "power_on_drag": 1,
    "nose_length": df.loc["nose_length"][1],
    "fin_span": df.loc["fin_span"][1],
    "fin_root_chord": df.loc["rootchord"][1],
    "fin_tip_chord": df.loc["tipchord"][1],
    "beta": df.loc["fin_beta"][1],
    "sweep_length": (df.loc["fin_span"][1])/np.tan(np.deg2rad(df.loc["fin_beta"][1])),
    "fin_position": df.loc["fin_position"][1],
    "inclination": df.loc["inclination"][1],
    "heading": df.loc["heading"][1],
    "rail_length": df.loc["rail_length"][1],
    "drogue_Cd_s": df.loc["drogue_cds"][1],
    "drogue_lag": df.loc["drogue_total_lag"][1],
    "main_Cd_s": df.loc["main_cds"][1],
    "main_lag": df.loc["main_total_lag"][1],
}


def drogue_trigger(p, h, y):
    return y[5] < 0


def main_trigger(p, h, y):
    return y[5] < 0 and h <= 500


# ---------------------------------------------------------------------------
# Worker function that builds the rocket and runs the Flight simulation
# ---------------------------------------------------------------------------
def worker_mc(env: Environment, inclination: float, heading: float) -> Data:
    fuel_T = 20
    ambient_T = 20

    fuel_mix = Mixture(
        [FluidsList.Water, FluidsList.Ethanol], [25, 75]
    ).with_state(
        Input.pressure(df.loc["fuel_pressure"][1] * 1e5),
        Input.temperature(df.loc["fuel_temp"][1])
    )
    N2O = PyFluid(FluidsList.NitrousOxide).with_state(
        Input.pressure(df.loc["ox_pressure"][1] * 1e5),
        Input.temperature(df.loc["ox_temp"][1])
    )
    n2i = PyFluid(FluidsList.Nitrogen).with_state(
        Input.pressure(df.loc["n2_pressure"][1] * 1e5),
        Input.temperature(ambient_T)
    )
    n2f = PyFluid(FluidsList.Nitrogen).with_state(
        Input.pressure(df.loc["n2_pressure"][1] * 1e5),
        Input.temperature(fuel_T)
    )

    # Tank geometries
    fuel_geom = CylindricalTank(
        analysis_parameters["fuel_tank_radius"],
        analysis_parameters["tank_length"],
        spherical_caps=False,
    )
    ox_geom = CylindricalTank(
        analysis_parameters["ox_tank_radius"],
        analysis_parameters["ox_tank_length"],
        spherical_caps=False,
    )
    n2_geom = CylindricalTank(
        analysis_parameters["n2_tank_radius"],
        analysis_parameters["n2_tank_length"],
        spherical_caps=True,
    )

    oxidizer = RPFluid(name="N2O", density=N2O.density)
    fuel = RPFluid(name="fuel", density=fuel_mix.density)
    gas_i = RPFluid(name="gas_i", density=n2i.density)
    gas_f = RPFluid(name="gas_f", density=n2f.density)

    ox_mdot = df.loc["Massflowrate"][1] * df.loc["OF_ratio"][1] / (1 + df.loc["OF_ratio"][1])
    prop_mdot = df.loc["Massflowrate"][1] - ox_mdot
    burnout_time = min(
        df.loc["fuel_mass"][1] / prop_mdot * 0.99999,
        df.loc["ox_mass"][1] / ox_mdot * 0.99999,
    )
    n2_mdot = df.loc["n2_volume"][1] * 1e-3 * n2i.density / burnout_time

    times = np.array([0.0, burnout_time])
    liquid_in_ox = np.column_stack([times, np.zeros_like(times)])
    liquid_out_ox = np.column_stack([times, [ox_mdot, 0.0]])
    gas_in_ox = np.column_stack([times, [df.loc["OF_ratio"][1] / (1 + df.loc["OF_ratio"][1]) * n2_mdot / 2, 0.0]])
    gas_out_ox = np.column_stack([times, np.zeros_like(times)])

    ox_tank = MassFlowRateBasedTank(
        name="ox",
        geometry=ox_geom,
        flux_time=(0.0, burnout_time),
        liquid=oxidizer,
        gas=gas_f,
        initial_liquid_mass=df.loc["ox_mass"][1],
        initial_gas_mass=0.0,
        liquid_mass_flow_rate_in=liquid_in_ox,
        gas_mass_flow_rate_in=gas_in_ox,
        liquid_mass_flow_rate_out=liquid_out_ox,
        gas_mass_flow_rate_out=gas_out_ox,
    )

    liquid_in_fuel = np.column_stack([times, np.zeros_like(times)])
    liquid_out_fuel = np.column_stack([times, [prop_mdot, 0.0]])
    gas_in_fuel = np.column_stack([times, [1 / (1 + df.loc["OF_ratio"][1]) * n2_mdot / 2, 0.0]])
    gas_out_fuel = np.column_stack([times, np.zeros_like(times)])

    fuel_tank = MassFlowRateBasedTank(
        name="fuel",
        geometry=fuel_geom,
        flux_time=(0.0, burnout_time),
        liquid=fuel,
        gas=gas_f,
        initial_liquid_mass=df.loc["fuel_mass"][1],
        initial_gas_mass=0.0,
        liquid_mass_flow_rate_in=liquid_in_fuel,
        gas_mass_flow_rate_in=gas_in_fuel,
        liquid_mass_flow_rate_out=liquid_out_fuel,
        gas_mass_flow_rate_out=gas_out_fuel,
    )

    liquid_in_n2 = np.column_stack([times, np.zeros_like(times)])
    liquid_out_n2 = np.column_stack([times, np.zeros_like(times)])
    gas_in_n2 = np.column_stack([times, np.zeros_like(times)])
    gas_out_n2 = np.column_stack([times, [n2_mdot, 0.0]])

    n2_tank = MassFlowRateBasedTank(
        name="n2",
        geometry=n2_geom,
        flux_time=(0.0, burnout_time),
        liquid=gas_i,
        gas=gas_i,
        initial_liquid_mass=0.0,
        initial_gas_mass=df.loc["n2_volume"][1] * 1e-3 * n2i.density,
        liquid_mass_flow_rate_in=liquid_in_n2,
        gas_mass_flow_rate_in=gas_in_n2,
        liquid_mass_flow_rate_out=liquid_out_n2,
        gas_mass_flow_rate_out=gas_out_n2,
    )

    thrust_curve = np.loadtxt("inputs/rocketpyeng-mc.csv", delimiter=",", skiprows=1)
    thrust_curve[2, 0] = burnout_time - 0.5
    thrust_curve[3, 0] = burnout_time

    motor = LiquidMotor(
        thrust_source=thrust_curve,
        center_of_dry_mass_position=0,
        dry_inertia=(0, 0, 0),
        dry_mass=0.1,
        burn_time=(0, burnout_time),
        nozzle_radius=0.068,
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    motor.add_tank(fuel_tank, position=analysis_parameters["fuel_tank_position"])
    motor.add_tank(ox_tank, position=analysis_parameters["ox_tank_position"])
    motor.add_tank(n2_tank, position=analysis_parameters["n2_tank_position"])

    heimdal = Rocket(
        radius=df.loc["rocket_radius"][1],
        mass=df.loc["rocket_mass"][1],
        inertia=(
            analysis_parameters["Ixx"],
            analysis_parameters["Iyy"],
            analysis_parameters["Izz"],
            analysis_parameters["Ixz"],
            analysis_parameters["Ixy"],
            analysis_parameters["Iyz"],
        ),
        power_off_drag="inputs/drag.off.csv",
        power_on_drag="inputs/drag.on.csv",
        center_of_mass_without_motor=analysis_parameters["cg"],
        coordinate_system_orientation="nose_to_tail",
    )

    heimdal.add_nose(analysis_parameters["nose_length"], "von karman", 0)
    heimdal.set_rail_buttons(
        analysis_parameters["rocket_length"] - 3.1,
        analysis_parameters["rocket_length"] - 1.1,
    )
    heimdal.add_trapezoidal_fins(
        4,
        analysis_parameters["fin_root_chord"],
        analysis_parameters["fin_tip_chord"],
        analysis_parameters["fin_span"],
        analysis_parameters["fin_position"],
        0,
        analysis_parameters["sweep_length"],
    )
    heimdal.add_motor(motor, position=analysis_parameters["rocket_length"])

    heimdal.add_parachute(
        "Drogue",
        cd_s=analysis_parameters["drogue_Cd_s"],
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=analysis_parameters["drogue_lag"],
    )
    heimdal.add_parachute(
        "Main",
        cd_s=analysis_parameters["main_Cd_s"],
        trigger=main_trigger,
        sampling_rate=105,
        lag=analysis_parameters["main_lag"],
    )

    flight = Flight(
        rocket=heimdal,
        environment=env,
        rail_length=analysis_parameters["rail_length"],
        inclination=inclination,
        heading=heading,
        max_time=1500,
        terminate_on_apogee=False,
    )

    pevents = flight.parachute_events or []
    n_events = len(pevents)
    d_t = d_inf = d_inf_v = None
    if n_events >= 2:
        d_t = pevents[0][0]
        d_inf = pevents[1][0]
        d_inf_v = flight.speed.get_value_opt(d_inf)

    out_rail_sm = flight.rocket.static_margin(flight.out_of_rail_time)
    burnout = flight.rocket.motor.burn_out_time
    final_sm = flight.rocket.static_margin(burnout)

    result = {
        "out_of_rail_time": flight.out_of_rail_time,
        "out_of_rail_velocity": flight.out_of_rail_velocity,
        "out_of_rail_static_margin": out_rail_sm,
        "max_velocity": flight.max_speed,
        "apogee_time": flight.apogee_time,
        "apogee_altitude": flight.altitude.get_value_opt(flight.apogee_time),
        "apogee_x": flight.apogee_x,
        "apogee_y": flight.apogee_y,
        "impact_time": flight.t_final,
        "impact_x": flight.x_impact,
        "impact_y": flight.y_impact,
        "impact_velocity": flight.impact_velocity,
        "number_of_events": n_events,
        "drogue_triggerTime": d_t,
        "drogue_inflated_time": d_inf,
        "drogue_inflated_velocity": d_inf_v,
        "final_static_margin": final_sm,
    }

    t = np.arange(0, flight.t_final, 0.1).tolist()
    coords = np.column_stack([flight.x(t), flight.y(t), flight.z(t)]).tolist()
    flight_data = FlightData(t, coords)

    return Data(
        result["max_velocity"],
        result["apogee_time"],
        result["apogee_altitude"],
        result["apogee_x"],
        result["apogee_y"],
        result["impact_x"],
        result["impact_y"],
        result["impact_velocity"],
        flight_data,
    )


# ---------------------------------------------------------------------------
# Aggregate results from several Monte Carlo runs
# ---------------------------------------------------------------------------
def aggregate(results: list[Data]) -> Data:
    if not results:
        raise ValueError("No Monte Carlo results")
    n = len(results)
    avg = lambda attr: float(np.mean([getattr(r, attr) for r in results]))
    first_fd = results[0].flight_data
    return Data(
        avg("max_velocity"),
        avg("apogee_time"),
        avg("apogee_altitude"),
        avg("apogee_x"),
        avg("apogee_y"),
        avg("impact_x"),
        avg("impact_y"),
        avg("impact_velocity"),
        first_fd,
    )


# ---------------------------------------------------------------------------
# Continuous Monte Carlo loop
# ---------------------------------------------------------------------------
state: list[Day] = []

async def simulate_mc_loop(iterations: int = 5) -> None:
    global state
    while True:
        envs, weather = construct_environment(
            lat=float(df.loc["latitude"][1]),
            lon=float(df.loc["longitude"][1]),
            launch_time=datetime.datetime.now(ZoneInfo("Europe/Oslo")),
            climatology_file="inputs/MC_env.nc",
        )
        new_state: list[Day] = []
        for env, w in zip(envs, weather):
            runs = []
            for _ in range(iterations):
                h = analysis_parameters["heading"] + np.random.normal(0, 0.5)
                inc = analysis_parameters["inclination"] + np.random.normal(0, 0.2)
                runs.append(worker_mc(env, inc, h))
            data = aggregate(runs)
            cur_weather = Weather(
                time=w.time,
                temperature=w.temperature,
                pressure=w.pressure,
                wind_speed=w.wind_speed,
                wind_from_direction=w.wind_from_direction,
                humidity=w.humidity,
            )
            new_state.append(Day(data, cur_weather))
        state = new_state
        await asyncio.sleep(1)
