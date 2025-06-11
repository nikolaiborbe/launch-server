from dataclasses import dataclass
import time
import zoneinfo
import pandas as pd
import numpy as np
import datetime 
from rocketpy import Environment, Rocket, Flight, LiquidMotor, CylindricalTank, MassFlowRateBasedTank, Fluid as RPFluid
from pyfluids import FluidsList, Mixture, Input, Fluid as PyFluid
from zoneinfo import ZoneInfo
from models import Day, Data, Weather, FlightData
import os
import contextlib
import csv
from time import process_time

from weather import construct_environment

import warnings

# ignore all FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

    

df = pd.read_excel("Input_values.xlsx", index_col=1)
header = df.iloc[1]
headers = df.iloc[0].values
df.columns = headers

analysis_parameters = {
    # Mass Details
    "rocket_length": df.loc["rocket_length"][1],
    "radius": df.loc["rocket_radius"][1],  # [m]
    "cg": df.loc["center_gravity"][1],# [m] from nose tip, dry weight
    "dry_mass": df.loc["rocket_mass"][1],  # [kg] without propellants
    "Ixx": df.loc["inertia_xx"][1],
    "Iyy": df.loc["inertia_yy"][1],
    "Izz": df.loc["inertia_zz"][1],
    "Ixy": df.loc["inertia_xy"][1],
    "Ixz": df.loc["inertia_xz"][1],
    "Iyz": df.loc["inertia_yz"][1],

    # ENGINE AND TANK DETAILS
    "burnout_time": 9,
    "tank_length": df.loc["fuel_length"][1],
    "fuel_tank_radius": df.loc["fuel_radius"][1],  # np.sqrt(tank_radius_outer**2 - tank_radius_inner**2)
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
    "fuel_pressure":df.loc["fuel_pressure"][1],        # 28 bar tank pressure
    "ox_pressure": df.loc["ox_pressure"][1],
    "n2_pressure": df.loc["n2_pressure"][1],
    "ox_temp": df.loc["ox_temp"][1],
    "fuel_temp": df.loc["fuel_temp"][1],
    "ambient_temp": df.loc["ambient_temp"][1],
    "ethanol_perc": df.loc["ethanol_perc"][1],
    "water_perc": df.loc["water_perc"][1],
    "mdot": df.loc["Massflowrate"][1],

    # Aerodynamic Details - run help(Rocket) for more information
    "nozzle_position": 0,
    "power_off_drag": 1,
    "power_on_drag": 1,
    "nose_length": df.loc["nose_length"][1],
    "fin_span": df.loc["fin_span"][1],
    "fin_root_chord": df.loc["rootchord"][1],
    "fin_tip_chord": df.loc["tipchord"][1],
    "beta": df.loc["fin_beta"][1],
    "sweep_length": (df.loc["fin_span"][1])/np.tan(np.deg2rad(df.loc["fin_beta"][1])),    # sweepLen = span / np.tan(np.deg2rad(beta))
    "fin_position": df.loc["fin_position"][1],
    "inclination": df.loc["inclination"][1],
    "heading": df.loc["heading"][1],
    #"inclination": (inclination[x], 20),
    #"heading": (heading[x], 2),
    "rail_length": df.loc["rail_length"][1],
    "ensemble_member": list(range(10)),
    "drogue_Cd_s": df.loc["drogue_cds"][1],
    "drogue_lag": df.loc["drogue_total_lag"][1],
    "main_Cd_s": df.loc["main_cds"][1],
    "main_lag": df.loc["main_total_lag"][1]

    }

def drogue_trigger(p, h, y): return y[5] < 0

def main_trigger(p, h, y): return y[5] < 0 and h <= 500

def coords_to_meters(lat, lon):
    """Convert latitude and longitude to meters."""
    R = 6371000  # Radius of the Earth in meters
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = R * lon_rad * np.cos(lat_rad)
    y = R * lat_rad
    return x, y

def worker(env: Environment) -> Data:
    today = datetime.datetime.now(ZoneInfo("Europe/Oslo"))

    fuel_T = 20
    ambient_T = 20
    K_to_C = lambda C: C-273.15 
    # Fluid states
    fuel_mix = Mixture(
        [FluidsList.Water, FluidsList.Ethanol], [25,75 ]).with_state(
            Input.pressure(df.loc["fuel_pressure"][1]*1e5), # 3800000 Pa tank pressure, 38 
            Input.temperature(df.loc["fuel_temp"][1]))
    N2O = PyFluid(FluidsList.NitrousOxide).with_state(
        Input.pressure(df.loc["ox_pressure"][1]*1e5), #must be over 18 bar or else it will not be liquid @ -20c
        Input.temperature(df.loc["ox_temp"][1])) #c 

    n2i = PyFluid(FluidsList.Nitrogen).with_state(
        Input.pressure(df.loc["n2_pressure"][1]*1e5), # 285 bar
        Input.temperature(ambient_T)) # Initial nitrogen state

    n2f = PyFluid(FluidsList.Nitrogen).with_state(
        Input.pressure(df.loc["n2_pressure"][1]*1e5), 
        Input.temperature(fuel_T)) # Nitrogen state after entering the other tanks
    
    # Geometries
    # Pull out the numeric values
    fuel_radius = analysis_parameters["fuel_tank_radius"]
    fuel_length = analysis_parameters["tank_length"]         # note: use "tank_length" here, not fuel_length from df

    ox_radius  = analysis_parameters["ox_tank_radius"]
    ox_length  = analysis_parameters["ox_tank_length"]

    n2_radius  = analysis_parameters["n2_tank_radius"]
    n2_length  = analysis_parameters["n2_tank_length"]

    # Now build the geometries with plain floats
    fuel_geom = CylindricalTank(fuel_radius, fuel_length, spherical_caps=False)

    n2_geom   = CylindricalTank(n2_radius,   n2_length,   spherical_caps=True)
    # Unpack the raw floats
    ox_radius = analysis_parameters["ox_tank_radius"]
    ox_length = analysis_parameters["ox_tank_length"]

    # Build the oxidizer‐tank geometry
    ox_geom = CylindricalTank(
        ox_radius,      # float radius in metres
        ox_length,      # float length in metres
        spherical_caps=False
    )

    
    # Properties
    oxidizer = RPFluid(name="N2O", density=N2O.density)
    fuel = RPFluid(name="fuel", density=fuel_mix.density)
    gas_i = RPFluid(name="gas_i", density=n2i.density)
    gas_f = RPFluid(name="gas_f", density=n2f.density)
    
    # Mass flow
    ox_mdot = df.loc["Massflowrate"][1] * df.loc["OF_ratio"][1] / (1+df.loc["OF_ratio"][1])
    prop_mdot = df.loc["Massflowrate"][1] - ox_mdot
    burnout_time = min(df.loc["fuel_mass"][1]/prop_mdot *0.99999, df.loc["ox_mass"][1] /ox_mdot *0.99999)
    n2_mdot = df.loc["n2_volume"][1] *1e-3 * n2i.density / burnout_time
    
    times = np.array([0.0, burnout_time])

    liquid_in_ox  = np.column_stack([times, np.zeros_like(times)])
    liquid_out_ox = np.column_stack([times, [ox_mdot, 0.0]])
    gas_in_ox     = np.column_stack([
        times,
        [df.loc["OF_ratio"][1] /(1+df.loc["OF_ratio"][1]) * n2_mdot/2, 0.0]
    ])
    gas_out_ox    = np.column_stack([times, np.zeros_like(times)])

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

    liquid_in_fuel  = np.column_stack([times, np.zeros_like(times)])
    liquid_out_fuel = np.column_stack([times, [prop_mdot, 0.0]])
    gas_in_fuel     = np.column_stack([
        times,
        [1/(1+df.loc["OF_ratio"][1]) * n2_mdot/2, 0.0]
    ])
    gas_out_fuel    = np.column_stack([times, np.zeros_like(times)])

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

    liquid_in_n2  = np.column_stack([times, np.zeros_like(times)])
    liquid_out_n2 = np.column_stack([times, np.zeros_like(times)])
    gas_in_n2     = np.column_stack([times, np.zeros_like(times)])
    gas_out_n2    = np.column_stack([times, [n2_mdot, 0.0]])

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

    # Thrust
    thrust_curve = np.loadtxt("inputs/rocketpyeng-mc.csv", delimiter=",", skiprows=1)
    thrust_curve[2, 0] = burnout_time - 0.5
    thrust_curve[3, 0] = burnout_time

    # Motor
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

    # Rocket
    heimdal = Rocket(
        radius = df.loc["rocket_radius"][1],
        mass = df.loc["rocket_mass"][1],
        inertia=(analysis_parameters["Ixx"], analysis_parameters["Iyy"], analysis_parameters["Izz"], analysis_parameters["Ixz"], analysis_parameters["Ixy"], analysis_parameters["Iyz"]),
        power_off_drag="inputs/drag.off.csv",
        power_on_drag="inputs/drag.on.csv",
        center_of_mass_without_motor=analysis_parameters["cg"],
        coordinate_system_orientation="nose_to_tail"
    )


    heimdal.add_nose(analysis_parameters["nose_length"],"von karman",0)
    heimdal.set_rail_buttons(analysis_parameters["rocket_length"]-3.1, analysis_parameters["rocket_length"]-1.1)
    heimdal.add_trapezoidal_fins(4, analysis_parameters["fin_root_chord"],analysis_parameters["fin_tip_chord"],analysis_parameters["fin_span"],analysis_parameters["fin_position"],0,analysis_parameters["sweep_length"])
    heimdal.add_motor(motor, position=analysis_parameters["rocket_length"])
    
    heimdal.add_parachute(
        'Drogue',
        cd_s=analysis_parameters["drogue_Cd_s"],
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=analysis_parameters["drogue_lag"]
    )

    heimdal.add_parachute(
        'Main',
        cd_s=analysis_parameters["main_Cd_s"],
        trigger=main_trigger,
        sampling_rate=105,
        lag=analysis_parameters["main_lag"]
    )
    
    # Flight
    flight = Flight(
        rocket = heimdal,
        environment = env,
        rail_length = analysis_parameters["rail_length"],
        inclination = analysis_parameters["inclination"],
        heading = analysis_parameters["heading"],
        max_time = 1500,
        terminate_on_apogee = False              
    )
    
    # Results
    pevents = flight.parachute_events or []
    n_events = len(pevents)

    d_t = None
    d_inf = None
    d_inf_v = None

    if n_events >= 2:
        # First event: drogue trigger time
        d_t = pevents[0][0]
        # Second event: drogue inflated time
        d_inf = pevents[1][0]
        # Velocity at inflation
        d_inf_v = flight.speed.get_value_opt(d_inf)

    out_rail_sm = flight.rocket.static_margin(flight.out_of_rail_time)
    burnout = flight.rocket.motor.burn_out_time
    final_sm = flight.rocket.static_margin(burnout)

    result = {
        "out_of_rail_time":       flight.out_of_rail_time,
        "out_of_rail_velocity":   flight.out_of_rail_velocity,
        "out_of_rail_static_margin": out_rail_sm,
        "max_velocity":           flight.max_speed,
        "apogee_time":            flight.apogee_time,
        "apogee_altitude":        flight.altitude.get_value_opt(flight.apogee_time),
        "apogee_x":               flight.apogee_x,
        "apogee_y":               flight.apogee_y,
        "impact_time":            flight.t_final,
        "impact_x":               flight.x_impact,
        "impact_y":               flight.y_impact,
        "impact_velocity":        flight.impact_velocity,
        "number_of_events":       n_events,
        "drogue_triggerTime":     d_t,
        "drogue_inflated_time":   d_inf,
        "drogue_inflated_velocity": d_inf_v,
        "final_static_margin":    final_sm,
    }

    t = np.arange(0, flight.t_final, 0.1)
    coords = np.column_stack([
        flight.x(t),
        flight.y(t),
        flight.z(t)
    ])

    flight_data = FlightData(t, coords)

    res = Data(result["max_velocity"], result["apogee_time"], result["apogee_altitude"], result["apogee_x"], result["apogee_y"], result["impact_x"], result["impact_y"], result["impact_velocity"], flight_data)

    return res



def get_data() -> list[Day]:
    ans = []
    env, weather = construct_environment(
        lat=df.loc["latitude"][1],
        lon=df.loc["longitude"][1],
        launch_time=datetime.datetime.now(ZoneInfo("Europe/Oslo")),
        climatology_file="inputs/MC_env.nc"
    )
    for i in range(len(weather)): 
        data: Data  = worker(env[i])
        cur_weather: Weather = Weather(
            time=weather[i].time,
            temperature=weather[i].temperature,
            pressure=weather[i].pressure,
            wind_speed=weather[i].wind_speed,
            wind_from_direction=weather[i].wind_from_direction,
            humidity=weather[i].humidity,
        )
        day: Day = Day(data, cur_weather)
        ans.append(day)
    return ans 

if __name__ == "__main__": 
    pass