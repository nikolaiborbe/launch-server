import pandas as pd
import numpy as np
from rocketpy import Environment, Rocket, Flight, LiquidMotor, CylindricalTank, MassFlowRateBasedTank, Fluid as RPFluid
from pyfluids import FluidsList, Mixture, Input, Fluid as PyFluid
import os
import contextlib
import csv
from time import process_time


_BASE_THRUST = None

df = pd.read_excel("Input_values.xlsx", index_col=1)
thrust_curve = np.loadtxt("inputs/rocketpyeng-mc.csv", delimiter=",", skiprows=1)

def drogue_trigger(p, h, y): return y[5] < 0

def main_trigger(p, h, y): return y[5] < 0 and h <= 500

def worker() -> dict:
        # TODO: Update date
    Env = Environment(
        date=(2024, 7, 2, 12),
        latitude=df.loc["latitude"],
        longitude=df.loc["longitude"],
        max_expected_height=12000,
        elevation=20
    )

    Env.set_atmospheric_model(type="Reanalysis", file="inputs/MC_env.nc", dictionary="ECMWF")
    
    # Fluid states
    fuel_mix = Mixture([FluidsList.Water, FluidsList.Ethanol], [df.loc["water_perc"], df.loc["ethanol_perc"]]).with_state(
        Input.pressure(df.loc["fuel_pressure"]), Input.temperature(df.loc["fuel_temp"]))
    N2O = PyFluid(FluidsList.NitrousOxide).with_state(
        Input.pressure(df.loc["ox_pressure"]), Input.temperature(df.loc["ox_temp"]))
    n2i = PyFluid(FluidsList.Nitrogen).with_state(
        Input.pressure(df.loc["n2_pressure"]), Input.temperature(df.loc["ambient_temp"]))
    n2f = PyFluid(FluidsList.Nitrogen).with_state(
        Input.pressure(df.loc["n2_pressure"]), Input.temperature(df.loc["fuel_temp"]))
    
    # Geometries
    fuel_geom = CylindricalTank(df.loc["fuel_tank_radius"], df.loc["fuel_length"], spherical_caps=False)
    ox_geom = CylindricalTank(df.loc["ox_tank_radius"], df.loc["ox_length"], spherical_caps=False)
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        n2_geom = CylindricalTank(df.loc["n2_tank_radius"], df.loc["n2_length"], spherical_caps=True)

    
    # Properties
    oxidizer = RPFluid(name="N2O", density=N2O.density)
    fuel = RPFluid(name="fuel", density=fuel_mix.density)
    gas_i = RPFluid(name="gas_i", density=n2i.density)
    gas_f = RPFluid(name="gas_f", density=n2f.density)
    
    # Mass flow
    ox_mdot = df.loc["Massflowrate"] * df.loc["OF_ratio"]/ (1+df.loc["OF_ratio"])
    prop_mdot = df.loc["Massflowrate"] - ox_mdot
    burnout_time = min(df.loc["fuel_mass"]/prop_mdot *0.99999, df.loc["ox_mass"]/ox_mdot *0.99999)
    n2_mdot = df.loc["n2_volume"]*1e-3 * n2i.density / burnout_time
    
    times = np.array([0.0, burnout_time])

    liquid_in_ox  = np.column_stack([times, np.zeros_like(times)])
    liquid_out_ox = np.column_stack([times, [ox_mdot, 0.0]])
    gas_in_ox     = np.column_stack([
        times,
        [df.loc["OF_ratio"]/(1+df.loc["OF_ratio"]) * n2_mdot/2, 0.0]
    ])
    gas_out_ox    = np.column_stack([times, np.zeros_like(times)])

    ox_tank = MassFlowRateBasedTank(
        name="ox",
        geometry=ox_geom,
        flux_time=(0.0, burnout_time),
        liquid=oxidizer,
        gas=gas_f,
        initial_liquid_mass=df.loc["ox_mass"],
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
        [1/(1+df.loc["OF_ratio"]) * n2_mdot/2, 0.0]
    ])
    gas_out_fuel    = np.column_stack([times, np.zeros_like(times)])

    fuel_tank = MassFlowRateBasedTank(
        name="fuel",
        geometry=fuel_geom,
        flux_time=(0.0, burnout_time),
        liquid=fuel,
        gas=gas_f,
        initial_liquid_mass=df.loc["fuel_mass"],
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
        initial_gas_mass=df.loc["n2_volume"] * 1e-3 * n2i.density,
        liquid_mass_flow_rate_in=liquid_in_n2,
        gas_mass_flow_rate_in=gas_in_n2,
        liquid_mass_flow_rate_out=liquid_out_n2,
        gas_mass_flow_rate_out=gas_out_n2,
    )

    # Thrust
    thrust_curve = _BASE_THRUST.copy()
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
    motor.add_tank(fuel_tank, position=df.loc["fuel_tank_position"])
    motor.add_tank(ox_tank, position=df.loc["ox_tank_position"])
    motor.add_tank(n2_tank, position=df.loc["n2_tank_position"])
    
    # Rocket
    heimdal = Rocket(
        radius = df.loc["radius"],
        mass = df.loc["dry_mass"],
        inertia=(df.loc["Ixx"], df.loc["Iyy"], df.loc["Izz"], df.loc["Ixz"], df.loc["Ixy"], df.loc["Iyz"]),
        power_off_drag="inputs/drag.off.csv",
        power_on_drag="inputs/drag.on.csv",
        center_of_mass_without_motor=df.loc["cg"],
        coordinate_system_orientation="nose_to_tail"
    )


    heimdal.add_nose(df.loc["nose_length"],"von karman",0)
    heimdal.set_rail_buttons(df.loc["rocket_length"]-3.1, df.loc["rocket_length"]-1.1)
    heimdal.add_trapezoidal_fins(4, df.loc["fin_root_chord"],df.loc["fin_tip_chord"],df.loc["fin_span"],df.loc["fin_position"],0,df.loc["sweep_length"])
    heimdal.add_motor(motor, position=df.loc["rocket_length"])
    
    heimdal.add_parachute(
        'Drogue',
        cd_s=df.loc["drogue_Cd_s"],
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=df.loc["drogue_lag"]
    )

    heimdal.add_parachute(
        'Main',
        cd_s=df.loc["main_Cd_s"],
        trigger=main_trigger,
        sampling_rate=105,
        lag=df.loc["main_lag"]
    )
    
    # Flight
    flight = Flight(
        rocket = heimdal,
        environment = Env,
        rail_length = df.loc["rail_length"],
        inclination = df.loc["inclination"],
        heading = df.loc["heading"],
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

    return result

        
if __name__ == "__main__": 
    result = worker()
    print(result[1])