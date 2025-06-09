import numpy as np
import xarray as xr
from datetime import datetime
from zoneinfo import ZoneInfo
from rocketpy import Environment
import requests

# constant
g = 9.80665  # m/s²

def get_met_current_weather(lat: float, lon: float):
    """
    Query MET Norway's LocationForecast API for the current weather at the given lat/lon.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        dict: Current weather details with local Europe/Oslo datetime under key 'time'.
    """
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
    headers = {
        "User-Agent": "rocketpy-demo (nikolaiborbe@gmail.com)",
        "Accept": "application/json"
    }
    params = {"lat": lat, "lon": lon}

    # Fetch data
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Extract timeseries and current entry
    timeseries = data.get("properties", {}).get("timeseries", [])
    if not timeseries:
        raise RuntimeError("No timeseries returned from MET API")

    first = timeseries[0]
    ts = first["time"]
    # Convert ISO 'Z' timestamp to timezone-aware datetime
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    obs_time = datetime.fromisoformat(ts).astimezone(ZoneInfo("Europe/Oslo"))

    # Combine time and weather details
    current = {
        "time": obs_time,
        **first["data"]["instant"]["details"]
    }
    return current




def construct_environment(
    lat: float,
    lon: float,
    launch_time: datetime,
    climatology_file: str,
) -> Environment:
    """
    Build a RocketPy Environment at (lat, lon) and launch_time,
    using MC_env.nc climatology + current MET obs.

    Internally this:
      - fetches current_obs from MET
      - slices MC_env.nc at nearest (time,lat,lon)
      - converts level→pressure, z→height
      - applies a surface‐temp anomaly
      - builds custom_atmosphere
    """
    # --- fetch the current MET surface obs ---
    current_obs = get_met_current_weather(lat, lon)
    T_obs   = current_obs["air_temperature"]      # °C
    wspd    = current_obs["wind_speed"]           # m/s
    wdir    = current_obs["wind_from_direction"]  # deg from north

    # --- load & slice your reanalysis NetCDF ---
    ds = xr.open_dataset(climatology_file)
    ts = np.datetime64(launch_time.astimezone(ZoneInfo("UTC")))
    clim = ds.sel(
        time      = ts,
        latitude  = lat,
        longitude = lon,
        method    = "nearest"
    )

    # --- extract and convert coordinates & variables ---
    levels_hpa   = clim["level"].values            # [hPa]
    pressure_pa  = levels_hpa * 100.0              # → Pa

    geopot       = clim["z"].values                # m²/s²
    height       = (geopot / g).astype(float)      # m

    T_profile    = clim["t"].values                # K
    u_profile    = clim["u"].values                # m/s
    v_profile    = clim["v"].values                # m/s

    # --- apply surface‐temp anomaly (MET in °C → K) ---
    T_obs_K      = T_obs + 273.15
    T_surf_clim  = np.interp(0.0, height, T_profile)
    delta_T      = T_obs_K - T_surf_clim
    temperature_profile = [
        (float(h), float(Ti + delta_T))
        for h, Ti in zip(height, T_profile)
    ]

    # --- build constant wind profiles from the obs ---
    θ     = np.deg2rad(wdir)
    u0    = -wspd * np.sin(θ)
    v0    = -wspd * np.cos(θ)
    wind_u_profile = [(float(h), float(u0)) for h in height]
    wind_v_profile = [(float(h), float(v0)) for h in height]

    # --- pressure profile ---
    pressure_profile = [
        (float(h), float(p))
        for h, p in zip(height, pressure_pa)
    ]

    # --- construct the RocketPy Environment ---
    env = Environment(
        max_expected_height=12000,
        latitude   = lat,
        longitude  = lon,
        elevation  = 20
    )
    env.set_date(launch_time, timezone="Europe/Oslo")

    # --- plug in your custom atmosphere ---
    env.set_atmospheric_model(
        type        = "custom_atmosphere",
        pressure    = pressure_profile,
        temperature = temperature_profile,
        wind_u      = wind_u_profile,
        wind_v      = wind_v_profile
    )

    return env



# ─── Example ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lat, lon = 63.43, 10.3951
    # 1) fetch obs + forecast
    # 2) build a tz-aware datetime for “now”
    now = datetime.astimezone(ZoneInfo("Europe/Oslo"))
    # 3) construct Env
    env = construct_environment(lat, lon, now, "inputs/MC_env.nc")
    # 4) inspect
    env.plots.atmospheric_model()
