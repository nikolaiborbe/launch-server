import requests
import numpy as np
import xarray as xr
import json
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from rocketpy import Environment
from models import Day, Data, Weather
from converters import ts_to_weather

# constant
g = 9.80665  # m/s²

USER_AGENT = "myApp/0.1 you@example.com"    # required by api.met.no
_ds_cache = {}

def _get_dataset(path: str) -> xr.Dataset:
    ds = _ds_cache.get(path)
    if ds is None:
        ds = xr.open_dataset(path)
        _ds_cache[path] = ds
    return ds


def select_forecasts(lat: float, lon: float, *, timeout=10) -> list[Weather]:
    """Return six Weather objects:
       [now, +1 h, +2 h, tomorrow 12:00, +2 d 12:00, +3 d 12:00]"""
    # 1 ─ fetch compact forecast
    url = (f"https://api.met.no/weatherapi/locationforecast/2.0/compact"
           f"?lat={lat}&lon={lon}")
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    timeseries = r.json()["properties"]["timeseries"]

    # 2 ─ helper to parse UTC timestamp once
    ts_dt = lambda item: datetime.fromisoformat(item["time"].replace("Z", "+00:00"))

    timeseries.sort(key=ts_dt)
    now_utc = datetime.now(ZoneInfo("UTC"))

    current = max(
        (ts for ts in timeseries if ts_dt(ts) <= now_utc),
        key=ts_dt,
        default=timeseries[0],
    )
    first_after = lambda moment: next(ts for ts in timeseries if ts_dt(ts) >= moment)

    plus1 = first_after(now_utc + timedelta(hours=1))
    plus2 = first_after(now_utc + timedelta(hours=2))

    oslo = ZoneInfo("Europe/Oslo")
    today_local = datetime.now(oslo).date()
    def noon(days_ahead):
        local_noon = datetime.combine(today_local + timedelta(days=days_ahead),
                                      dtime(12, 0), oslo)
        return first_after(local_noon.astimezone(ZoneInfo("UTC")))

    noon1, noon2, noon3 = noon(1), noon(2), noon(3)

    # 3 ────────────────────────────────  ### NEW
    # map the six raw JSON blobs → Weather dataclass instances
    return [ts_to_weather(ts) for ts in
            (current, plus1, plus2, noon1, noon2, noon3)]





def construct_environment(
    lat: float,
    lon: float,
    launch_time: datetime,
    climatology_file: str,
) -> tuple[list[Environment], list[Weather]]:
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
    weather_list: list[Weather] = select_forecasts(lat, lon)
    env_list = []
    
    # Load the climatology dataset only once (cached internally by _get_dataset)
    ds = _get_dataset(climatology_file)

    # numpy.datetime64 doesn't carry tz info.  Convert to UTC, drop tz,
    # then build the timestamp to avoid the warning.
    ts_utc = launch_time.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    ts = np.datetime64(ts_utc)

    # ------------------------------------------------------------------
    # The climatology slice at (ts, lat, lon) is identical for every
    # Weather snapshot, so we compute it once to avoid re‑allocating the
    # same arrays (height, pressure, base‑temperature) in each loop
    # iteration.  This saves both CPU and per‑request memory.
    # ------------------------------------------------------------------
    # Use float32 to halve memory footprint; precision is ±0.01 Pa / m, ample.
    levels_hpa   = clim["level"].values.astype(np.float32)   # [hPa]
    pressure_pa  = (levels_hpa * 100.0).astype(np.float32)   # → Pa

    geopot       = clim["z"].values.astype(np.float32)       # m²/s²
    height       = (geopot / g).astype(np.float32)           # m

    T_profile_base = clim["t"].values.astype(np.float32)      # K (climatology)

    # Pressure profile is invariant across weather_list entries
    pressure_profile = [
        (float(h), float(p))
        for h, p in zip(height, pressure_pa)
    ]

    for w in weather_list:
        T_obs   = w.temperature      # °C
        wspd    = w.wind_speed       # m/s
        wdir    = w.wind_from_direction  # deg from north

        # --- reuse the pre‑computed climatology slice ---
        T_profile = T_profile_base

        # --- apply surface‐temp anomaly (MET in °C → K) ---
        T_obs_K      = T_obs + 273.15
        T_surf_clim  = np.interp(0.0, height, T_profile)
        delta_T      = T_obs_K - T_surf_clim
        temp_arr = (T_profile_base + delta_T).astype(np.float32)
        temperature_profile = [
            (float(h), float(Ti)) for h, Ti in zip(height, temp_arr)
        ]
        del temp_arr  # free immediately

        # --- build constant wind profiles from the obs ---
        θ     = np.deg2rad(wdir)
        u0    = -wspd * np.sin(θ)
        v0    = -wspd * np.cos(θ)
        wind_u_profile = [(float(h), float(u0)) for h in height]
        wind_v_profile = [(float(h), float(v0)) for h in height]

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

        env_list.append(env)

    return (env_list, weather_list)



# ─── Example ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lat, lon = 63.43, 10.3951
    # 1) fetch obs + forecast
    # 2) build a tz-aware datetime for “now”
    now = datetime.now()
    # 3) construct Env
    #env, weather = construct_environment(lat, lon, now, "inputs/MC_env.nc")
    forecast: list[Weather] = select_forecasts(lat, lon)
    for i in range(len(forecast)):
        print(forecast[i].temperature)
    # 4) inspect
    #env.plots.atmospheric_model()
