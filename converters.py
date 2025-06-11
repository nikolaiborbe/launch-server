# ──────────────────────────────────────────────────────────────
# converters.py
# ──────────────────────────────────────────────────────────────
from datetime import datetime
from typing import TypedDict
from models import Weather

class _Ts(TypedDict):      # narrow JSON type just for autocompletion
    time: str
    data: dict

def ts_to_weather(ts: _Ts) -> Weather:
    """Convert one Locationforecast timeseries element to Weather."""
    # 1) parse the ISO-8601 time stamp (always given in UTC = Zulu)
    t_utc = datetime.fromisoformat(ts["time"].replace("Z", "+00:00"))

    # 2) extract the five scalar values from the 'instant→details' dict
    d = ts["data"]["instant"]["details"]
    return Weather(
        time=t_utc,
        temperature=d["air_temperature"],
        pressure=d["air_pressure_at_sea_level"],
        wind_speed=d["wind_speed"],
        wind_from_direction=d["wind_from_direction"],
        humidity=d["relative_humidity"],
    )
