from dataclasses import dataclass
from datetime import datetime

@dataclass(slots=True)
class FlightData:
    time: list
    coords: list

@dataclass(slots=True)
class Data:
    max_velocity: float
    apogee_time: float
    apogee_altitude: float
    apogee_x: float
    apogee_y: float
    impact_x: float
    impact_y: float
    impact_velocity: float
    flight_data: FlightData


@dataclass(slots=True)
class Weather:
    time: datetime
    temperature: float
    pressure: float
    wind_speed: float
    wind_from_direction: float
    humidity: float

@dataclass(slots=True)
class Day:
    data: Data
    weather: Weather
