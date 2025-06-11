# coord_utils.py
import math

def dms_to_dd(degrees: float, minutes: float, seconds: float) -> float:
    """Convert coordinates from degrees/minutes/seconds to decimal degrees.

    The sign of ``degrees`` determines the overall sign of the result.  This
    ensures that ``dms_to_dd(-1, 30, 0)`` correctly returns ``-1.5`` instead of
    ``-0.5``.
    """

    sign = -1 if degrees < 0 else 1
    return sign * (abs(degrees) + minutes / 60 + seconds / 3600)

def latlon_to_xy(lat_ref, lon_ref, coord):
    # Earth's radius in meters
    R = 6371000.0
    lat, lon = coord
    # Convert degrees to radians
    phi_ref = math.radians(lat_ref)
    lambda_ref = math.radians(lon_ref)
    phi = math.radians(lat)
    lambda_ = math.radians(lon)

    # Differences
    delta_phi = phi - phi_ref
    delta_lambda = lambda_ - lambda_ref

    # Use the average latitude to minimize distortion in the east-west distance
    phi_avg = (phi + phi_ref) / 2.0

    # Equirectangular approximation
    x = R * delta_lambda * math.cos(phi_avg)
    y = R * delta_phi
    xy = [x,y]
    return xy
