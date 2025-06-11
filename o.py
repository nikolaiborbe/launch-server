from weather import select_forecasts
import json

lat, lon = 63.43, 10.3951
curr = select_forecasts(lat, lon)
print(json.dumps(curr[4], indent=2, ensure_ascii=False))
