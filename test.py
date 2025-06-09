import requests
import math

# —————— 1) KONFIGURASJON ——————
CLIENT_ID     = "nikolaiborbe@gmail.com:launch-server"
CLIENT_SECRET = "f826f38e-a74e-4f5a-a477-d73b1dbd7ca2"
SCOPE         = "api"
TOKEN_URL     = "https://id.barentswatch.no/connect/token"
BASE_API_URL  = "https://www.barentswatch.no/bwapi/v1"
MODEL_NAME    = "folda"   # du kan også prøve andre modeller som 'nordland', 'havområdet', etc.

# —————— 2) HENT TOKEN ——————
tok = requests.post(
    TOKEN_URL,
    data={
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope":         SCOPE
    },
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)
tok.raise_for_status()
token = tok.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# —————— 3) LIST OPP FAIRWAYS ——————
fw_resp = requests.get(
    f"{BASE_API_URL}/geodata/waveforecast/fairways",
    headers=headers,
    params={"modelname": MODEL_NAME}
)
fw_resp.raise_for_status()
fairways = fw_resp.json()    # Liste av fairway-objekter

print("Tilgjengelige fairways:")
for fw in fairways:
    print(f"  ID={fw['fairwayId']:2d}  Name={fw['name']}")
print()

# —————— 4) LA OSS VELGE EN FAIRWAY ——————
# Her velger vi den første som heter noe annet enn “Folda” (bare som eksempel),
# men du kan bytte ut dette med ID-en du ønsker.
if not fairways:
    raise RuntimeError("Fant ingen fairways for modellen!")
chosen = fairways[0]["fairwayId"]
print(f"→ Velger fairway ID={chosen}\n")

# —————— 5) HENT FORECAST-TIDSPUNKTER ——————
times_resp = requests.get(
    f"{BASE_API_URL}/geodata/waveforecast/available",
    headers=headers,
    params={"modelname": MODEL_NAME, "fairwayid": chosen}
)
times_resp.raise_for_status()
forecast_times = times_resp.json().get("forecasts", [])

print(f"Fant {len(forecast_times)} tider for fairway {chosen}. Første 5:")
for t in forecast_times[:5]:
    print("  •", t)
print()

# —————— 6) LOOP OG HENT REELLE BØLGEDATA ——————
g = 9.81
found_any = False

for t_iso in forecast_times:
    resp = requests.get(
        f"{BASE_API_URL}/geodata/waveforecast",
        headers=headers,
        params={
            "modelname":    MODEL_NAME,
            "fairwayid":    chosen,
            "forecastTime": t_iso
        }
    )
    if resp.status_code != 200 or not resp.text.strip():
        continue
    if "application/json" not in resp.headers.get("Content-Type", ""):
        continue

    data = resp.json()
    Hs = data.get("totalSignificantWaveHeight") or data.get("waveHeight")
    Tp = data.get("totalPeakPeriod") or data.get("peakWavePeriod")
    if not Hs or not Tp:
        continue

    # beregn Stokes-drift
    us = (math.pi**2 * Hs**2) / (g * Tp**3)
    print(f"{t_iso}: Hs={Hs:.2f} m, Tp={Tp:.1f} s → drift ≈ {us:.3f} m/s")
    found_any = True

if not found_any:
    print("Ingen reelle bølgedata funnet for denne fairwayen. Prøv en annen fairway-ID.")
