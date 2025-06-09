import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Sett inn din Frost API client_id
CLIENT_ID = "2f65a0ee-37fc-4c91-9362-3ec909f18687"

# Stasjon for Hitra (Sandstad)
STATION_ID = "SN71550"

# Hvilken måned og hvilket år?
month = int(input("Hvilken måned i tall? \n"))     # For eksempel 9
year = str(input("Hvilket år? \n"))               # For eksempel 2023

# Neste måned (for slutt-dato):
# OBS: Denne enkle varianten håndterer ikke desember -> januar,
#      men fungerer for 1 <= month < 12
next_month = month + 1

# Bruk "to-sifret" format (02d) for måned, slik at 9 -> '09'
start_date = f"{year}-{month:02d}-01"       # for eksempel '2023-09-01'
end_date   = f"{year}-{next_month:02d}-01"  # for eksempel '2023-10-01'

# URL til Frost API for observasjoner
endpoint = "https://frost.met.no/observations/v0.jsonld"

# Parametere: Hent vindstyrke og vindretning
parameters = {
    "sources": STATION_ID,
    "elements": "wind_speed,wind_from_direction",
    "referencetime": f"{start_date}/{end_date}"
}

# Gjør forespørsel til Frost API
response = requests.get(endpoint, params=parameters, auth=(CLIENT_ID, ""))

# Sjekk statuskode
if response.status_code != 200:
    print(f"Feil ved forespørsel til Frost API. Statuskode: {response.status_code}")
    print(response.text)
    raise SystemExit

# Konverter responsen til JSON
data = response.json()

# Sjekk at vi har fått data
if "data" not in data:
    print("Ingen data funnet i svaret, sjekk parametrene dine.")
    raise SystemExit

# Konvertering til Pandas DataFrame
records = []
for obs in data["data"]:
    ref_time = obs.get("referenceTime")  # Tidsstempel
    for measurement in obs.get("observations", []):
        element_id = measurement.get("elementId")
        value = measurement.get("value")
        records.append((ref_time, element_id, value))

df = pd.DataFrame(records, columns=["time", "element", "value"])

# Pivotér data slik at 'wind_speed' og 'wind_from_direction' blir kolonner
df_pivot = df.pivot_table(index="time", columns="element", values="value").reset_index()

# Konverter "time" til datetime
df_pivot["time"] = pd.to_datetime(df_pivot["time"])

# Gi kolonnene mer forståelige navn
df_pivot.rename(
    columns={
        "wind_speed": "wind_speed_m_s",
        "wind_from_direction": "wind_dir_deg"
    },
    inplace=True
)

# Filtrer kun for den måneden brukeren ba om
df_month = df_pivot[df_pivot["time"].dt.month == month]

# Ta kun ut tidene mellom kl 07-19 (heltime)
df_month_hourly = df_month[
    #(df_month["time"].dt.hour >= 7) &
    #(df_month["time"].dt.hour <= 19) &
    (df_month["time"].dt.hour == 12)&
    (df_month["time"].dt.minute == 0)
].copy()

# Filtrer på threshold-verdi for vindhastighet
threshold = int(input("Threshold på vind (m/s): \n"))
df_month_hourly_filtered = df_month_hourly[df_month_hourly["wind_speed_m_s"] < threshold].copy()

# Filtrer vindretning innenfor ±10° fra 315°
dir_center = 180
dir_tolerance = 110
dir_min = dir_center - dir_tolerance
dir_max = dir_center + dir_tolerance

df_month_hourly_filtered = df_month_hourly_filtered[
    (df_month_hourly_filtered["wind_dir_deg"] >= dir_min) &
    (df_month_hourly_filtered["wind_dir_deg"] <= dir_max)
]

# Sjekk om datasettet er tomt etter filtrering:
if df_month_hourly_filtered.empty:
    print("Ingen målinger tilfredsstiller kriteriene for vindhastighet og retning.")
    # Avslutt eller ta en beslutning
    raise SystemExit

# Gruppér etter dato for histogram
df_month_hourly_filtered["date"] = df_month_hourly_filtered["time"].dt.date

# --- Skrive til en .txt-fil ---
# Her sorterer vi etter tid først, så skriver vi hver rad til filen
df_month_hourly_filtered.sort_values("time", inplace=True)

with open("filtrerte_tider.txt", "w", encoding="utf-8") as f:
    for idx, row in df_month_hourly_filtered.iterrows():
        f.write(
            f"{row['time']} | "
            f"vindhastighet: {row['wind_speed_m_s']} m/s | "
            f"vindretning: {row['wind_dir_deg']}°\n"
        )

# Plot histogram
plt.figure(figsize=(10, 5))
df_month_hourly_filtered["date"].value_counts().sort_index().plot(
    kind="bar",
    color="blue",
    edgecolor="black"
)
plt.xlabel("Dato")
plt.ylabel("Antall målinger (maks 13)")
plt.title(f"Antall målinger < {threshold} m/s og vindretning {dir_min}°–{dir_max}°\n"
          f"kl. 07-19, {month:02d}-{year}")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
