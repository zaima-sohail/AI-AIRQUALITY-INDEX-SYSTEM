import numpy as np, pandas as pd, requests

CITIES = {"Lahore": (31.52, 74.36), "Karachi": (24.86, 67.01),
          "Islamabad": (33.68, 73.05)}
START, END = "2025-05-01", "2026-11-10"   # mausam API mein thora delay hota hai

# ---------- 1) Mausam ----------
weather = []
for city, (lat, lon) in CITIES.items():
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                  "wind_direction_10m,precipitation,surface_pressure",
        "timezone": "Asia/Karachi"})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    rad = np.deg2rad(h["wind_direction_10m"])
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(rad)   # hawa ka east-west hissa
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(rad)   # hawa ka north-south hissa
    d = h.set_index("time").resample("D").agg({
        "temperature_2m": "mean", "relative_humidity_2m": "mean",
        "wind_speed_10m": "mean", "wind_u": "mean", "wind_v": "mean",
        "precipitation": "sum", "surface_pressure": "mean"}).reset_index()
    d.columns = ["date", "temp", "humidity", "wind_speed", "wind_u", "wind_v",
                 "rain", "pressure"]
    d["city"] = city
    weather.append(d)
    print("Mausam", city, len(d))
weather = pd.concat(weather)

# ---------- 2) Fires ----------
f = pd.read_csv("fires_raw.csv", low_memory=False)
f = f[f["confidence"] != "l"]            # kam bharose wali detections hata do
f["date"] = pd.to_datetime(f["acq_date"])
flat, flon = np.deg2rad(f["latitude"].values), np.deg2rad(f["longitude"].values)

rows = []
for city, (lat, lon) in CITIES.items():
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    a = (np.sin((flat - la) / 2) ** 2
         + np.cos(la) * np.cos(flat) * np.sin((flon - lo) / 2) ** 2)
    dist = 6371 * 2 * np.arcsin(np.sqrt(a))          # km
    g = f.assign(dist=dist)
    out = pd.DataFrame({
        "fires_100km": g[g.dist <= 100].groupby("date").size(),
        "fires_300km": g[g.dist <= 300].groupby("date").size(),
        "frp_300km": g[g.dist <= 300].groupby("date")["frp"].sum(),
    })
    out = out.reindex(pd.date_range(START, END)).fillna(0)
    out.index.name = "date"
    out = out.reset_index()
    out["city"] = city
    rows.append(out)
fires = pd.concat(rows)

# ---------- 3) Sab jodo ----------
pm = pd.read_csv("pm25_clean.csv", parse_dates=["date"])
data = (pm.merge(weather, on=["city", "date"], how="inner")
          .merge(fires, on=["city", "date"], how="left"))
data.to_csv("dataset.csv", index=False)

print(data.shape)
print(data.isna().sum())
print(data.groupby("city").agg(days=("date", "count"), pm25=("pm25", "mean"),
                               fires300=("fires_300km", "mean")))
print(data.head())