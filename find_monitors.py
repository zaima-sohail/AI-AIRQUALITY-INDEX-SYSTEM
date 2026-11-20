import requests, pandas as pd

KEY = "a464ddbec95e9a5d1eac8ffef31ba80f5586596b90b8a18352e0e7cecc7b3f84"  
H = {"X-API-Key": KEY}

CITIES = {
    "Lahore": (31.52, 74.36), "Karachi": (24.86, 67.01),
    "Islamabad": (33.68, 73.05), "Peshawar": (34.01, 71.58),
    "Faisalabad": (31.42, 73.08), "Multan": (30.20, 71.47),
}

rows = []
for city, (lat, lon) in CITIES.items():
    r = requests.get(
        "https://api.openaq.org/v3/locations",
        headers=H,
        params={"coordinates": f"{lat},{lon}", "radius": 25000,
                "parameters_id": 2, "limit": 100},
    )
    r.raise_for_status()
    for loc in r.json()["results"]:
        sensors = [s for s in loc.get("sensors", [])
                   if s["parameter"]["name"] == "pm25"]
        rows.append({
            "city": city,
            "location_id": loc["id"],
            "name": loc["name"],
            "sensor_id": sensors[0]["id"] if sensors else None,
            "first": (loc.get("datetimeFirst") or {}).get("utc"),
            "last": (loc.get("datetimeLast") or {}).get("utc"),
        })

df = pd.DataFrame(rows)
print(df.to_string())
df.to_csv("monitors.csv", index=False)