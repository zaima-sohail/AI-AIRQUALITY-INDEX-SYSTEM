import pandas as pd

FIRMS_KEY = "4323049029bf99d3cd2ee67042075e5a"
BBOX = "66,23,80,36"   # west,south,east,north (Pakistan + Punjab/Haryana)

start = pd.Timestamp("2025-05-01")
end = pd.Timestamp("2026-11-15")
parts = []

d = start
while d <= end:
    # purana data SP mein, taaza data NRT mein hota hai
    source = "VIIRS_SNPP_SP" if d < pd.Timestamp("2026-07-01") else "VIIRS_SNPP_NRT"
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{FIRMS_KEY}/{source}/{BBOX}/5/{d.date()}")
    try:
        part = pd.read_csv(url)
        print(d.date(), source, len(part))
        parts.append(part)
    except Exception as e:
        print(d.date(), "ERROR", e)
    d += pd.Timedelta(days=5)

fires = pd.concat(parts, ignore_index=True).drop_duplicates()
fires.to_csv("fires_raw.csv", index=False)
print(fires.shape)