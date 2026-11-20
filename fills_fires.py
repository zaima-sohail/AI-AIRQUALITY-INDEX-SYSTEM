import pandas as pd

FIRMS_KEY = "4323049029bf99d3cd2ee67042075e5a"
BBOX = "66,23,80,36"

# (shuru, aakhir, source)
WINDOWS = [
    ("2025-07-03", "2025-07-09", "VIIRS_SNPP_SP"),
    ("2026-03-08", "2026-03-14", "VIIRS_SNPP_SP"),
    ("2026-04-26", "2026-05-02", "VIIRS_SNPP_SP"),
    ("2026-05-30", "2026-06-04", "VIIRS_SNPP_SP"),
    ("2026-07-01", "2026-08-05", "VIIRS_SNPP_SP"),
    ("2026-10-30", "2026-11-17", "VIIRS_SNPP_NRT"),
]

parts = [pd.read_csv("fires_raw.csv", low_memory=False)]
for start, end, source in WINDOWS:
    d, stop = pd.Timestamp(start), pd.Timestamp(end)
    while d <= stop:
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
               f"{FIRMS_KEY}/{source}/{BBOX}/5/{d.date()}")
        try:
            part = pd.read_csv(url, low_memory=False)
            print(d.date(), source, len(part))
            parts.append(part)
        except Exception as e:
            print(d.date(), source, "ERROR", e)
        d += pd.Timedelta(days=5)

fires = pd.concat(parts, ignore_index=True).drop_duplicates()
fires.to_csv("fires_raw.csv", index=False)

days = pd.to_datetime(fires["acq_date"]).dt.date
full = pd.date_range("2025-05-01", "2026-11-15").date
missing = [d for d in full if d not in set(days)]
print("Total rows:", len(fires), "| Abhi bhi gayab din:", len(missing))
print(missing)