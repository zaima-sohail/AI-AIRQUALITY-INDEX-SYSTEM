import time, requests, pandas as pd

KEY = "a464ddbec95e9a5d1eac8ffef31ba80f5586596b90b8a18352e0e7cecc7b3f84"
H = {"X-API-Key": KEY}
CITIES = ["Lahore", "Karachi", "Islamabad"]
PER_CITY = 5

m = pd.read_csv("monitors.csv")
m["first"] = pd.to_datetime(m["first"], utc=True)
m["last"] = pd.to_datetime(m["last"], utc=True)
m["days"] = (m["last"] - m["first"]).dt.days

# Abhi chalne wale, lambe data wale monitors
m = m[(m["last"] >= "2026-11-01") & (m["name"] != "Test Device")]

frames = []
for city in CITIES:
    top = m[m["city"] == city].sort_values("days", ascending=False).head(PER_CITY)
    for _, row in top.iterrows():
        r = requests.get(
            f"https://api.openaq.org/v3/sensors/{int(row.sensor_id)}/days",
            headers=H,
            params={"datetime_from": "2025-05-01T00:00:00Z",
                    "datetime_to": "2026-11-16T00:00:00Z",
                    "limit": 1000},
        )
        print(city, row["name"], r.status_code)
        if r.status_code != 200:
            time.sleep(2); continue
        res = r.json().get("results", [])
        if res and not frames:
            print("Sample result:", res[0])   # field names dekhne ke liye
        for x in res:
            cov = (x.get("coverage") or {}).get("percentCoverage")
            frames.append({
                "city": city,
                "monitor": row["name"],
                "date": x["period"]["datetimeFrom"]["local"][:10],   # local tareekh
                "pm25": x["value"],
                "coverage": cov,
            })
        time.sleep(1.2)   # rate limit se bachne ke liye

df = pd.DataFrame(frames)
df.to_csv("pm25_by_monitor.csv", index=False)

print(df["coverage"].describe())
print(df["coverage"].value_counts(dropna=False).head(10))

daily = df.groupby(["city", "date"], as_index=False)["pm25"].mean()
daily.to_csv("pm25_daily.csv", index=False)
print(daily.groupby("city")["pm25"].agg(["count", "mean", "max"]))