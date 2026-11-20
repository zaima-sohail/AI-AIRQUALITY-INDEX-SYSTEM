import pandas as pd

df = pd.read_csv("pm25_by_monitor.csv", parse_dates=["date"])
print("Shuru mein rows:", len(df))

# 1) Kam coverage wale din hata do (kam az kam aadhe din ka data)
df = df[df["coverage"] >= 50]

# 2) Be-tuki values hata do (PM2.5 negative ya 1000 se zyada nahi ho sakta)
df = df[(df["pm25"] >= 0) & (df["pm25"] < 1000)]

# 3) Shehar ka roz ka average, aur ye bhi ke kitne monitors se bana
daily = (df.groupby(["city", "date"])
           .agg(pm25=("pm25", "mean"), monitors=("pm25", "count"))
           .reset_index())

# 4) Kam az kam 3 monitors wale din rakho
daily = daily[daily["monitors"] >= 3].sort_values(["city", "date"])

daily.to_csv("pm25_clean.csv", index=False)
print(daily.groupby("city").agg(days=("date", "count"),
                                start=("date", "min"),
                                end=("date", "max"),
                                mean=("pm25", "mean"),
                                max=("pm25", "max")))
