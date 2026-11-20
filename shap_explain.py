import numpy as np, pandas as pd, shap, matplotlib
matplotlib.use("Agg")                       # save charts to files, no popup window
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

CENTER = True                               # True: subtract each city's mean; False: raw values
CITY, DATE = "Lahore", "2025-11-05"         # day to explain
SPLIT = "2026-08-01"
TAG = "centered" if CENTER else "raw"

# ---------- 1) Load data and build features ----------
df = pd.read_csv("dataset.csv", parse_dates=["date"])
parts = []
for city, g in df.groupby("city"):
    g = g.set_index("date")
    g = g.reindex(pd.date_range(g.index.min(), g.index.max()))
    g["city"] = city
    g["fires300_3d"] = g["fires_300km"].rolling(3, min_periods=1).mean()
    g["fires300_7d"] = g["fires_300km"].rolling(7, min_periods=1).mean()
    g["month"] = g.index.month
    parts.append(g)
data = pd.concat(parts)
data.index.name = "date"
data = data.reset_index().dropna(subset=["pm25"]).reset_index(drop=True)
data["city_code"] = data["city"].astype("category").cat.codes

# Optional: remove city-level differences (e.g. Islamabad is at higher altitude,
# so its surface pressure is always lower than Lahore's)
if CENTER:
    for c in ["temp", "pressure", "humidity"]:
        data[c] = data[c] - data.groupby("city")[c].transform("mean")

# Weather + fires only (no PM2.5 history), so SHAP explains the level of PM2.5
GROUPS = {
    "Fires": ["fires_100km", "fires_300km", "frp_300km", "fires300_3d", "fires300_7d"],
    "Wind": ["wind_speed", "wind_u", "wind_v"],
    "Humidity & rain": ["humidity", "rain"],
    "Cold-season conditions (temp, pressure, month)": ["temp", "pressure", "month"],
    "City baseline": ["city_code"],
}
FEATS = [f for fs in GROUPS.values() for f in fs]

def make_model():
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)

# ---------- 2) Sanity check: does the model generalise to unseen days? ----------
train, test = data[data["date"] < SPLIT], data[data["date"] >= SPLIT]
chk = make_model().fit(train[FEATS], train["pm25"])
print(f"[{TAG}] Sanity check, R2 on unseen days:",
      round(r2_score(test["pm25"], chk.predict(test[FEATS])), 2))

# ---------- 3) Fit on all data and compute SHAP ----------
model = make_model().fit(data[FEATS], data["pm25"])
exp = shap.TreeExplainer(model)(data[FEATS])
sv = exp.values

# Chart 1: feature importance overview
plt.figure()
shap.summary_plot(sv, data[FEATS], show=False)
plt.tight_layout()
plt.savefig(f"shap_summary_{TAG}.png", dpi=150)
plt.close()

# Per-day impact of each factor group
grp = pd.DataFrame({g: sv[:, [FEATS.index(f) for f in fs]].sum(axis=1)
                    for g, fs in GROUPS.items()})
grp["city"] = data["city"].values

# Chart 2: mean absolute impact per city (size of effect, no sign)
city_share = grp.groupby("city")[list(GROUPS)].agg(lambda s: s.abs().mean())
print(f"\n[{TAG}] Mean absolute impact on PM2.5 (µg/m³):\n", city_share.round(1))
ax = city_share.plot(kind="bar", stacked=True, figsize=(8, 5))
ax.set_ylabel("Mean absolute impact (µg/m³)")
ax.set_xlabel("City")
ax.set_title(f"Estimated impact on PM2.5 by city (SHAP, {TAG})")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"shap_city_groups_{TAG}.png", dpi=150)
plt.close()

# Chart 3: smog season only (Oct-Jan), with sign
smog_season = data["month"].isin([10, 11, 12, 1]).values
signed = grp[smog_season].groupby("city")[list(GROUPS)].mean()
print(f"\n[{TAG}] Smog season (Oct-Jan): mean signed impact (µg/m³):\n", signed.round(1))
ax = signed.plot(kind="bar", figsize=(8, 5))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean impact on PM2.5 (µg/m³)")
ax.set_xlabel("City")
ax.set_title(f"Estimated drivers of smog, Oct-Jan (SHAP, {TAG})")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"shap_smog_season_{TAG}.png", dpi=150)
plt.close()

# ---------- 4) Explain one day ----------
mask = (data["city"] == CITY) & (data["date"] == pd.Timestamp(DATE))
if not mask.any():
    i = data[data["city"] == CITY]["pm25"].idxmax()
    print(f"\n{DATE} not found; using the highest-PM2.5 day for {CITY} instead.")
else:
    i = data.index[mask][0]
base = float(exp.base_values[i])
print(f"\n{data.loc[i, 'city']} | {data.loc[i, 'date'].date()}")
print(f"Actual PM2.5: {data.loc[i, 'pm25']:.0f} | Model estimate: {base + sv[i].sum():.0f} "
      f"| Average day: {base:.0f}")
print(grp.loc[i, list(GROUPS)].sort_values(key=abs, ascending=False).round(1).to_string())

shap.plots.waterfall(exp[i], max_display=12, show=False)
plt.savefig(f"shap_day_{TAG}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: shap_summary_{TAG}.png, shap_city_groups_{TAG}.png, "
      f"shap_smog_season_{TAG}.png, shap_day_{TAG}.png")