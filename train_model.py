import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("dataset.csv", parse_dates=["date"])

parts = []
for city, g in df.groupby("city"):
    g = g.set_index("date")
    g = g.reindex(pd.date_range(g.index.min(), g.index.max()))
    g["city"] = city
    g["pm25_today"] = g["pm25"]
    g["pm25_lag1"] = g["pm25"].shift(1)
    g["pm25_lag2"] = g["pm25"].shift(2)
    g["pm25_roll3"] = g["pm25"].rolling(3, min_periods=2).mean()
    g["pm25_roll7"] = g["pm25"].rolling(7, min_periods=4).mean()
    g["fires300_3d"] = g["fires_300km"].rolling(3, min_periods=1).mean()
    g["fires300_7d"] = g["fires_300km"].rolling(7, min_periods=1).mean()
    g["frp300_3d"] = g["frp_300km"].rolling(3, min_periods=1).mean()
    g["wind_chg"] = g["wind_speed"].diff()
    g["pressure_chg"] = g["pressure"].diff()
    g["humidity_chg"] = g["humidity"].diff()
    g["month"] = g.index.month
    g["target"] = g["pm25"].shift(-1)
    parts.append(g)
data = pd.concat(parts)
data.index.name = "date"
data = data.reset_index().dropna(subset=["target", "pm25_today"])
data["city_code"] = data["city"].astype("category").cat.codes
data["delta"] = data["target"] - data["pm25_today"]      # kal aaj se kitna badla

FEATURES = ["pm25_today", "pm25_lag1", "pm25_lag2", "pm25_roll3", "pm25_roll7",
            "temp", "humidity", "wind_speed", "wind_u", "wind_v", "rain", "pressure",
            "wind_chg", "pressure_chg", "humidity_chg",
            "fires_100km", "fires_300km", "frp_300km",
            "fires300_3d", "fires300_7d", "frp300_3d", "month", "city_code"]

SPLIT = "2026-08-01"
train = data[data["date"] < SPLIT]
test = data[data["date"] >= SPLIT]
print("Train rows:", len(train), "| Test rows:", len(test))

def report(name, y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    print(f"{name:26s} RMSE={rmse:6.1f}  MAE={mean_absolute_error(y, p):6.1f}  R2={r2_score(y, p):5.2f}")

models = {
    "Ridge (delta)": make_pipeline(SimpleImputer(), StandardScaler(), Ridge()),
    "RandomForest (delta)": make_pipeline(SimpleImputer(),
                                          RandomForestRegressor(300, min_samples_leaf=3, random_state=42)),
    "XGBoost (delta)": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42),
}

preds = {}
for name, m in models.items():
    m.fit(train[FEATURES], train["delta"])
    preds[name] = test["pm25_today"].values + m.predict(test[FEATURES])  # aaj + badlao

def evaluate(title, mask):
    print("\n==", title, "| rows:", int(mask.sum()))
    y = test["target"][mask]
    report("Baseline (kal = aaj)", y, test["pm25_today"][mask])
    for name, p in preds.items():
        report(name, y, p[mask])

evaluate("Poora test (Aug-Nov 2026)", np.ones(len(test), bool))
evaluate("Smog season (Oct-Nov 2026)", (test["date"] >= "2026-10-01").values)

# Behtareen model ka shehar-wise natija
best = min(preds, key=lambda n: mean_squared_error(test["target"], preds[n]))
print("\nBehtareen model:", best)
for city in test["city"].unique():
    m = (test["city"] == city).values
    report("  " + city, test["target"][m], preds[best][m])