import json
from pathlib import Path

import pandas as pd, joblib
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent

SPLIT = "2026-08-01"      # Train on data before this date and evaluate on later dates.
CITY_CODES = {"Islamabad": 0, "Karachi": 1, "Lahore": 2}

df = pd.read_csv(SCRIPT_DIR / "dataset.csv", parse_dates=["date"])
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
    for H in (1, 2, 3):
        g[f"target{H}"] = g["pm25"].shift(-H)
    parts.append(g)

data = pd.concat(parts)
data.index.name = "date"
data = data.reset_index().dropna(subset=["pm25_today"])
data["city_code"] = data["city"].map(CITY_CODES)

FEATURES = ["pm25_today", "pm25_lag1", "pm25_lag2", "pm25_roll3", "pm25_roll7",
            "temp", "humidity", "wind_speed", "wind_u", "wind_v", "rain", "pressure",
            "wind_chg", "pressure_chg", "humidity_chg",
            "fires_100km", "fires_300km", "frp_300km",
            "fires300_3d", "fires300_7d", "frp300_3d", "month", "city_code"]

for H in (1, 2, 3):
    tr = data[(data["date"] < SPLIT) & data[f"target{H}"].notna()]
    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
    model.fit(
        tr[FEATURES],
        tr[f"target{H}"] - tr["pm25_today"],
    )  # Predict the change from today's PM2.5 level.
    joblib.dump(model, SCRIPT_DIR / f"ridge_h{H}.joblib")
    print(f"Saved ridge_h{H}.joblib | training rows: {len(tr)}")

joblib.dump(FEATURES, SCRIPT_DIR / "features_list.joblib")
data[["city", "date", "target1", "target2", "target3"] + FEATURES].to_csv(
    SCRIPT_DIR / "features.csv", index=False)
with (SCRIPT_DIR / "model_metadata.json").open("w", encoding="utf-8") as metadata_file:
    json.dump({"scikit_learn_version": sklearn.__version__}, metadata_file, indent=2)
print("Saved features.csv:", len(data), "rows")
print("Saved model_metadata.json for scikit-learn:", sklearn.__version__)