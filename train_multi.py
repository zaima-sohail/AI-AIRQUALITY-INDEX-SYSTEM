import numpy as np, pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv("dataset.csv", parse_dates=["date"])

def build(H):
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
        g["target"] = g["pm25"].shift(-H)          # PM2.5 H days ahead
        parts.append(g)
    d = pd.concat(parts)
    d.index.name = "date"
    d = d.reset_index().dropna(subset=["target", "pm25_today"])
    d["city_code"] = d["city"].astype("category").cat.codes
    d["delta"] = d["target"] - d["pm25_today"]
    return d

FEATURES = ["pm25_today", "pm25_lag1", "pm25_lag2", "pm25_roll3", "pm25_roll7",
            "temp", "humidity", "wind_speed", "wind_u", "wind_v", "rain", "pressure",
            "wind_chg", "pressure_chg", "humidity_chg",
            "fires_100km", "fires_300km", "frp_300km",
            "fires300_3d", "fires300_7d", "frp300_3d", "month", "city_code"]

def rmse(y, p):
    return np.sqrt(mean_squared_error(y, p))

SPLIT = "2026-08-01"
print(f"{'Day':>4} {'Period':>10} {'Baseline':>9} {'Ridge':>7} {'XGBoost':>8}")
for H in [1, 2, 3]:
    data = build(H)
    train = data[data["date"] < SPLIT]
    test = data[data["date"] >= SPLIT]
    ridge = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
    xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    ridge.fit(train[FEATURES], train["delta"])
    xgb.fit(train[FEATURES], train["delta"])
    p_r = test["pm25_today"] + ridge.predict(test[FEATURES])
    p_x = test["pm25_today"] + xgb.predict(test[FEATURES])
    for name, mask in [("Overall", np.ones(len(test), bool)),
                       ("Oct-Nov", (test["date"] >= "2026-10-01").values)]:
        y = test["target"][mask]
        print(f"{H:>4} {name:>10} {rmse(y, test['pm25_today'][mask]):9.1f} "
              f"{rmse(y, p_r[mask]):7.1f} {rmse(y, p_x[mask]):8.1f}")

    shap_values = shap.TreeExplainer(xgb).shap_values(test[FEATURES])
    shap_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)
    ridge_coefficients = pd.Series(
        ridge.named_steps["ridge"].coef_, index=FEATURES
    )

    print(f"\n{H}-day forecast: top 10 XGBoost features by mean absolute SHAP")
    print("Feature                   Mean |SHAP|")
    for feature, importance in shap_importance.head(10).items():
        print(f"{feature:<25} {importance:>12.3f}")

    print(f"\n{H}-day forecast: top 10 Ridge coefficients (per standardized feature)")
    print("Feature                   Coefficient")
    for feature, coefficient in ridge_coefficients.abs().sort_values(
        ascending=False
    ).head(10).items():
        print(f"{feature:<25} {ridge_coefficients[feature]:>12.3f}")