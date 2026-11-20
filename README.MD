# AI Air Quality Predictor

Forecasts **PM2.5 (smog) 1–3 days ahead** for **Lahore, Karachi and Islamabad**, explains
which factors push pollution up, and provides **English, general health guidance** in an
interactive Streamlit dashboard.

> **Live demo:** TODO (add your Streamlit link here)
> **Author:** TODO (your name) · [LinkedIn](TODO) · [GitHub](https://github.com/TODO)

![Dashboard screenshot](docs/dashboard.png)

---

## Why this project
Smog is a serious public-health problem in Pakistan, especially in Punjab during Oct–Jan.
Ground monitors are sparse and most people only see today's AQI. This project asks:

1. Can we predict the next 1–3 days of PM2.5 from open data?
2. Which factors (crop fires, wind, season) drive pollution in each city?
3. How can a forecast be turned into advice people can act on?

---

## Features
- **1, 2 and 3-day PM2.5 forecast** for three cities, compared against a "tomorrow = today" baseline
- **Explainability with SHAP**: estimated contribution of fires, wind, humidity/rain and cold-season conditions
- **Health advisor** with PM2.5-band guidance, sensitive groups, practical precautions,
  symptom warnings, and a three-day outlook
- **Data monitoring** for dataset freshness and daily record coverage by city
- **Interactive dashboard** built with Streamlit

---

## Data
All data is real and open; nothing is simulated.

| Source | What it provides | How it is used |
|---|---|---|
| [OpenAQ](https://openaq.org) | Ground-monitor PM2.5 | Target variable. Up to 5 long-running monitors per city, daily means, days with <50% hourly coverage removed, at least 3 monitors per day |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) (VIIRS) | Satellite fire detections | Fire counts and fire radiative power within 100 km and 300 km of each city (low-confidence detections removed) |
| [Open-Meteo](https://open-meteo.com) (ERA5-based archive) | Temperature, humidity, wind, rain, pressure | Daily weather features |

- **Period:** May 2025 – Nov 2026 (about 17 months)
- **Dataset size:** about 1,200 city-days (Lahore ≈ 510, Islamabad ≈ 420, Karachi ≈ 270)
- **Raw data is not included** (API keys required). The processed `dataset.csv` and `features.csv` are included so the dashboard runs out of the box.

---

## Method

```
OpenAQ  ─┐
NASA FIRMS ─┼─► clean + merge ─► dataset.csv ─► features ─► models ─► dashboard
Open-Meteo ─┘                                      │
                                                   └─► SHAP attribution
```

**Features:** today's PM2.5 and its recent history (lags, 3 and 7-day averages), weather and
its day-to-day change, fire counts within 100 and 300 km (plus rolling 3 and 7-day values), month, city.

**Target:** the *change* in PM2.5 over 1, 2 or 3 days (added back to today's value). This stops
the model from simply copying today's value.

**Validation:** time-based split. Trained on data before 2026-08-01 and tested on
Aug–Nov 2026, which includes the start of the 2026 smog season.

**Models compared:** Ridge regression, Random Forest, XGBoost.

---

## Results

Test period: Aug–Nov 2026. RMSE in µg/m³ (lower is better). Baseline = "tomorrow's PM2.5 equals today's".

**Ridge vs. baseline, by forecast horizon**

| Horizon | Baseline (all test days) | Ridge (all test days) | Baseline (Oct–Nov) | Ridge (Oct–Nov) |
|---|---|---|---|---|
| 1 day | 21.3 | 19.9 | 29.5 | 27.1 |
| 2 days | 27.1 | 23.6 | 37.5 | 32.0 |
| 3 days | 31.0 | 25.5 | 43.7 | 34.4 |

The model's advantage grows with the horizon: about **8% lower RMSE at 1 day and about 21% at 3 days**
during the Oct–Nov smog season.

**Model comparison, 1-day forecast (all test days)**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline | 21.3 | 12.9 | 0.78 |
| Ridge | 19.9 | 12.4 | 0.81 |
| Random Forest | 20.4 | 11.8 | 0.80 |
| XGBoost | 19.8 | 11.6 | 0.81 |

The three models perform similarly at 1 day, and the simple Ridge model generalised best at 3 days
(Oct–Nov RMSE 34.4 vs. 38.9 for XGBoost). With only one full smog season in training, simpler models
did better than more flexible ones. The dashboard therefore uses Ridge.

### What drives smog? (SHAP)
A separate model predicts same-day PM2.5 from **weather and fires only**, and SHAP splits each
prediction into contributions from factor groups. In Oct–Jan, for Lahore, crop-fire activity and
wind conditions raised predicted PM2.5 above the average across all cities, alongside a strong
cold-season signal (temperature, pressure, month).

Temperature and pressure were centred on each city's average, because raw values mostly identified
the *city* (for example, Islamabad's higher altitude means lower surface pressure).

> SHAP shows what the model relies on. It does **not** prove causation.

TODO: add `docs/shap_smog_season.png` and, if you like, the sanity-check R² of the SHAP model.

---

## Project structure

```
├── find_monitors.py      # find OpenAQ PM2.5 monitors near each city
├── download_pm25.py      # download daily PM2.5 per monitor
├── clean_pm25.py         # filter by coverage, average to city-level daily PM2.5
├── download_fires.py     # download NASA FIRMS fire detections
├── fill_fires.py         # fill date gaps in the fire data
├── build_dataset.py      # add weather + fire features, merge into dataset.csv
├── train_model.py        # first model and baseline
├── train_multi.py        # 1/2/3-day forecasts, Ridge vs XGBoost vs baseline
├── shap_explain.py       # SHAP attribution and charts
├── save_models.py        # train final models, save features.csv for the dashboard
├── app.py                # Streamlit dashboard
├── dataset.csv, features.csv, model_metadata.json, *.joblib
└── requirements.txt
```

---

## Run it

**Dashboard only (no API keys needed)**
```bash
git clone https://github.com/TODO/ai-air-quality-predictor.git
cd ai-air-quality-predictor
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python save_models.py
python -m streamlit run app.py
```

Run `save_models.py` and the dashboard with the same Python environment. The saved
scikit-learn models are version-sensitive; regenerate them with `python save_models.py`
after switching environments or changing the scikit-learn version. The dashboard checks
the recorded version before loading the models and reloads regenerated artifacts.

**Rebuild everything from raw data**
1. Create free API keys at [OpenAQ](https://explore.openaq.org) and [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/map_key).
2. Create a `.env` file in the project folder:
   ```
   OPENAQ_KEY=your_openaq_key
   FIRMS_KEY=your_firms_map_key
   ```
3. Run the scripts in order:
   ```bash
   python find_monitors.py
   python download_pm25.py
   python clean_pm25.py
   python download_fires.py
   python fill_fires.py
   python build_dataset.py
   python train_multi.py
   python save_models.py
   python shap_explain.py
   streamlit run app.py
   ```

---

## Limitations
- **Short history:** about 17 months, with only one full smog season in the training data.
- **Small test set:** about 200 city-days (about 84 in Oct–Nov), so small differences between models are not significant.
- **Karachi has fewer valid days** than Lahore and Islamabad, so its results are less reliable.
- **2–3 day forecasts use today's weather,** not forecast weather.
- **Single-city averages** of several monitors, so local hot spots are not captured.
- **SHAP contributions are estimates.** Temperature, pressure, season and fires are correlated
  in winter, so their separate effects are hard to untangle.
- **Model chosen on the test period.** The best model was picked using the same test set, so
  reported gains may be slightly optimistic.
- **Health advice is general guidance** based on common AQI recommendations (US EPA-style
  categories), not medical advice. The Urdu text has not been reviewed by a medical professional.

---

## Roadmap
- [ ] Add 2019–2025 US Embassy monitor data for more smog seasons
- [ ] Use forecast weather (Open-Meteo forecast API) for 2–3 day horizons
- [ ] Prediction intervals (uncertainty bands)
- [ ] Upwind-fire feature and wind back-trajectories (HYSPLIT)
- [ ] Hopsworks feature store and model registry
- [ ] Daily automatic data refresh (GitHub Actions)
- [ ] More cities (Faisalabad, Peshawar, Multan)

---

## Data attribution and licences
- Air quality data: [OpenAQ](https://openaq.org) and its data providers (check their terms before reuse)
- Fire data: NASA FIRMS / LANCE (NASA Earth Science Data)
- Weather data: [Open-Meteo](https://open-meteo.com) (CC BY 4.0, free for non-commercial use)

Code licence: TODO (e.g. MIT)

## Disclaimer
This is a student/portfolio project. Forecasts and advice are for information only and
should not replace official air-quality alerts or medical advice.