from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Air quality dashboard",
    page_icon=":material/air:",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
TRAINING_CUTOFF = pd.Timestamp("2026-08-01")


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(APP_DIR / "features.csv", parse_dates=["date"])


@st.cache_resource
def load_models() -> tuple[list[str], dict[int, Pipeline]]:
    feature_names: list[str] = joblib.load(APP_DIR / "features_list.joblib")
    models = {
        horizon: joblib.load(APP_DIR / f"ridge_h{horizon}.joblib")
        for horizon in (1, 2, 3)
    }
    return feature_names, models


FEATURES, MODELS = load_models()
features = load_features()

@dataclass(frozen=True)
class AirQualityGuidance:
   upper_limit: float
   name: str
   summary: str
   color: str
   actions: tuple[str, ...]


AIR_QUALITY_LEVELS = [
   AirQualityGuidance(
       9.0,
       "Good",
       "PM2.5 is in the lowest band used by this dashboard. Most people can continue their usual outdoor activities.",
       "green",
       ("Continue normal activities and follow local alerts if conditions change.",),
   ),
   AirQualityGuidance(
       35.4,
       "Moderate",
       "Most people can continue usual activities. People unusually sensitive to particle pollution may notice symptoms during prolonged, strenuous activity.",
       "blue",
       (
           "Most people can continue their usual outdoor routine.",
           "If you are sensitive to air pollution, reduce prolonged or strenuous outdoor activity if symptoms occur.",
       ),
   ),
   AirQualityGuidance(
       55.4,
       "Unhealthy for sensitive groups",
       "Sensitive groups may experience health effects. The general public is less likely to be affected at this level.",
       "orange",
       (
           "Children, older adults, pregnant people, and people with heart or lung conditions should reduce prolonged or strenuous outdoor activity.",
           "Move intense exercise indoors or reschedule it; take breaks and watch for symptoms.",
       ),
   ),
   AirQualityGuidance(
       125.4,
       "Unhealthy",
       "Health effects are possible for everyone, with greater risk for sensitive groups.",
       "red",
       (
           "Everyone should reduce prolonged or strenuous outdoor activity.",
           "Sensitive groups should avoid strenuous outdoor activity and spend more time in cleaner indoor air.",
           "If an outdoor trip is unavoidable, keep it brief. A well-fitting N95 or KN95 respirator may reduce particle exposure.",
       ),
   ),
   AirQualityGuidance(
       225.4,
       "Very unhealthy",
       "The risk of health effects is increased for everyone. Sensitive groups should take extra care.",
       "red",
       (
           "Avoid strenuous outdoor activity and limit time outdoors.",
           "Sensitive groups should stay in cleaner indoor air and follow their clinician-provided care plans.",
           "If going outside cannot be avoided, keep the trip brief; a well-fitting N95 or KN95 respirator may reduce particle exposure.",
       ),
   ),
   AirQualityGuidance(
       float("inf"),
       "Hazardous",
       "Health risk is very high at this concentration. Avoid exposure as much as possible.",
       "red",
       (
           "Avoid outdoor physical activity and stay in cleaner indoor air.",
           "Follow local public-health instructions and move school or work activities indoors where possible.",
           "If travel outside is unavoidable, minimize time outside. A well-fitting N95 or KN95 respirator may reduce particle exposure but cannot eliminate it.",
       ),
   ),
]

SENSITIVE_GROUPS = (
   "Children and teenagers; older adults; pregnant people; people with asthma, "
   "COPD, heart disease, or other lung conditions; and people who work outdoors."
)

INDOOR_AIR_TIPS = (
   "Keep windows and doors closed while outdoor pollution is high, if it is safe "
   "and comfortable to do so.",
   "If available, run a properly sized HEPA air cleaner in the room where people spend the most time.",
   "Avoid adding indoor particles from tobacco smoke, incense, candles, or cooking fumes.",
)


def air_quality_level(pm25: float) -> AirQualityGuidance:
   for guidance in AIR_QUALITY_LEVELS:
       if pm25 <= guidance.upper_limit:
           return guidance
   return AIR_QUALITY_LEVELS[-1]


def build_city_monitoring_table(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for city, city_data in data.groupby("city"):
        first_date = city_data["date"].min()
        latest_date = city_data["date"].max()
        expected_days = (latest_date - first_date).days + 1
        record_count = city_data["date"].nunique()
        records.append(
            {
                "City": city,
                "First record": first_date,
                "Latest record": latest_date,
                "Records": record_count,
                "Expected days": expected_days,
                "Missing days": expected_days - record_count,
                "Coverage": 100 * record_count / expected_days,
            }
        )
    return pd.DataFrame(records)


city_names = sorted(features["city"].dropna().unique())
with st.sidebar:
    st.title("Air quality")
    st.caption("Pakistan city forecasts and data monitoring")
    selected_city = st.selectbox("City", city_names)
    city_data = features[features["city"] == selected_city].sort_values("date")
    min_date = city_data["date"].min().date()
    max_date = city_data["date"].max().date()
    selected_date = st.date_input(
        "Forecast date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )
    trend_days = st.slider("Trend window (days)", min_value=7, max_value=90, value=30)
    st.caption("PM2.5 values are shown in µg/m³.")

selected_timestamp = pd.Timestamp(selected_date)
selected_rows = city_data[city_data["date"] == selected_timestamp]

st.title("Air quality dashboard")
st.caption(
    "Explore recent PM2.5 readings, short-range forecasts, and data coverage "
    "for Islamabad, Karachi, and Lahore."
)

if selected_rows.empty:
    st.warning(
        "There is no PM2.5 reading for this date. Choose another date from the sidebar.",
        title="No reading available",
    )
    st.stop()

selected_row = selected_rows.iloc[0]
today_pm25 = float(selected_row["pm25_today"])
feature_row = selected_row[FEATURES].to_frame().T.astype(float)
forecasts = {
    horizon: max(
        0.0,
        today_pm25 + float(MODELS[horizon].predict(feature_row)[0]),
    )
    for horizon in (1, 2, 3)
}
current_guidance = air_quality_level(today_pm25)

previous_rows = city_data[city_data["date"] < selected_timestamp]
previous_delta = None
if not previous_rows.empty:
    previous_delta = today_pm25 - float(previous_rows.iloc[-1]["pm25_today"])

with st.container(horizontal=True):
    st.metric(
        "Current PM2.5",
        f"{today_pm25:.1f} µg/m³",
        delta=f"{previous_delta:+.1f}" if previous_delta is not None else None,
        delta_description="versus previous reading",
        delta_color="inverse",
        border=True,
    )
    for horizon in (1, 2, 3):
        prediction = forecasts[horizon]
        level = air_quality_level(prediction)
        st.metric(
            f"Day +{horizon} forecast · {level.name}",
            f"{prediction:.1f} µg/m³",
            delta=f"{prediction - today_pm25:+.1f}",
            delta_description=f"versus {selected_date:%b %d}",
            delta_color="inverse",
            border=True,
        )

st.badge(
    f"Current air quality: {current_guidance.name}",
    color=current_guidance.color,
)
st.caption(
    f"Forecasts are for {selected_city} starting {selected_date:%B %d, %Y}. "
    "Positive changes indicate higher PM2.5."
)

overview_tab, health_tab, comparison_tab, monitoring_tab = st.tabs(
    ["Forecast and trends", "Health advisor", "City comparison", "Data monitoring"]
)

with overview_tab:
    st.subheader(f"{selected_city} PM2.5 trend")
    st.caption(
        f"Showing the {trend_days} days before the selected date, followed by "
        "the three-day forecast."
    )

    history_start = selected_timestamp - pd.Timedelta(days=trend_days)
    history = city_data[
        (city_data["date"] <= selected_timestamp)
        & (city_data["date"] >= history_start)
    ][["date", "pm25_today"]].rename(columns={"pm25_today": "Observed PM2.5"})
    forecast_dates = [
        selected_timestamp + pd.Timedelta(days=horizon)
        for horizon in (1, 2, 3)
    ]
    forecast_data = pd.DataFrame(
        {
            "date": forecast_dates,
            "Forecast PM2.5": [forecasts[horizon] for horizon in (1, 2, 3)],
        }
    )
    history_chart = history.rename(columns={"Observed PM2.5": "PM2.5"})
    history_chart["Series"] = "Observed PM2.5"
    forecast_chart = forecast_data.rename(columns={"Forecast PM2.5": "PM2.5"})
    forecast_chart["Series"] = "Forecast PM2.5"
    chart_data = [history_chart, forecast_chart]
    actual_values = [selected_row[f"target{horizon}"] for horizon in (1, 2, 3)]
    if any(pd.notna(value) for value in actual_values):
        actual_chart = pd.DataFrame(
            {
                "date": forecast_dates,
                "PM2.5": actual_values,
                "Series": "Actual PM2.5",
            }
        )
        chart_data.append(actual_chart.dropna(subset=["PM2.5"]))
    trend_chart = pd.concat(chart_data, ignore_index=True).sort_values("date")
    st.line_chart(
        trend_chart,
        x="date",
        y="PM2.5",
        color="Series",
        height=360,
    )

    worst_horizon = max(forecasts, key=forecasts.get)
    worst_guidance = air_quality_level(forecasts[worst_horizon])
    st.info(
        worst_guidance.summary,
        title=f"Health guidance for the forecast: {worst_guidance.name}",
    )
    st.caption("Open the Health advisor tab for tailored precautions and the full forecast outlook.")
    st.caption(
        "General guidance only, not medical advice. Models were trained on "
        f"data before {TRAINING_CUTOFF:%B %d, %Y}; later dates are out-of-sample."
    )

with health_tab:
    peak_horizon = max(forecasts, key=forecasts.get)
    peak_forecast = forecasts[peak_horizon]
    peak_date = selected_timestamp + pd.Timedelta(days=peak_horizon)
    peak_guidance = air_quality_level(peak_forecast)
    highest_exposure = max(today_pm25, peak_forecast)
    highest_guidance = air_quality_level(highest_exposure)
    highest_date = selected_timestamp if today_pm25 >= peak_forecast else peak_date

    st.subheader(f"Health advisor for {selected_city}")
    st.caption(
        "Advice is based on the selected day's PM2.5 reading and the next three "
        "daily forecasts. It is general guidance, not a medical assessment."
    )

    with st.container(horizontal=True):
        st.metric("Selected-day PM2.5", f"{today_pm25:.1f} µg/m³", border=True)
        st.metric(
            "Highest 3-day forecast",
            f"{peak_forecast:.1f} µg/m³",
            delta=f"{peak_forecast - today_pm25:+.1f}",
            delta_description=f"on {peak_date:%b %d}",
            delta_color="inverse",
            border=True,
        )

    current_column, forecast_column = st.columns(2)
    with current_column:
        with st.container(border=True):
            st.markdown("**Selected-day conditions**")
            st.badge(current_guidance.name, color=current_guidance.color)
            st.write(current_guidance.summary)
    with forecast_column:
        with st.container(border=True):
            st.markdown(f"**Peak forecast · {peak_date:%B %d}**")
            st.badge(peak_guidance.name, color=peak_guidance.color)
            st.write(peak_guidance.summary)

    st.subheader(f"Recommended actions for {highest_guidance.name.lower()}")
    st.caption(f"Based on the highest expected concentration on {highest_date:%B %d, %Y}.")
    for action in highest_guidance.actions:
        st.markdown(f"- {action}")

    st.subheader("Who may be more sensitive to particle pollution?")
    st.info(SENSITIVE_GROUPS, title="Groups to take extra care")

    indoor_column, symptoms_column = st.columns(2)
    with indoor_column:
        with st.container(border=True):
            st.markdown("**Reduce indoor exposure**")
            for tip in INDOOR_AIR_TIPS:
                st.markdown(f"- {tip}")
    with symptoms_column:
        with st.container(border=True):
            st.markdown("**Watch for symptoms**")
            st.write(
                "Irritated eyes or throat, coughing, wheezing, or unusual shortness "
                "of breath can be signs that pollution is affecting you. Reduce "
                "exposure and follow your existing care plan."
            )
            st.warning(
                "Seek urgent medical care for severe breathing difficulty, chest pain, or fainting.",
                title="Get help for severe symptoms",
            )

    st.subheader("Three-day health outlook")
    outlook = []
    for horizon in (1, 2, 3):
        value = forecasts[horizon]
        guidance = air_quality_level(value)
        outlook.append(
            {
                "Date": selected_timestamp + pd.Timedelta(days=horizon),
                "Forecast PM2.5": value,
                "Exposure band": guidance.name,
                "Recommended focus": guidance.actions[0],
            }
        )
    st.dataframe(
        pd.DataFrame(outlook),
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Forecast PM2.5": st.column_config.NumberColumn(
                "Forecast PM2.5", format="%.1f µg/m³"
            ),
        },
    )
    st.caption(
        "Bands are approximate PM2.5 concentration categories, not an official "
        "local AQI. Check local public-health alerts; forecast values are uncertain."
    )

with comparison_tab:
    st.subheader("Compare the latest city readings")
    latest_by_city = []
    for city, city_group in features.groupby("city"):
        city_group = city_group.sort_values("date")
        latest_row = city_group.iloc[-1]
        recent_start = latest_row["date"] - pd.Timedelta(days=6)
        recent_mean = city_group.loc[
            city_group["date"] >= recent_start, "pm25_today"
        ].mean()
        level = air_quality_level(float(latest_row["pm25_today"]))
        latest_by_city.append(
            {
                "City": city,
                "Latest reading": float(latest_row["pm25_today"]),
                "7-day average": float(recent_mean),
                "Latest date": latest_row["date"],
                "Air quality": level.name,
            }
        )
    comparison = pd.DataFrame(latest_by_city).sort_values("City")

    st.bar_chart(
        comparison,
        x="City",
        y="7-day average",
        height=320,
    )
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "Latest reading": st.column_config.NumberColumn(
                "Latest PM2.5", format="%.1f µg/m³"
            ),
            "7-day average": st.column_config.NumberColumn(
                "7-day average", format="%.1f µg/m³"
            ),
            "Latest date": st.column_config.DateColumn("Latest date"),
        },
    )

    trend_start = features["date"].max() - pd.Timedelta(days=trend_days)
    city_trends = features[features["date"] >= trend_start].pivot_table(
        index="date",
        columns="city",
        values="pm25_today",
        aggfunc="mean",
    )
    st.subheader(f"City trends over the last {trend_days} days")
    st.line_chart(
        city_trends,
        height=360,
        y_label="PM2.5 (µg/m³)",
    )

with monitoring_tab:
    monitoring = build_city_monitoring_table(features)
    latest_data_date = features["date"].max().normalize()
    days_since_latest = (pd.Timestamp.now().normalize() - latest_data_date).days

    if days_since_latest <= 3:
        freshness_label, freshness_color = "Fresh", "green"
    elif days_since_latest <= 7:
        freshness_label, freshness_color = "Delayed", "orange"
    else:
        freshness_label, freshness_color = "Stale", "red"

    st.subheader("Dataset health")
    st.badge(f"Data status: {freshness_label}", color=freshness_color)
    with st.container(horizontal=True):
        st.metric(
            "Latest observation",
            latest_data_date.strftime("%b %d, %Y"),
            border=True,
        )
        st.metric("Days since latest observation", days_since_latest, border=True)
        st.metric("Available records", f"{len(features):,}", border=True)
        st.metric("Cities monitored", f"{features['city'].nunique()}", border=True)

    st.caption(
        "Freshness describes the latest date in the local dataset. "
        "The app does not fetch or refresh source data automatically."
    )
    st.subheader("Daily record coverage by city")
    st.bar_chart(
        monitoring,
        x="City",
        y="Coverage",
        height=280,
        y_label="Coverage (%)",
    )
    st.dataframe(
        monitoring,
        hide_index=True,
        column_config={
            "First record": st.column_config.DateColumn("First record"),
            "Latest record": st.column_config.DateColumn("Latest record"),
            "Coverage": st.column_config.NumberColumn("Coverage", format="%.1f%%"),
        },
    )
