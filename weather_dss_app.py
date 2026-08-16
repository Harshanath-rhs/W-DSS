# Import required libraries and utilities

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

st.set_page_config(page_title="Sri Lanka Weather DSS", layout="wide", page_icon="🌦️")

ARTIFACT_DIR = "dss_artifacts"


# Load model and preprocessing artifacts

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        f"{ARTIFACT_DIR}/sri_lanka_weather_dss_model.keras", compile=False
    )
    scaler = joblib.load(f"{ARTIFACT_DIR}/weather_scaler_v3.pkl")
    district_encoder = joblib.load(f"{ARTIFACT_DIR}/district_encoder.pkl")
    config = joblib.load(f"{ARTIFACT_DIR}/feature_config.pkl")
    history_df = pd.read_csv(f"{ARTIFACT_DIR}/Sri_Lanka_weather_clean_v3.csv",
                              parse_dates=["time"])
    return model, scaler, district_encoder, config, history_df

model, scaler, district_encoder, config, history_df = load_artifacts()

STATIC_FEATURES = config["STATIC_FEATURES"]
DYNAMIC_REGRESSION_FEATURES = config["DYNAMIC_REGRESSION_FEATURES"]
ENCODER_FEATURES = config["ENCODER_FEATURES"]
MONSOON_COLS = config["MONSOON_COLS"]
RAIN_LABEL_NAMES = config["RAIN_LABEL_NAMES"]
LOOK_BACK = config["LOOK_BACK"]
HORIZON = config["HORIZON"]

DISTRICTS = sorted(history_df["district"].unique().tolist())



# monsoon phase, future known-features, forecast generation

def monsoon_phase(month):
    if month in (5, 6, 7, 8, 9):
        return "SW"
    elif month in (12, 1, 2):
        return "NE"
    return "INTER"


def build_future_static_row(date, lat, lon, elev):
    """Builds one row of STATIC_FEATURES (all deterministic/known) for a
    future calendar date. Sunrise/sunset are approximated from the same
    calendar day one year back for this district (they vary by only
    seconds year to year in Sri Lanka's latitude range)."""
    days_in_year = 366 if pd.Timestamp(date).is_leap_year else 365
    month_sin = np.sin(2 * np.pi * date.month / 12)
    month_cos = np.cos(2 * np.pi * date.month / 12)
    day_sin = np.sin(2 * np.pi * date.dayofyear / days_in_year)
    day_cos = np.cos(2 * np.pi * date.dayofyear / days_in_year)


    # Approximate sunrise/sunset hour from history - same day-of-year, any past year, same district, Takes overall mean if unavailable data.


    dist_hist = history_df[history_df["district"] ==
                            st.session_state.get("selected_district", DISTRICTS[0])]
    same_doy = dist_hist[dist_hist["time"].dt.dayofyear == date.dayofyear]
    if len(same_doy) > 0:
        sunrise_hour = same_doy["sunrise_hour"].mean()
        sunset_hour = same_doy["sunset_hour"].mean()
    else:
        sunrise_hour = dist_hist["sunrise_hour"].mean()
        sunset_hour = dist_hist["sunset_hour"].mean()
    daylight_hours = sunset_hour - sunrise_hour

    row = {
        "latitude": lat, "longitude": lon, "elevation": elev,
        "month_sin": month_sin, "month_cos": month_cos,
        "day_sin": day_sin, "day_cos": day_cos,
        "sunrise_hour": sunrise_hour, "sunset_hour": sunset_hour,
        "daylight_hours": daylight_hours,
    }
    phase = monsoon_phase(date.month)
    for col in MONSOON_COLS:
        row[col] = 1.0 if col == f"monsoon_{phase}" else 0.0
    return row


def run_forecast(district_name, override_rain, override_temp, override_wind,
                  override_windgusts, forecast_start_date):
    dist_id = int(district_encoder.transform([district_name])[0])
    dist_hist = history_df[history_df["district"] == district_name].sort_values("time")



    # Building the LOOK_BACK-day encoder window from real history days

    window = dist_hist.tail(LOOK_BACK).copy()
    if len(window) < LOOK_BACK:
        st.error(f"Not enough history for {district_name} (need {LOOK_BACK} days).")
        st.stop()

    # Override the today's row with the user manual input

    last_idx = window.index[-1]
    if override_rain is not None:
        window.loc[last_idx, "log_precipitation"] = np.log1p(override_rain)
    if override_temp is not None:


        # Shift mean/max/min/apparent together by the same delta 

        delta = override_temp - window.loc[last_idx, "temperature_2m_mean"]
        for c in ["temperature_2m_mean", "temperature_2m_max",
                  "temperature_2m_min", "apparent_temperature_mean"]:
            window.loc[last_idx, c] = window.loc[last_idx, c] + delta
    if override_wind is not None:
        window.loc[last_idx, "windspeed_10m_max"] = override_wind
    if override_windgusts is not None:
        window.loc[last_idx, "windgusts_10m_max"] = override_windgusts


    # Re-scale the window with the saved scaler.

    encoder_input_raw = window[ENCODER_FEATURES].values.copy()
    encoder_input_scaled = scaler.transform(encoder_input_raw)
    encoder_input_scaled = encoder_input_scaled.reshape(1, LOOK_BACK, len(ENCODER_FEATURES))

    # Building the HORIZON-day decoder known-future input

    lat = dist_hist["latitude"].iloc[-1]
    lon = dist_hist["longitude"].iloc[-1]
    elev = dist_hist["elevation"].iloc[-1]

    forecast_start_ts = pd.Timestamp(forecast_start_date)
    future_dates = [forecast_start_ts + timedelta(days=i) for i in range(HORIZON)]
    future_rows = [build_future_static_row(d, lat, lon, elev) for d in future_dates]
    decoder_static_df = pd.DataFrame(future_rows)[STATIC_FEATURES]

    # Scale static columns using the same fitted scaler

    dummy_full = pd.DataFrame(np.zeros((HORIZON, len(ENCODER_FEATURES))), columns=ENCODER_FEATURES)
    dummy_full[STATIC_FEATURES] = decoder_static_df.values
    dummy_scaled = scaler.transform(dummy_full.values)
    decoder_input_scaled = dummy_scaled[:, [ENCODER_FEATURES.index(c) for c in STATIC_FEATURES]]
    decoder_input_scaled = decoder_input_scaled.reshape(1, HORIZON, len(STATIC_FEATURES))

    district_arr = np.array([[dist_id]])

    # Single forward pass for all 30 days 

    y_pred_reg, y_pred_cls_probs = model.predict(
        [encoder_input_scaled, decoder_input_scaled, district_arr], verbose=0
    )

    # Inverse-transform regression outputs

    flat = y_pred_reg.reshape(-1, len(DYNAMIC_REGRESSION_FEATURES))
    full_width = np.zeros((flat.shape[0], len(ENCODER_FEATURES)))
    dyn_idx = [ENCODER_FEATURES.index(c) for c in DYNAMIC_REGRESSION_FEATURES]
    full_width[:, dyn_idx] = flat
    inv = scaler.inverse_transform(full_width)[:, dyn_idx]

    result = pd.DataFrame(inv, columns=DYNAMIC_REGRESSION_FEATURES)
    result["precipitation_mm"] = np.expm1(result["log_precipitation"]).clip(lower=0)
    result["date"] = pd.to_datetime(future_dates)
    result["rain_category"] = np.argmax(y_pred_cls_probs[0], axis=-1)
    result["rain_category_name"] = result["rain_category"].map(RAIN_LABEL_NAMES)
    result["confidence_pct"] = (y_pred_cls_probs[0].max(axis=-1) * 100).round(1)

    return result




# Recommendation engine 

def recommend_tea_planting(forecast):
    """Wijeratne et al. (2007); Chen et al. (2022); Carr (1972).
    Avoid if next fortnight shows drought <2mm/day sustained or heat >30C.
    Look for a recent/upcoming 20-50mm wetting window before recommending sowing."""
    fortnight = forecast.head(14)
    drought_days = (fortnight["precipitation_mm"] < 2.0).sum()
    heat_days = (fortnight["temperature_2m_max"] > 30.0).sum()
    conf = fortnight["confidence_pct"].mean()

    rolling_7d = forecast["precipitation_mm"].rolling(7, min_periods=7).sum()
    wetting_window = forecast[(rolling_7d >= 20) & (rolling_7d <= 50)]

    if heat_days >= 5:
        return ("⚠️ Avoid", f"{heat_days} of next 14 days exceed 30°C max temp "
                             "- heat stress risk to seedlings (Chen et al., 2022). "
                             f"Avg. model confidence over this window: {conf:.0f}%.")
    if drought_days >= 10:
        return ("⚠️ Avoid", f"{drought_days} of next 14 days under 2mm rainfall "
                             "- drought stress risk (Wijeratne et al., 2007). "
                             f"Avg. model confidence over this window: {conf:.0f}%.")
    if len(wetting_window) > 0:
        best = wetting_window.iloc[0]
        window_conf = forecast.loc[
            (forecast["date"] >= best["date"] - timedelta(days=6)) &
            (forecast["date"] <= best["date"]), "confidence_pct"
        ].mean()
        return ("✅ Suitable", f"7-day rainfall window of {rolling_7d.loc[best.name]:.0f}mm "
                                f"around {best['date']:%Y-%m-%d} matches the 20-50mm soil-wetting "
                                "requirement for sowing (Carr, 1972). "
                                f"Avg. model confidence over this window: {window_conf:.0f}%.")
    return ("⚠️ Caution", "No clear 20-50mm wetting window in the next 30 days; "
                           f"monitor before committing to sowing. "
                           f"Avg. model confidence over next 14 days: {conf:.0f}%.")


def recommend_infrastructure(forecast):
    """USACE (2008); ACI (2016). Dry weather, low precip probability,
    stable wind < 15-20 m/s (54-72 km/h), 0mm for concrete pours."""
    dry_mask = forecast["precipitation_mm"] < 1.0
    wind_ok = forecast["windgusts_10m_max"] < 54  # 15 m/s, conservative bound
    workable = dry_mask & wind_ok

    # Find the longest consecutive workable run

    runs = (~workable).cumsum()
    run_lengths = workable.groupby(runs).transform("sum") * workable
    best_run_len = int(run_lengths.max()) if len(run_lengths) else 0

    total_dry = int(dry_mask.sum())
    concrete_safe_days = forecast.loc[forecast["precipitation_mm"] < 0.2, "date"].dt.strftime("%Y-%m-%d").tolist()
    conf = forecast.loc[workable, "confidence_pct"].mean() if workable.any() else forecast["confidence_pct"].mean()

    if best_run_len >= 10:
        return ("✅ Suitable", f"Longest workable dry/stable-wind window is {best_run_len} "
                                f"consecutive days ({total_dry} dry days total in 30) "
                                "- adequate for earthwork and phased construction (USACE, 2008). "
                                f"Avg. model confidence on workable days: {conf:.0f}%.")
    if best_run_len >= 5:
        return ("⚠️ Caution", f"Best available window is only {best_run_len} consecutive "
                               "workable days - plan shorter work packages. "
                               f"Avg. model confidence on workable days: {conf:.0f}%.")
    return ("❌ High Risk", f"No workable window longer than {best_run_len} days found; "
                             "concrete pours should only proceed on the "
                             f"{len(concrete_safe_days)} near-zero-rainfall days "
                             "(ACI, 2016) with protection available. "
                             f"Avg. model confidence on workable days: {conf:.0f}%.")


def recommend_livestock(forecast):
    """Nardone et al. (2010); FAO (2011). Moderate temp, low rainfall,
    avoid prolonged wet conditions in the settling-in period after distribution."""
    settle_window = forecast.head(21)  
    heat_stress_days = (settle_window["temperature_2m_mean"] > 30.0).sum()
    wet_days = (settle_window["precipitation_mm"] > 5.0).sum()
    rolling_3d = settle_window["precipitation_mm"].rolling(3, min_periods=3).sum()
    prolonged_wet = (rolling_3d > 15).sum()
    conf = settle_window["confidence_pct"].mean()

    if heat_stress_days >= 7:
        return ("⚠️ Avoid", f"{heat_stress_days} of next 21 days show mean temp >30°C "
                             "- heat stress risk during the adaptation period (Nardone et al., 2010). "
                             f"Avg. model confidence over this window: {conf:.0f}%.")
    if prolonged_wet >= 3:
        return ("⚠️ Avoid", "Multiple 3-day windows exceed 15mm cumulative rainfall "
                             "- elevated respiratory/parasitic disease risk (FAO, 2011). "
                             f"Avg. model confidence over this window: {conf:.0f}%.")
    if wet_days <= 3:
        return ("✅ Suitable", f"Only {wet_days} moderately wet days expected in the "
                                "3-week adaptation window - favourable for distribution. "
                                f"Avg. model confidence over this window: {conf:.0f}%.")
    return ("⚠️ Caution", f"{wet_days} wet days expected in the adaptation window; "
                           f"ensure shelter is available. "
                           f"Avg. model confidence over this window: {conf:.0f}%.")


def recommend_training(forecast, target_date, search_window_days=7):
    """Wilkinson et al. (2018); WMO (2023). Community events need low
    rainfall probability; ranks candidate days in a window before a target date."""
    target_date = pd.Timestamp(target_date)
    candidates = forecast[
        (forecast["date"] >= target_date - timedelta(days=search_window_days)) &
        (forecast["date"] <= target_date + timedelta(days=search_window_days))
    ].copy()
    if candidates.empty:
        return None, "Target date is outside the 30-day forecast horizon."

    candidates["risk_score"] = (
        candidates["precipitation_mm"] + candidates["rain_category"] * 5
    )
    ranked = candidates.sort_values("risk_score")
    return ranked, None



# Streamlit User Interface

st.title("🌦️ Weather Decision Support System")
st.caption("Machine Learning-Based Weather Forecasting for "
           "Development Planning in Sri Lanka")

st.sidebar.header("🚩 Location & Today's Observations")
selected_district = st.sidebar.selectbox("District", DISTRICTS)
st.session_state["selected_district"] = selected_district

st.sidebar.markdown(
    "History for the past 60 days is loaded automatically. "
    "Only override today's readings below if you have them - "
    "otherwise the last recorded historical values are used."
)

use_override = st.sidebar.checkbox("Override today's observed weather", value=True)
override_rain = override_temp = override_wind = override_windgusts = None
if use_override:
    override_rain = st.sidebar.number_input("Today's rainfall (mm)", min_value=0.0, value=2.0, step=0.1)
    override_temp = st.sidebar.number_input("Today's mean temperature (°C)", min_value=10.0, value=27.0, step=0.5)
    override_wind = st.sidebar.number_input("Today's max wind speed (km/h)", min_value=0.0, value=15.0, step=1.0)
    override_windgusts = st.sidebar.number_input("Today's max wind gusts (km/h)", min_value=0.0, value=30.0, step=1.0)

forecast_start = st.sidebar.date_input("Forecast start date", value=datetime.now().date())

st.sidebar.divider()
target_training_date = st.sidebar.date_input(
    "Planned community training date",
    value=datetime.now().date() + timedelta(days=10)
)

generate = st.sidebar.button("Generate 30-Day Forecast & Recommendations", type="primary")

if generate:
    with st.spinner("Running 30-day forecast (single forward pass, no rollover)..."):
        forecast = run_forecast(
            selected_district, override_rain, override_temp,
            override_wind, override_windgusts, forecast_start
        )

    st.header(f"30-Day Outlook: {selected_district}")

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(forecast.set_index("date")[["precipitation_mm"]])
        st.caption("Daily rainfall forecast (mm)")
    with c2:
        st.line_chart(forecast.set_index("date")[["temperature_2m_max", "temperature_2m_min"]])
        st.caption("Daily temperature range forecast (°C)")

    st.divider()
    st.header("💡 Activity-Specific Recommendations")
    st.caption("Thresholds sourced from the literature review (Table 1: "
               "Summary of Weather Requirements for Development Activities)")

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("🍃 Tea Planting")
        status, reason = recommend_tea_planting(forecast)
        st.markdown(f"**{status}**")
        st.write(reason)

        st.subheader("🧑‍🏭 Infrastructure / Irrigation Construction")
        status, reason = recommend_infrastructure(forecast)
        st.markdown(f"**{status}**")
        st.write(reason)

    with r2:
        st.subheader("🐄 Livestock Distribution")
        status, reason = recommend_livestock(forecast)
        st.markdown(f"**{status}**")
        st.write(reason)

        st.subheader("👨🏻‍👩🏻‍👦🏻‍👦🏻 Community Training Day Finalization")
        ranked, err = recommend_training(forecast, target_training_date)
        if err:
            st.warning(err)
        else:
            best = ranked.iloc[0]
            st.markdown(f"**✅ Recommended date: {best['date']:%Y-%m-%d}**")
            st.write(f"Forecast: {best['precipitation_mm']:.1f}mm, "
                     f"{best['rain_category_name']} "
                     f"(model confidence: {best['confidence_pct']:.0f}%)")
            with st.expander("See all candidate dates, ranked"):
                st.dataframe(
                    ranked[["date", "precipitation_mm", "rain_category_name", "confidence_pct"]]
                    .rename(columns={
                        "date": "Date", "precipitation_mm": "Rainfall (mm)",
                        "rain_category_name": "Category",
                        "confidence_pct": "Confidence (%)"
                    }).reset_index(drop=True)
                )

    st.divider()
    with st.expander("⛅ Full 30-day forecast table"):
        st.caption(
            "Confidence = the model's softmax probability for its predicted "
            "category. Model skill decays with lead time (see the horizon-"
            "decay evaluation), so treat low-confidence / later-day rows as "
            "indicative rather than firm."
        )
        display_table = forecast[[
            "date", "precipitation_mm", "rain_category_name", "confidence_pct",
            "temperature_2m_max", "temperature_2m_min", "windspeed_10m_max"
        ]].rename(columns={
            "date": "Date", "precipitation_mm": "Rainfall (mm)",
            "rain_category_name": "Category", "confidence_pct": "Confidence (%)",
            "temperature_2m_max": "Max Temp (°C)",
            "temperature_2m_min": "Min Temp (°C)",
            "windspeed_10m_max": "Max Wind (km/h)"
        })
        st.dataframe(
            display_table.style.format({"Confidence (%)": "{:.1f}%"})
            .background_gradient(subset=["Confidence (%)"], cmap="RdYlGn", vmin=33, vmax=100)
            )

    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download 30-Day Forecast as CSV", data=csv,
        file_name=f"forecast_{selected_district}_{forecast_start}.csv",
        mime="text/csv"
    )
else:
    st.info("⬅️ Select a district, optionally override today's readings, and "
            "click Generate to produce the 30-day forecast and recommendations.")

st.write("---")
st.caption(f"MSc Data Science | Harshanath Senanayake | Kingston University, London {datetime.now().year}")
