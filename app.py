"""
app.py  -  Streamlit dashboard
Run:  streamlit run app.py
(Run eda.py is optional, but train_model.py MUST be run first.)
"""

import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from common import (load_data, FEATURES, TARGET, ZONE_COLS, WEATHER_COLS,
                    MODELS_DIR, DAY_NAMES)

st.set_page_config(page_title="Electricity Consumption System",
                   page_icon="⚡", layout="wide")


# ------------------------------------------------------------------
# Cached loaders (so the app stays fast)
# ------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(file_name):
    return joblib.load(MODELS_DIR / file_name)


@st.cache_data
def get_stats():
    with open(MODELS_DIR / "feature_stats.json") as f:
        return json.load(f)


models_ready = (MODELS_DIR / "random_forest.joblib").exists()

try:
    df = get_data()
except Exception as error:
    st.error(str(error))
    st.stop()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("⚡ Electricity Analysis")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Consumption Trends",
    "Peak Analysis",
    "Weather Impact",
    "Prediction",
    "Model Comparison",
])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Power Consumption of Tetouan City (Morocco), UCI Machine Learning "
    "Repository, Salam & El Hibaoui (2018), CC BY 4.0. "
    "This project demonstrates analysis and prediction on this dataset; "
    "it does not predict electricity use in Karachi or Pakistan."
)

# ==================================================================
# PAGE 1: Overview
# ==================================================================
if page == "Overview":
    st.title("Electricity Consumption Analysis and Prediction System")
    st.write("Explore historical electricity consumption of three distribution "
             "zones of Tetouan City and predict Zone 1 consumption.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Average (Zone 1)", f"{df[TARGET].mean():,.0f}")
    c3.metric("Maximum (Zone 1)", f"{df[TARGET].max():,.0f}")
    c4.metric("Minimum (Zone 1)", f"{df[TARGET].min():,.0f}")

    st.caption(f"Data from {df['DateTime'].min():%d %b %Y} "
               f"to {df['DateTime'].max():%d %b %Y}, one record every 10 minutes.")

    st.subheader("Zone-wise summary")
    summary = df[ZONE_COLS].agg(["mean", "max", "min"]).T
    summary.columns = ["Average", "Maximum", "Minimum"]
    st.dataframe(summary.round(0))

    zone_avg = (df[ZONE_COLS].mean().rename("Average")
                .rename_axis("Zone").reset_index())
    fig = px.bar(zone_avg, x="Zone", y="Average",
                 title="Average consumption by zone")
    fig.update_layout(yaxis_title="Average consumption")
    st.plotly_chart(fig)

    st.subheader("Sample of the data")
    st.dataframe(df[["DateTime"] + WEATHER_COLS + ZONE_COLS].head(100))

# ==================================================================
# PAGE 2: Consumption Trends
# ==================================================================
elif page == "Consumption Trends":
    st.title("Consumption Trends")

    zones = st.multiselect("Select zones", ZONE_COLS, default=ZONE_COLS)
    view = st.radio("View", ["Hourly pattern", "Daily trend", "Monthly trend"],
                    horizontal=True)

    min_date = df["DateTime"].min().date()
    max_date = df["DateTime"].max().date()
    date_range = st.date_input("Date range", (min_date, max_date),
                               min_value=min_date, max_value=max_date)

    if len(date_range) != 2:
        st.info("Please select both a start and an end date.")
        st.stop()
    if not zones:
        st.info("Please select at least one zone.")
        st.stop()

    start, end = date_range
    data = df[(df["DateTime"].dt.date >= start) & (df["DateTime"].dt.date <= end)]

    if data.empty:
        st.warning("No data in the selected range.")
        st.stop()

    if view == "Hourly pattern":
        grouped = data.groupby("Hour")[zones].mean().reset_index()
        fig = px.line(grouped, x="Hour", y=zones, markers=True,
                      title="Average consumption by hour of the day")
    elif view == "Daily trend":
        grouped = data.groupby(data["DateTime"].dt.date)[zones].mean().reset_index()
        grouped = grouped.rename(columns={"DateTime": "Date"})
        fig = px.line(grouped, x="Date", y=zones, title="Daily average consumption")
    else:
        grouped = data.groupby(data["DateTime"].dt.to_period("M"))[zones].mean()
        grouped.index = grouped.index.astype(str)
        grouped = grouped.reset_index().rename(columns={"DateTime": "Month"})
        fig = px.bar(grouped, x="Month", y=zones, barmode="group",
                     title="Monthly average consumption")

    fig.update_layout(yaxis_title="Consumption")
    st.plotly_chart(fig)

# ==================================================================
# PAGE 3: Peak Analysis
# ==================================================================
elif page == "Peak Analysis":
    st.title("Peak and Low Consumption")

    zone = st.selectbox("Zone", ZONE_COLS)

    hourly = df.groupby("Hour")[zone].mean()
    peak_hour, low_hour = hourly.idxmax(), hourly.idxmin()
    c1, c2 = st.columns(2)
    c1.metric("Peak hour", f"{peak_hour}:00", f"{hourly.max():,.0f} average")
    c2.metric("Lowest hour", f"{low_hour}:00", f"{hourly.min():,.0f} average",
              delta_color="inverse")

    hourly_df = hourly.reset_index()
    hourly_df["Level"] = "Normal"
    hourly_df.loc[hourly_df[zone] >= hourly.quantile(0.75), "Level"] = "High"
    hourly_df.loc[hourly_df[zone] <= hourly.quantile(0.25), "Level"] = "Low"
    fig = px.bar(hourly_df, x="Hour", y=zone, color="Level",
                 color_discrete_map={"High": "#C44E52", "Normal": "#8DA0CB",
                                     "Low": "#55A868"},
                 title="Average consumption by hour (high / low periods)")
    st.plotly_chart(fig)

    dow = df.groupby("DayOfWeek")[zone].mean()
    dow.index = DAY_NAMES
    dow_df = dow.rename_axis("Day").reset_index()
    fig2 = px.bar(dow_df, x="Day", y=zone,
                  title="Average consumption by day of week")
    st.plotly_chart(fig2)

    daily = df.groupby(df["DateTime"].dt.date)[zone].mean().sort_values()
    col_a, col_b = st.columns(2)
    col_a.subheader("Top 10 highest days")
    col_a.dataframe(daily.tail(10).iloc[::-1].round(0).rename("Average"))
    col_b.subheader("Top 10 lowest days")
    col_b.dataframe(daily.head(10).round(0).rename("Average"))

# ==================================================================
# PAGE 4: Weather Impact
# ==================================================================
elif page == "Weather Impact":
    st.title("Weather Impact on Consumption")

    zone = st.selectbox("Zone", ZONE_COLS)
    weather = st.selectbox("Weather variable",
                           ["Temperature", "Humidity", "Wind Speed"])

    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(sample, x=weather, y=zone, opacity=0.4,
                     trendline="ols", trendline_color_override="red",
                     title=f"{weather} vs {zone}")
    st.plotly_chart(fig)

    corr_value = df[weather].corr(df[zone])
    st.info(f"Correlation between {weather} and {zone}: **{corr_value:.2f}** "
            "(close to +1 or -1 = strong relationship, close to 0 = weak).")

    st.subheader("Correlation heatmap")
    corr = df[WEATHER_COLS + ZONE_COLS].corr().round(2)
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig2)

# ==================================================================
# PAGE 5: Prediction
# ==================================================================
elif page == "Prediction":
    st.title("Predict Zone 1 Electricity Consumption")

    if not models_ready:
        st.warning("Models not found. Run `python train_model.py` first.")
        st.stop()

    stats = get_stats()
    model_name = st.selectbox("Model", ["Random Forest", "Linear Regression"])
    model_file = model_name.lower().replace(" ", "_") + ".joblib"
    model = get_model(model_file)

    st.write("Choose the conditions below. Sliders are limited to the range "
             "found in the dataset.")

    def slider(label, key, step=0.1):
        s = stats[key]
        return st.slider(label, s["min"], s["max"], s["mean"], step=step)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weather")
        temperature = slider("Temperature", "Temperature")
        humidity = slider("Humidity", "Humidity")
        wind = slider("Wind Speed", "Wind Speed", 0.01)
        general_diffuse = slider("General diffuse flows", "general diffuse flows", 1.0)
        diffuse = slider("Diffuse flows", "diffuse flows", 1.0)
    with col2:
        st.subheader("Time")
        hour = st.slider("Hour of the day", 0, 23, 12)
        month = st.slider("Month", 1, 12, 6)
        day = st.slider("Day of the month", 1, 31, 15)
        day_name = st.selectbox("Day of the week", DAY_NAMES)
        day_of_week = DAY_NAMES.index(day_name)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Temperature": temperature, "Humidity": humidity,
            "Wind Speed": wind, "general diffuse flows": general_diffuse,
            "diffuse flows": diffuse, "Hour": hour, "Day": day,
            "Month": month, "DayOfWeek": day_of_week,
        }])[FEATURES]                     # same column order as training
        prediction = float(model.predict(row)[0])

        # --- compare the prediction with the historical data of Zone 1 ---
        average = df[TARGET].mean()
        lowest, highest = df[TARGET].min(), df[TARGET].max()
        low_limit = df[TARGET].quantile(0.25)    # below this = low usage
        high_limit = df[TARGET].quantile(0.75)   # above this = high usage
        difference = (prediction - average) / average * 100

        month_name = pd.Timestamp(2017, month, 1).strftime("%B")
        situation = (f"{hour:02d}:00, {day_name}, {month_name} "
                     f"(temperature {temperature:.1f}, humidity {humidity:.0f})")

        st.subheader("Prediction Result")
        st.write(f"**Situation:** {situation}")

        col_a, col_b = st.columns(2)
        col_a.metric("Predicted consumption (Zone 1)", f"{prediction:,.0f}",
                     f"{difference:+.1f}% compared to average",
                     delta_color="inverse")
        col_b.metric("Normal (average) consumption", f"{average:,.0f}")

        # --- explain the result in simple words ---
        if prediction >= high_limit:
            st.error("🔴 **HIGH usage expected.** Electricity demand in this "
                     "situation is higher than usual, so this is a busy "
                     "(peak) period.")
        elif prediction <= low_limit:
            st.success("🟢 **LOW usage expected.** Electricity demand in this "
                       "situation is lower than usual, so this is a quiet "
                       "period.")
        else:
            st.info("🟡 **NORMAL usage expected.** Electricity demand in this "
                    "situation is close to the usual level.")

        # --- progress bar: where does this value sit in the data? ---
        position = (prediction - lowest) / (highest - lowest)
        st.progress(min(max(position, 0.0), 1.0),
                    text="Position between the lowest and the highest "
                         "consumption in the dataset")
        st.caption(f"Lowest in data: {lowest:,.0f}  |  "
                   f"Highest in data: {highest:,.0f}  |  "
                   "Values are in the same units as the dataset. "
                   "This is an estimate made by a machine learning model.")

# ==================================================================
# PAGE 6: Model Comparison
# ==================================================================
elif page == "Model Comparison":
    st.title("Model Comparison")

    if not (MODELS_DIR / "metrics.csv").exists():
        st.warning("Results not found. Run `python train_model.py` first.")
        st.stop()

    metrics = pd.read_csv(MODELS_DIR / "metrics.csv")
    st.subheader("Evaluation on the test set")
    st.dataframe(metrics.round(4))
    best = metrics.sort_values("RMSE").iloc[0]["Model"]
    st.success(f"Best model (lowest RMSE): **{best}**")
    st.caption("MAE and RMSE: lower is better. R²: closer to 1 is better.")

    c1, c2, c3 = st.columns(3)
    for col, metric in zip((c1, c2, c3), ("MAE", "RMSE", "R2")):
        col.plotly_chart(px.bar(metrics, x="Model", y=metric, color="Model",
                                title=metric))

    st.subheader("Actual vs Predicted (test period)")
    preds = pd.read_csv(MODELS_DIR / "test_predictions.csv",
                        parse_dates=["DateTime"])
    days = st.slider("Show the last N days of the test period", 1, 30, 7)
    recent = preds.tail(days * 144)       # 144 records = 1 day
    fig = px.line(recent, x="DateTime",
                  y=["Actual", "Linear Regression", "Random Forest"])
    fig.update_layout(yaxis_title="Zone 1 consumption")
    st.plotly_chart(fig)

    if (MODELS_DIR / "feature_importance.csv").exists():
        st.subheader("Which inputs matter most? (Random Forest)")
        imp = pd.read_csv(MODELS_DIR / "feature_importance.csv")
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature",
                               orientation="h"))