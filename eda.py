"""
eda.py  -  Exploratory Data Analysis
Run:  python eda.py
It prints basic information in the console and saves all charts
as PNG images inside the 'outputs' folder (use them in your report).
"""

import matplotlib
matplotlib.use("Agg")            # save charts as files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns

from common import (load_data, OUTPUT_DIR, ZONE_COLS, WEATHER_COLS,
                    TARGET, DAY_NAMES)

OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def save(fig_name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fig_name, dpi=120)
    plt.close()
    print("saved ->", OUTPUT_DIR / fig_name)


df = load_data()

# ------------------------------------------------------------------
# 1. Basic understanding of the data
# ------------------------------------------------------------------
print("\n=== SHAPE (rows, columns) ===")
print(df.shape)
print("\n=== DATA TYPES ===")
print(df.dtypes)
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())
print("\n=== DUPLICATE ROWS (already removed in load_data) ===")
print(df.duplicated().sum())
print("\n=== STATISTICS ===")
print(df.describe().T)
print("\n=== DATE RANGE ===")
print(df["DateTime"].min(), "->", df["DateTime"].max())

# ------------------------------------------------------------------
# 2. Outlier check (boxplots). We only LOOK at outliers, we do not delete
#    them, because high electricity usage can be a real peak.
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[ZONE_COLS])
plt.xticks(rotation=10)
plt.title("Boxplot of electricity consumption (outlier check)")
save("01_boxplot_zones.png")

# ------------------------------------------------------------------
# 3. Hourly pattern (average per hour of the day)
# ------------------------------------------------------------------
hourly = df.groupby("Hour")[ZONE_COLS].mean()
hourly.plot(figsize=(10, 5), marker="o")
plt.title("Average consumption by hour of the day")
plt.xlabel("Hour")
plt.ylabel("Consumption")
save("02_hourly_pattern.png")
print("\nPeak hour (Zone 1):", hourly[TARGET].idxmax(),
      "| Lowest hour (Zone 1):", hourly[TARGET].idxmin())

# ------------------------------------------------------------------
# 4. Daily trend
# ------------------------------------------------------------------
daily = df.groupby(df["DateTime"].dt.date)[ZONE_COLS].mean()
daily.plot(figsize=(12, 5))
plt.title("Daily average consumption")
plt.xlabel("Date")
plt.ylabel("Consumption")
save("03_daily_trend.png")

# ------------------------------------------------------------------
# 5. Monthly trend
# ------------------------------------------------------------------
monthly = df.groupby("Month")[ZONE_COLS].mean()
monthly.plot(kind="bar", figsize=(10, 5))
plt.title("Monthly average consumption")
plt.xlabel("Month")
plt.ylabel("Consumption")
save("04_monthly_trend.png")
print("Peak month (Zone 1):", monthly[TARGET].idxmax(),
      "| Lowest month (Zone 1):", monthly[TARGET].idxmin())

# ------------------------------------------------------------------
# 6. Day of week
# ------------------------------------------------------------------
dow = df.groupby("DayOfWeek")[ZONE_COLS].mean()
dow.index = DAY_NAMES
dow.plot(kind="bar", figsize=(10, 5))
plt.title("Average consumption by day of week")
plt.xticks(rotation=30)
save("05_day_of_week.png")

# ------------------------------------------------------------------
# 7. Zone comparison
# ------------------------------------------------------------------
totals = df[ZONE_COLS].mean()
print("\nAverage consumption per zone:\n", totals)
totals.plot(kind="bar", figsize=(7, 5), color=["#4C72B0", "#55A868", "#C44E52"])
plt.title("Average consumption: zone comparison")
plt.xticks(rotation=10)
save("06_zone_comparison.png")

# ------------------------------------------------------------------
# 8. Weather vs consumption (scatter plots, sampled so they are fast)
# ------------------------------------------------------------------
sample = df.sample(5000, random_state=42) if len(df) > 5000 else df
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["Temperature", "Humidity", "Wind Speed"]):
    sns.scatterplot(data=sample, x=col, y=TARGET, ax=ax, alpha=0.3, s=10)
    ax.set_title(f"{col} vs Zone 1 consumption")
save("07_weather_scatter.png")

# ------------------------------------------------------------------
# 9. Correlation heatmap
# ------------------------------------------------------------------
corr = df[WEATHER_COLS + ZONE_COLS].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
save("08_correlation_heatmap.png")
print("\nCorrelation with Zone 1:\n",
      corr[TARGET].sort_values(ascending=False))

print("\nEDA finished. Check the 'outputs' folder for charts.")
