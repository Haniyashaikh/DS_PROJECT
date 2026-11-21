"""
train_model.py  -  Train, evaluate and save the machine learning models
Run:  python train_model.py

What it does:
  1. Loads and cleans the data
  2. Splits it by TIME (first 80% = training, last 20% = testing)
  3. Trains Linear Regression and Random Forest to predict Zone 1
  4. Calculates MAE, RMSE and R2 for both
  5. Saves models and results inside the 'models' folder
     (the Streamlit app reads these files)
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import load_data, FEATURES, TARGET, MODELS_DIR

MODELS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = load_data()
print("Rows:", len(df))

X = df[FEATURES]
y = df[TARGET]

# ------------------------------------------------------------------
# 2. Chronological split (NO shuffling, because this is time-series data)
# ------------------------------------------------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("Training period:", df["DateTime"].iloc[0], "->",
      df["DateTime"].iloc[split_index - 1])
print("Testing period :", df["DateTime"].iloc[split_index], "->",
      df["DateTime"].iloc[-1])

# ------------------------------------------------------------------
# 3. Define models
# ------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,      # number of trees
        max_depth=15,          # how deep each tree can go
        min_samples_leaf=5,    # keeps the saved file small
        random_state=42,
        n_jobs=-1,             # use all CPU cores
    ),
}

# ------------------------------------------------------------------
# 4. Train + evaluate
# ------------------------------------------------------------------
results = []
test_predictions = pd.DataFrame({
    "DateTime": df["DateTime"].iloc[split_index:].values,
    "Actual": y_test.values,
})

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)                 # <-- this is the "training"
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"  MAE : {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R2  : {r2:.4f}")

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    test_predictions[name] = predictions

    # save the trained model
    file_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODELS_DIR / file_name, compress=3)

# ------------------------------------------------------------------
# 5. Save results for the Streamlit app
# ------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(MODELS_DIR / "metrics.csv", index=False)
test_predictions.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

# min / max / average of every input (used by the sliders in the app)
stats = {col: {"min": float(df[col].min()),
               "max": float(df[col].max()),
               "mean": float(df[col].mean())} for col in FEATURES}
with open(MODELS_DIR / "feature_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# feature importance of Random Forest
rf = models["Random Forest"]
importance = pd.DataFrame({"Feature": FEATURES,
                           "Importance": rf.feature_importances_})
importance = importance.sort_values("Importance", ascending=False)
importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

print("\n=== FINAL COMPARISON ===")
print(results_df.round(4).to_string(index=False))
best = results_df.sort_values("RMSE").iloc[0]["Model"]
print(f"\nBest model (lowest RMSE): {best}")
print("All files saved in the 'models' folder.")
