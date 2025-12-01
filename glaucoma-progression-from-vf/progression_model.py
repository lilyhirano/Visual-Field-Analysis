# progression model

"""
progression_model.py

Estimate glaucoma progression as MS slope (dB/year) for each eye
and train a regression model to predict that slope from baseline
PD maps.

Outputs:
- models/ms_slope_rf.pkl
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

DATA_PATH = "data/uw_vf.csv"
MODEL_PATH = "models/ms_slope_rf.pkl"


def compute_ms_slope(group: pd.DataFrame) -> float:
    """Fit MS ~ time and return slope (dB/year)."""
    # need at least 2 time points to get a slope
    if group["Time_from_Baseline"].nunique() < 2:
        return np.nan

    x = group["Time_from_Baseline"].values
    y = group["MS"].values

    # simple linear fit y = a*x + b
    a, b = np.polyfit(x, y, 1)
    return a


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")
    os.makedirs("models", exist_ok=True)

    vf = pd.read_csv(DATA_PATH)
    pd_cols = [c for c in vf.columns if c.startswith("PD_")]

    # define a (patient, eye) key
    vf["key"] = vf["PatID"].astype(str) + "_" + vf["Eye"].astype(str)

    # compute slope per key
    slopes = vf.groupby("key").apply(compute_ms_slope).rename("MS_slope")
    slopes = slopes.dropna()
    print("Number of eyes with valid slope:", len(slopes))

    # select baseline rows (time_from_baseline == 0)
    baseline = vf[vf["Time_from_Baseline"] == 0].copy()
    baseline["key"] = baseline["PatID"].astype(str) + "_" + baseline["Eye"].astype(str)

    # join baseline features with slopes
    data_prog = baseline.merge(slopes, on="key", how="inner")
    print("Rows with baseline + slope:", data_prog.shape)

    X = data_prog[pd_cols].values
    y = data_prog["MS_slope"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=0,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("R^2:", r2)
    print("RMSE (dB/year):", rmse)

    # save model
    joblib.dump(rf, MODEL_PATH)
    print("Saved progression model to:", MODEL_PATH)

    # simple true vs predicted scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.axline((0, 0), (1, 1), color="red", linestyle="--")
    plt.xlabel("True MS slope (dB/year)")
    plt.ylabel("Predicted MS slope (dB/year)")
    plt.title("Glaucoma progression model")
    plt.show()


if __name__ == "__main__":
    main()
