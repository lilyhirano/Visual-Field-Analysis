# progression_model.py


"""
Student-written script for modeling glaucoma progression.
Loads the cleaned UW dataset and trains a simple regression model
to estimate MS_slope (dB/year).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# cleaned CSV produced by data_cleaning.py
RAW_PATH = "data/uw_vf_clean.csv"


def main():

    # 1. Check dataset
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Could not find dataset at {RAW_PATH}")

    print("Loading cleaned dataset from:", RAW_PATH)
    df = pd.read_csv(RAW_PATH)

    # make sure progression target exists
    if "MS_slope" not in df.columns:
        raise ValueError("MS_slope not found. Did data_cleaning.py compute it?")

    # 
    # 2. Feature extraction
    feature_cols = [c for c in df.columns if c.startswith(("PD_", "TD_", "Sens_"))]

    X = df[feature_cols]
    y = df["MS_slope"]

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train model
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Evaluate model
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.4f}")

    print("Finished progression model.")


if __name__ == "__main__":
    main()
