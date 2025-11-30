# Random Forest Regression / Severity Prediction for UW Visual Field Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # 1. Load UW visual field data
    vf_path = "data/UW_VF_Data.csv"

    vf_df = pd.read_csv(vf_path)
    print("VF data shape:", vf_df.shape)
    print("Columns:", vf_df.columns)

    # 2. Build per-eye baseline table
    #    Use PatID + Eye as unique eye identifier
    #    Baseline = earliest Time_from_Baseline per eye
    vf_df["EyeID"] = vf_df["PatID"].astype(str) + "_" + vf_df["Eye"].astype(str)

    baseline_vf = (
        vf_df
        .sort_values(["EyeID", "Time_from_Baseline"])
        .groupby("EyeID")
        .first()
        .reset_index()
    )

    print("Baseline VF shape:", baseline_vf.shape)

    # For convenience, call this merged
    merged = baseline_vf.copy()
    print("Merged shape:", merged.shape)

    # 3. Feature selection

    # Basic features
    basic_features = ["MS", "Age", "Time_from_Baseline"]

    # Pattern deviation features (PD_45 ... PD_54, etc.)
    pd_features = [c for c in merged.columns if c.startswith("PD_")]

    # MS cluster features
    cluster_features = [c for c in merged.columns if c.startswith("MS_Cluster")]

    feature_cols = basic_features + pd_features + cluster_features

    print("Using", len(feature_cols), "features.")

    # 4. Build model dataframe (drop missing)
    # NOTE: Dataset does not include MS_slope directly here, so we use
    # MS_Cluster3 as a proxy target (severity). You can swap this later
    # if you add a real slope column.
    target_col = "MS_Cluster3"

    model_df = merged.dropna(subset=feature_cols + [target_col])

    X = model_df[feature_cols].values
    y = model_df[target_col].values

    print("Model dataframe shape:", model_df.shape)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # 7. Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    results_path = "RandomForestRegressor/RFR_results.txt"
    with open(results_path, "w") as f:
        f.write("RandomForestRegressor – UW dataset\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2:  {r2:.4f}\n")

    print(f"Saved metrics to {results_path}")

    # 8. Feature importance plot
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]  # top 15

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(
        range(len(idx)),
        [feature_cols[i] for i in idx],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importances (UW VF baseline)")
    plt.tight_layout()

    fig_path = "RandomForestRegressor/RFR_feature_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved feature importance plot to {fig_path}")


if __name__ == "__main__":
    main()
